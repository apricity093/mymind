"""
无外部依赖的前端契约 E2E 服务。

用途：在 Redis/Chroma/LLM 不可用的本地环境，启动真实 mymind FastAPI 路由处理器
（/health、/search、/knowledge/add、/knowledge/upload、/knowledge/stats、/chat），
仅把数据库和模型组件替换为内存实现。它验证的是 HTTP 契约与前端工作流，
不替代真实 Redis/Chroma/LLM 的性能与功能验收。

启动：
  D:\\anaconda3\\envs\\learn_claude\\python.exe tests\\e2e_contract_server.py --port 8000
"""
from __future__ import annotations

import argparse
import re
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

from api.main import (
    ChatResponse,
    KnowledgeContextResult,
    _background_tasks,
    _build_knowledge_context,
    _context_builder,
    _knowledge_base,
    _knowledge_policy,
    _last_context_metadata,
    _memory,
    _monitor,
    _orchestrator,
    _skill_manager,
    _tool_manager,
    app,
)
from core.intent_recognizer import IntentCategory, IntentResult, UrgencyLevel
from core.knowledge_policy import KnowledgePolicy
from core.retrieval import Chunker, variant_config
from agents.agent_orchestrator import AgentType


class InMemoryKnowledge:
    """内存知识库：Markdown 感知切块、稳定 chunk_id、简单相关性检索。"""

    def __init__(self):
        self._chunker = Chunker(variant_config("r2"))
        self._chunks: List[Dict[str, Any]] = []

    @property
    def index_version(self):
        return "rag-index-v1"

    @property
    def chunk_config(self):
        return {"chunk_mode": "markdown", "chunk_size": 500, "overlap": 80}

    @property
    def doc_count(self):
        return len(self._chunks)

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        before = len(self._chunks)
        for doc in documents:
            for record in self._chunker.chunk_document(doc.get("title", ""), doc.get("content", "")):
                self._chunks.append({
                    "title": record.title,
                    "content": record.text,
                    "score": 0.0,
                    "chunk": record.chunk_index,
                    "chunk_id": record.chunk_id,
                    "source_id": record.source_id,
                    "section_path": record.section_path,
                    "index_version": self.index_version,
                    "chunk_config": self.chunk_config,
                })
        return len(self._chunks) - before

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_tokens = set(_tokens(query))
        scored = []
        for item in self._chunks:
            text_tokens = set(_tokens(str(item["content"])))
            overlap = len(query_tokens & text_tokens)
            section_tokens = set(_tokens(str(item["section_path"])))
            section_overlap = len(query_tokens & section_tokens)
            score = 0.55 + 0.25 * overlap + 0.20 * section_overlap
            scored.append((score, dict(item)))
        scored.sort(key=lambda row: (-row[0], row[1]["chunk_id"]))
        return [{**item, "score": round(score, 4), "retrieval_sources": ["vector"]}
                for score, item in scored[:max(0, top_k)]]

    async def search_handler(self, params, context):
        return self.search(params.get("query", ""), params.get("top_k", 5))


def _tokens(text: str) -> List[str]:
    latin = re.findall(r"[a-z0-9]+", str(text).casefold())
    cjk = re.findall(r"[\u4e00-\u9fff]", str(text))
    return latin + cjk + [a + b for a, b in zip(cjk, cjk[1:])]


class InMemoryToolManager:
    def __init__(self, knowledge: InMemoryKnowledge):
        self._knowledge = knowledge

    async def search_with_rewrite(self, tool_name, query, top_k=5):
        results = self._knowledge.search(query, top_k=max(5, top_k))[:top_k]
        return SimpleNamespace(data=results, reranked=True, degraded=False, success=True, error=None)

    def invalidate_cache(self):
        return 0


class E2EOrchestrator:
    def get_stats(self):
        return {"agents": 3}

    async def recognize_intent(self, message, history=None):
        return IntentResult(
            intent=IntentCategory.REFUND,
            confidence=0.9,
            urgency=UrgencyLevel.LOW,
            intent_group="billing",
            entities={"order_id": ["#12345"]},
            reasoning="refund",
            latency_ms=1.0,
            source_scores={"pattern": 0.9},
        )

    async def run(self, request):
        return SimpleNamespace(
            response="已根据知识库为您处理退款：提交后 1-3 个工作日审核，通过后 5-7 个工作日原路退回。",
            intent=IntentCategory.REFUND,
            agent_type=AgentType.BILLING,
            agent_types=[AgentType.BILLING],
            primary_agent=AgentType.BILLING,
            supporting_agents=[],
            routing_reason="intent=refund",
            routing_confidence=0.9,
            escalated=False,
            latency_ms=3.0,
        )


class E2EMemory:
    async def get_context(self, user_id, conv_id, query=None):
        return SimpleNamespace(recent_messages=[])

    async def add_message(self, *args):
        return None

    async def update_profile(self, *args):
        return None


class E2EMonitor:
    def summary(self):
        return {"agents": 3, "context": dict(_last_context_metadata)}


async def _used_knowledge(message, intent, top_k=3):
    return KnowledgeContextResult(
        text="[知识库检索结果]",
        used=True,
        status="used",
        reason="intent:refund",
    )


@asynccontextmanager
async def _noop_lifespan(application):
    yield


def install():
    import api.main as api_module

    knowledge = InMemoryKnowledge()
    manager = InMemoryToolManager(knowledge)
    api_module._orchestrator = E2EOrchestrator()
    api_module._memory = E2EMemory()
    api_module._tool_manager = manager
    api_module._knowledge_base = knowledge
    api_module._knowledge_policy = KnowledgePolicy()
    api_module._context_builder = SimpleNamespace(build=lambda *args: SimpleNamespace(text="ctx", metadata={"e2e": True}))
    api_module._build_knowledge_context = _used_knowledge
    api_module._monitor = E2EMonitor()
    app.router.lifespan_context = _noop_lifespan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    install()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
