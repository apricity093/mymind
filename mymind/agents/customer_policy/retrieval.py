"""复用生产原子检索；诊断与策略可见材料分别保存。"""
from __future__ import annotations

import time
from dataclasses import asdict

from agents.customer_policy.protocol import KnowledgeSnippet, SearchObservation, ToolStatus


class PolicyRetrieval:
    def __init__(self, knowledge, manager, *, candidates=12, top_k=5, observation_chars=16000):
        self.knowledge, self.manager = knowledge, manager
        self.candidates, self.top_k = candidates, top_k
        self.observation_chars = observation_chars

    async def search(self, query, number):
        start = time.monotonic()
        diagnostics = {}
        result = await self.manager.call("knowledge_search", {"query": query, "top_k": self.candidates},
                                         {"retrieval_diagnostics": diagnostics}, use_cache=False,
                                         rerank_top_k=self.top_k)
        diagnostics.update(asdict(result))
        snippets = []
        state = "unavailable" if not result.success or result.error else "empty"
        degraded = (result.degraded or diagnostics.get("index_route") == "backup" or
                    result.diagnostics.get("rerank_status") in {"failed_fallback", "skipped_budget"})
        if state != "unavailable":
            used = 0
            for item in result.data or []:
                # 故障提示或缺正文的返回项不能成为政策依据。
                if not item.get("source_id") or not item.get("content"):
                    continue
                snippet = KnowledgeSnippet(reference_id=f"search:{number}:{len(snippets)+1}",
                    source_id=item["source_id"], title=item.get("title", ""),
                    section_path=item.get("section_path", "").split("/") if item.get("section_path") else [],
                    content=item["content"])
                size = len(snippet.model_dump_json())
                if used + size > self.observation_chars:
                    continue
                snippets.append(snippet)
                used += size
            state = "candidates" if snippets else "empty"
        observation = SearchObservation(request_number=number,
            status=ToolStatus(reference_id=f"search:{number}:status", state=state, degraded=degraded,
                              detail="检索服务未返回可用正文" if state == "unavailable" else ""),
            snippets=snippets)
        diagnostics.update(total_latency_ms=(time.monotonic()-start)*1000,
                           visible_observation=observation.model_dump(), cached=result.cached)
        return observation, diagnostics


def build_retrieval(root, config, gateway):
    import chromadb
    from chromadb.config import Settings
    from core.embedding import build_embedding
    from mcp.indexed_knowledge_base import KnowledgeBase
    from mcp.tool_manager import MCPToolManager, Tool

    directory = root / config["knowledge_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    embedding = build_embedding()
    if embedding.settings.model != "qwen3.7-text-embedding-flash":
        raise ValueError("实验主线须配置 Qwen Embedding")
    client = chromadb.PersistentClient(path=str(directory / "chroma"),
                                      settings=Settings(anonymized_telemetry=False))
    knowledge = KnowledgeBase(chroma_path=str(directory), embedding=embedding, client=client,
                              load_defaults=False, collection_prefix=config["collection_prefix"])
    manager = MCPToolManager(api_key="unused", model=config["rerank_model"], gateway=gateway)
    manager.register(Tool(name="knowledge_search", description="原子客服政策检索",
        handler=knowledge.search_handler, schema={"type": "object", "required": ["query"],
        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}}},
        timeout_s=config["timeout_s"], cache_ttl=0, supports_rerank=True))
    return PolicyRetrieval(knowledge, manager, candidates=config["candidates"], top_k=config["top_k"],
                           observation_chars=config["observation_chars"])
