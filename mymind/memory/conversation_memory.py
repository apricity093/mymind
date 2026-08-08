"""
亮点：多轮对话记忆管理

三级记忆架构，模拟人类记忆机制：
  1. 工作记忆（Redis）—— 当前会话的最近 N 条消息，毫秒级读写
  2. 情景记忆（ChromaDB）—— 跨会话的历史对话，按语义相似度检索
  3. 用户画像（ChromaDB）—— 从对话中提炼的长期偏好和实体

关键设计：
  - 上下文构建时三级记忆融合，按重要性 + 时效性排序
  - 工作记忆超过阈值时自动压缩（LLM 摘要），防止 context 爆炸
  - 所有 Embedding 通过 Anthropic API 生成，无本地模型
"""
import hashlib
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import chromadb
import redis
from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class MsgRole(Enum):
    USER      = "user"
    ASSISTANT = "assistant"
    SYSTEM    = "system"


@dataclass
class Message:
    role:       MsgRole
    content:    str
    timestamp:  datetime = field(default_factory=datetime.now)
    metadata:   Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryContext:
    """传给 Agent 的完整上下文。"""
    recent_messages:  List[Message]   # 工作记忆：最近对话
    relevant_history: List[str]       # 情景记忆：语义相关的历史片段
    user_profile:     Dict[str, Any]  # 用户画像：偏好、常用实体
    summary:          str             # 当前会话摘要（压缩后）

    @staticmethod
    def _clean(text: str) -> str:
        """移除 Unicode 代理字符，防止编码错误。"""
        return text.encode("utf-8", errors="ignore").decode("utf-8")

    def to_prompt_text(self) -> str:
        """将记忆上下文格式化为 LLM 可用的文本。"""
        parts = []
        if self.summary:
            parts.append(f"[会话摘要]\n{self._clean(self.summary)}")
        if self.relevant_history:
            parts.append("[相关历史]\n" + "\n".join(f"- {self._clean(h)}" for h in self.relevant_history[:3]))
        if self.user_profile:
            parts.append(f"[用户画像]\n{json.dumps(self.user_profile, ensure_ascii=True)}")
        if self.recent_messages:
            parts.append("[最近对话]")
            for m in self.recent_messages:
                parts.append(f"{m.role.value}: {self._clean(m.content)}")
        return "\n\n".join(parts)


class MemoryManager:
    """
    三级记忆管理器。

    工作记忆存 Redis（TTL 24h），情景记忆和用户画像存 ChromaDB（持久化）。
    """

    WORKING_MAX   = 20    # 工作记忆最大条数，超过则触发压缩
    COMPRESS_AT   = 15    # 达到此条数时压缩，保留摘要 + 最近 5 条
    HISTORY_TOP_K = 5     # 情景记忆检索返回条数
    SUMMARY_MAX_CHARS = 1500
    PROFILE_UPDATE_INTERVAL_S = 600
    MIN_EPISODIC_RELEVANCE = 0.35

    def __init__(
        self,
        redis_url:    str = "redis://localhost:6379/0",
        chroma_host:  str = "localhost",
        chroma_port:  int = 8000,
        chroma_path:  str = "./data/chroma",
        api_key:      str = "",
        base_url:     Optional[str] = None,
        model:        str = "claude-3-5-sonnet-20241022",
        redis_client: Optional[Any] = None,
        chroma_client: Optional[Any] = None,
        llm_client: Optional[Any] = None,
        clock: Callable[[], float] = time.time,
        episodic_collection: str = "episodic",
        profile_collection: str = "user_profile",
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = llm_client or AsyncAnthropic(**kwargs)
        self._model  = model
        self._clock = clock

        self._redis = redis_client or redis.from_url(redis_url, decode_responses=True)

        # ChromaDB：优先连接独立服务（docker compose 模式），连不上则降级为本地嵌入式
        if chroma_client is not None:
            chroma = chroma_client
        else:
            try:
                # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
                chroma = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )
                chroma.heartbeat()  # 测试连接
                logger.info(f"ChromaDB 已连接: {chroma_host}:{chroma_port}")
            except Exception:
                logger.info(f"ChromaDB 服务不可用，使用本地嵌入式模式: {chroma_path}")
                chroma = chromadb.PersistentClient(
                    path=chroma_path,
                    settings=chromadb.Settings(anonymized_telemetry=False),
                )

        # 情景记忆：存储历史对话片段
        self._episodic = chroma.get_or_create_collection(episodic_collection)
        # 用户画像：存储提炼出的偏好和实体
        self._profile  = chroma.get_or_create_collection(profile_collection)

    # ── 写入 ──────────────────────────────────────────────────────────────────

    async def add_message(
        self,
        user_id: str,
        conv_id: str,
        role:    MsgRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """将一条消息写入工作记忆，超阈值时自动压缩。"""
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        clean_metadata = {
            self._safe_text(k): self._safe_metadata_value(v)
            for k, v in (metadata or {}).items()
        }
        msg = Message(role=role, content=self._safe_text(content), metadata=clean_metadata)
        key = self._wm_key(user_id, conv_id)

        lock_key = f"lock:memory:{self._key_part(user_id)}:{self._key_part(conv_id)}"
        lock_token = await self._acquire_lock(lock_key)
        try:
            # Serialize append and compression so a slow summary cannot overwrite newer messages.
            self._redis.lpush(key, json.dumps({
                "role":      msg.role.value,
                "content":   msg.content,
                "ts":        msg.timestamp.isoformat(),
                "metadata":  msg.metadata,
            }))
            self._redis.expire(key, 86400)
            if self._redis.llen(key) >= self.COMPRESS_AT:
                await self._compress_locked(user_id, conv_id)
        finally:
            self._release_lock(lock_key, lock_token)

    async def update_profile(self, user_id: str, conv_id: str) -> None:
        lock_key = f"lock:profile:{self._key_part(user_id)}"
        try:
            lock_token = await self._acquire_lock(lock_key)
        except TimeoutError as ex:
            logger.warning(f"用户画像更新跳过: {ex}")
            return
        try:
            await self._update_profile_locked(user_id, conv_id)
        finally:
            self._release_lock(lock_key, lock_token)

    async def _update_profile_locked(self, user_id: str, conv_id: str) -> None:
        """
        从当前工作记忆中提炼用户偏好，更新用户画像。
        用 LLM 提炼偏好，然后存入 ChromaDB（ChromaDB 内置 embedding，不依赖外部 API）。
        """
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        messages = await self._get_working_memory(user_id, conv_id)
        if not messages:
            return

        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in messages[-10:]))
        user_text = self._safe_text("\n".join(m.content for m in messages[-10:] if m.role == MsgRole.USER))
        if not user_text:
            return
        fingerprint = hashlib.sha256(user_text.encode("utf-8")).hexdigest()
        fingerprint_key = f"profile:fingerprint:{self._key_part(user_id)}"
        updated_key = f"profile:updated_at:{self._key_part(user_id)}"
        if self._redis.get(fingerprint_key) == fingerprint:
            return
        last_updated = float(self._redis.get(updated_key) or 0.0)
        if self._clock() - last_updated < self.PROFILE_UPDATE_INTERVAL_S:
            return
        prompt = f"""从以下对话中提炼用户偏好和关键实体，返回 JSON。
对话:
{text}

返回格式: {{"preferences": ["..."], "entities": {{"产品": [], "问题类型": []}}}}"""
        prompt = self._safe_text(prompt)

        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=512, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            profile_data = json.loads(raw[s:e])

            doc_id = f"{user_id}_profile"
            doc_text = self._safe_text(json.dumps(profile_data, ensure_ascii=False))
            metadata = {"user_id": user_id, "conv_id": conv_id, "ts": datetime.now().isoformat()}
            if hasattr(self._profile, "upsert"):
                self._profile.upsert(ids=[doc_id], documents=[doc_text], metadatas=[metadata])
            else:
                self._profile.delete(ids=[doc_id])
                self._profile.add(ids=[doc_id], documents=[doc_text], metadatas=[metadata])
            self._redis.set(fingerprint_key, fingerprint)
            self._redis.set(updated_key, str(self._clock()))
            logger.info(f"用户画像已更新: {user_id}")
        except Exception as ex:
            logger.warning(f"更新用户画像失败: {ex}")

    # ── 读取 ──────────────────────────────────────────────────────────────────

    async def get_context(self, user_id: str, conv_id: str, query: str = "") -> MemoryContext:
        """
        构建完整的记忆上下文。

        query 用于从情景记忆中检索语义相关的历史片段。
        """
        # 1. 工作记忆（当前会话最近消息）
        user_id = self._safe_text(user_id)
        conv_id = self._safe_text(conv_id)
        query = self._safe_text(query)

        recent = await self._get_working_memory(user_id, conv_id)

        # 2. 情景记忆（跨会话语义检索）
        history = await self._search_episodic(user_id, query or (recent[-1].content if recent else ""))

        # 3. 用户画像
        profile = await self._get_profile(user_id)

        # 4. 会话摘要（如果已压缩过）
        summary = self._redis.get(self._summary_key(user_id, conv_id)) or ""

        return MemoryContext(
            recent_messages=recent,
            relevant_history=history,
            user_profile=profile,
            summary=summary,
        )

    # ── 压缩（防止 context 爆炸）─────────────────────────────────────────────

    async def _compress(self, user_id: str, conv_id: str) -> None:
        """
        工作记忆压缩：
          1. 用 LLM 对旧消息生成摘要
          2. 摘要存 Redis（覆盖旧摘要）
          3. 旧消息存入情景记忆（ChromaDB）供跨会话检索
          4. 工作记忆只保留最近 5 条
        """
        lock_key = f"lock:memory:{self._key_part(user_id)}:{self._key_part(conv_id)}"
        lock_token = await self._acquire_lock(lock_key)
        try:
            await self._compress_locked(user_id, conv_id)
        finally:
            self._release_lock(lock_key, lock_token)

    async def _compress_locked(self, user_id: str, conv_id: str) -> None:
        messages = await self._get_working_memory(user_id, conv_id)
        if len(messages) < self.COMPRESS_AT:
            return

        to_compress = messages[:-5]   # 保留最近 5 条
        keep        = messages[-5:]

        # LLM 摘要
        text = self._safe_text("\n".join(f"{m.role.value}: {m.content}" for m in to_compress))
        prompt = self._safe_text(f"用 2-3 句话总结以下对话的关键信息：\n{text}")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=256, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            summary = self._safe_text(extract_text_content(resp.content)).strip()
        except Exception:
            summary = f"对话包含 {len(to_compress)} 条消息（摘要生成失败）"

        # 存摘要到 Redis
        skey = self._summary_key(user_id, conv_id)
        old_summary = self._redis.get(skey) or ""
        new_summary = await self._merge_summary(old_summary, summary)
        self._redis.setex(skey, 86400, new_summary)

        # 旧消息存入情景记忆
        await self._store_episodic(user_id, conv_id, text, summary)

        # 重置工作记忆为最近 5 条
        key = self._wm_key(user_id, conv_id)
        self._redis.delete(key)
        for m in keep:
            self._redis.lpush(key, json.dumps({
                "role": m.role.value, "content": m.content,
                "ts": m.timestamp.isoformat(), "metadata": m.metadata,
            }))
        self._redis.expire(key, 86400)
        logger.info(f"工作记忆压缩完成: {user_id}/{conv_id}，摘要 {len(summary)} 字")

    async def _merge_summary(self, old_summary: str, summary: str) -> str:
        combined = self._safe_text(f"{old_summary}\n{summary}").strip()
        if len(combined) <= self.SUMMARY_MAX_CHARS:
            return combined
        prompt = self._safe_text(f"将以下会话摘要重新归并，保留事实、偏好和未解决事项，控制在 1200 字以内：\n{combined}")
        try:
            resp = await self._client.messages.create(
                model=self._model, max_tokens=512, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            merged = self._safe_text(extract_text_content(resp.content)).strip()
            return merged[: self.SUMMARY_MAX_CHARS]
        except Exception:
            return combined[-self.SUMMARY_MAX_CHARS :]

    def _release_lock(self, key: str, token: str) -> None:
        script = "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
        try:
            self._redis.eval(script, 1, key, token)
        except Exception:
            if self._redis.get(key) == token:
                self._redis.delete(key)

    async def _acquire_lock(self, key: str, timeout_s: float = 30.0) -> str:
        token = uuid.uuid4().hex
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._redis.set(key, token, nx=True, px=120000):
                return token
            await asyncio.sleep(0.05)
        raise TimeoutError(f"memory lock timeout: {key}")

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    async def _get_working_memory(self, user_id: str, conv_id: str) -> List[Message]:
        key  = self._wm_key(user_id, conv_id)
        raws = self._redis.lrange(key, 0, self.WORKING_MAX - 1)
        msgs = []
        for raw in reversed(raws):  # Redis lpush 最新在前，reversed 还原时序
            d = json.loads(raw)
            msgs.append(Message(
                role=MsgRole(d["role"]),
                content=d["content"],
                timestamp=datetime.fromisoformat(d["ts"]),
                metadata=d.get("metadata", {}),
            ))
        return msgs

    async def _search_episodic(self, user_id: str, query: str) -> List[str]:
        """语义检索情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        query_text = self._safe_text(query).strip()
        if not query_text:
            return []
        try:
            # 直接传 query_texts，ChromaDB 内置模型自动生成向量做匹配
            results = self._episodic.query(
                query_texts=[query_text],
                n_results=self.HISTORY_TOP_K,
                where={"user_id": self._safe_text(user_id)},
                include=["documents", "distances", "metadatas"],
            )
            docs = results["documents"][0] if results["documents"] else []
            distances = results.get("distances", [[]])[0] if results.get("distances") else []
            metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
            metric = (getattr(self._episodic, "metadata", None) or {}).get("hnsw:space", "l2")
            candidates = []
            for index, doc in enumerate(docs):
                text = self._safe_text(doc).strip() if isinstance(doc, str) else ""
                if not text:
                    continue
                distance = float(distances[index]) if index < len(distances) else 0.0
                relevance = 1.0 - distance if metric == "cosine" else 1.0 - distance / 2.0
                if relevance < self.MIN_EPISODIC_RELEVANCE:
                    continue
                metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
                try:
                    timestamp = datetime.fromisoformat(str(metadata.get("ts", ""))).timestamp()
                except (TypeError, ValueError):
                    timestamp = 0.0
                candidates.append((relevance, timestamp, index, text))
            candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))

            selected = []
            seen = set()
            for _, _, _, text in candidates:
                normalized = " ".join(text.casefold().split())
                if normalized in seen:
                    continue
                seen.add(normalized)
                selected.append(text)
            return selected
        except Exception as ex:
            logger.warning(f"情景记忆检索失败: {ex}")
            return []

    async def _store_episodic(self, user_id: str, conv_id: str, text: str, summary: str) -> None:
        """将压缩后的对话片段存入情景记忆。ChromaDB 内置 embedding，不依赖外部 API。"""
        try:
            user_id = self._safe_text(user_id)
            conv_id = self._safe_text(conv_id)
            text = self._safe_text(text)
            summary = self._safe_text(summary)
            doc_id = uuid.uuid4().hex
            # 直接传 documents，ChromaDB 内置模型自动生成 embedding
            self._episodic.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"user_id": user_id, "conv_id": conv_id,
                            "ts": datetime.now().isoformat(), "full_text": self._safe_text(text[:500])}],
            )
        except Exception as ex:
            logger.warning(f"存储情景记忆失败: {ex}")

    async def _get_profile(self, user_id: str) -> Dict[str, Any]:
        """获取用户画像（取最新一条）。"""
        try:
            results = self._profile.get(ids=[f"{user_id}_profile"])
            if results["documents"]:
                return json.loads(results["documents"][0])
            legacy = self._profile.get(where={"user_id": user_id})
            rows = list(zip(legacy.get("documents", []), legacy.get("metadatas", [])))
            if rows:
                rows.sort(key=lambda row: str((row[1] or {}).get("ts", "")), reverse=True)
                return json.loads(rows[0][0])
        except Exception:
            pass
        return {}

    @staticmethod
    def _wm_key(user_id: str, conv_id: str) -> str:
        return f"wm:{MemoryManager._key_part(user_id)}:{MemoryManager._key_part(conv_id)}"

    @staticmethod
    def _summary_key(user_id: str, conv_id: str) -> str:
        return f"summary:{MemoryManager._key_part(user_id)}:{MemoryManager._key_part(conv_id)}"

    @staticmethod
    def _key_part(value: Any) -> str:
        return quote(MemoryManager._safe_text(value), safe="-_.~")

    @staticmethod
    def _safe_text(value: Any) -> str:
        """转成 ChromaDB 可接受的普通 UTF-8 字符串。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @classmethod
    def _safe_metadata_value(cls, value: Any) -> Any:
        """递归清洗 metadata，避免 Redis/ChromaDB 后续读写遇到非法 UTF-8。"""
        if isinstance(value, str):
            return cls._safe_text(value)
        if isinstance(value, dict):
            return {cls._safe_text(k): cls._safe_metadata_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._safe_metadata_value(v) for v in value]
        return value
