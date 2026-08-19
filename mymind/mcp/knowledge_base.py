"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片后存入 ChromaDB（自动生成 Embedding）
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。

本文件同时支持 check.md 定义的 R0-R4 检索变体：
  - 默认构造（variant=None）保持生产 R0 行为与 knowledge_base collection；
  - 传入 variant 时使用独立、版本化的 collection，并从原始文档重建索引。
"""
import asyncio
import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

import chromadb

from core.retrieval import (
    BM25Index,
    Chunker,
    VariantConfig,
    decorate,
    deterministic_rank,
    source_id_for,
    variant_config,
    weighted_rrf,
)

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        variant: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Any = None,
    ):
        # 默认构造保持生产行为；variant 用于实验的独立、版本化 collection。
        self._config: Optional[VariantConfig] = variant_config(variant) if variant else None
        self._variant_name = self._config.name if self._config else None
        self._collection_name = collection_name or (
            self._config.collection_name if self._config else self.COLLECTION_NAME
        )
        self._chunker = Chunker(self._config) if self._config else None

        # 优先连接独立 ChromaDB 服务（服务端内置 embedding 模型，客户端无需下载）
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        metadata: Dict[str, Any] = {"description": f"mymind RAG collection ({self._collection_name})"}
        if self._config and self._config.cosine:
            # cosine 只对新 collection 生效；版本化名称保证实验 collection 一定是新建的。
            metadata["hnsw:space"] = "cosine"
        collection_kwargs: Dict[str, Any] = {"name": self._collection_name, "metadata": metadata}
        if not self._use_server and embedding_function is not None:
            # 服务端 Chroma 使用其内置默认 embedding；本地模式允许注入同一
            # all-MiniLM-L6-v2 实例（例如把模型下载到项目 workspace）。
            collection_kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(**collection_kwargs)

        self._bm25_index: Optional[BM25Index] = None
        if self._config and self._config.hybrid:
            self._bm25_index = BM25Index()
            self._load_bm25_index()

        # 只有默认生产知识库为空时才导入内置文档；实验 collection 必须从原始语料显式导入。
        if self._config is None and self._collection.count() == 0:
            self._load_default_docs()

    # ── 变体信息 ──────────────────────────────────────────────────────────────

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def variant_name(self) -> Optional[str]:
        return self._variant_name

    @property
    def bm25_index(self) -> Optional[BM25Index]:
        return self._bm25_index

    @property
    def chunk_config(self) -> Dict[str, Any]:
        config = getattr(self, "_config", None)
        if config is None:
            return {"chunk_mode": "sentence", "chunk_size": 500, "overlap": 0, "index_version": "production-v1"}
        return config.chunk_config

    @property
    def index_version(self) -> str:
        config = getattr(self, "_config", None)
        if config is None:
            return "production-v1"
        return config.index_version

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        - 默认生产模式：保持原有 sentence 切块（每片 500 字，无 overlap）。
        - 实验变体：使用对应 chunk_mode / overlap，并生成稳定 chunk_id/source_id/section_path。
        """
        ids, docs, metas = [], [], []

        config = getattr(self, "_config", None)
        chunker = getattr(self, "_chunker", None)
        bm25_index = getattr(self, "_bm25_index", None)

        for doc in documents:
            title = doc.get("title", "")
            content = doc.get("content", "")
            if config is None or config.chunk_mode == "sentence" and not config.stable_ids:
                chunks = self._legacy_chunk_records(title, content)
            else:
                assert chunker is not None
                chunks = chunker.chunk_document(title, content)

            for record in chunks:
                metadata = {
                    "title": record.title,
                    "chunk_index": record.chunk_index,
                    "total_chunks": record.total_chunks,
                    "chunk_id": record.chunk_id,
                    "source_id": record.source_id,
                    "section_path": record.section_path,
                    "index_version": self.index_version,
                    # Chroma metadata 只接受标量；chunk_config 序列化后存储并在返回时还原。
                    "chunk_config": json.dumps(self.chunk_config, ensure_ascii=False, sort_keys=True),
                }
                ids.append(record.chunk_id)
                docs.append(record.text)
                metas.append(metadata)
                if bm25_index is not None:
                    bm25_index.add(record.chunk_id, record.text, {
                        "chunk_id": record.chunk_id,
                        "title": record.title,
                        "content": record.text,
                        "chunk": record.chunk_index,
                        "chunk_index": record.chunk_index,
                        "source_id": record.source_id,
                        "section_path": record.section_path,
                        "total_chunks": record.total_chunks,
                    })

        if ids:
            # ChromaDB 会自动生成 Embedding
            self._collection.upsert(ids=ids, documents=docs, metadatas=metas)
            logger.info(f"知识库处理 {len(ids)} 个文档片段")

        return len(ids)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        - 默认生产模式：纯向量检索，返回 title/content/score/chunk（保持旧契约）。
        - 实验 hybrid 变体：自动执行 BM25/向量加权 RRF，并附带 fusion_score/retrieval_sources。
        """
        if self._config and self._config.hybrid:
            return self.hybrid_search(query, top_k)
        return self.vector_search(query, top_k)

    def vector_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """只走 Chroma 向量召回，供 RetrievalPipeline 和 hybrid 使用。"""
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        return self._format_results(results)

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """BM25/向量加权 RRF，并对融合结果做确定性排序。"""
        vector_ranked = self.vector_search(query, top_k=max(top_k, 20))
        bm25_ranked: List[Dict[str, Any]] = []
        if self._bm25_index is not None:
            for chunk_id, score in self._bm25_index.search(query, max(top_k, 20)):
                item = self._bm25_item(chunk_id, score)
                if item is not None:
                    bm25_ranked.append(item)
        fused = weighted_rrf(vector_ranked, bm25_ranked)
        return deterministic_rank(fused, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await asyncio.to_thread(self.search, query, top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")
        ids = results.get("ids") or [[None] * len(documents[0])] if documents and documents[0] else [[]]
        if not documents or not documents[0]:
            return items

        for doc, meta, dist, ident in zip(
            documents[0],
            metadatas[0] if metadatas and metadatas[0] else [{} for _ in documents[0]],
            distances[0] if distances and distances[0] else [0.0 for _ in documents[0]],
            ids[0],
        ):
            item: Dict[str, Any] = {
                "title": (meta or {}).get("title", ""),
                "content": doc,
                "score": round(1.0 - dist, 4),
                "chunk": (meta or {}).get("chunk_index", 0),
            }
            # 新增字段均为可选增量字段，旧字段 title/content/score/chunk 保持不变。
            chunk_config = (meta or {}).get("chunk_config", self.chunk_config)
            if isinstance(chunk_config, str):
                try:
                    chunk_config = json.loads(chunk_config)
                except json.JSONDecodeError:
                    chunk_config = self.chunk_config
            item.update({
                "chunk_id": ident or (meta or {}).get("chunk_id", ""),
                "source_id": (meta or {}).get("source_id", ""),
                "section_path": (meta or {}).get("section_path", ""),
                "index_version": (meta or {}).get("index_version", self.index_version),
                "chunk_config": chunk_config,
            })
            if ident and self._bm25_index is not None and ident in self._bm25_index.items:
                item["chunk_id"] = ident
                item["retrieval_sources"] = ["vector"]
            items.append(item)
        return items

    def _bm25_item(self, chunk_id: str, score: float) -> Optional[Dict[str, Any]]:
        if self._bm25_index is None:
            return None
        stored = self._bm25_index.items.get(chunk_id)
        if stored is None:
            return None
        item = dict(stored)
        item["score"] = round(score, 4)
        return decorate(item, None, ["bm25"])

    def _load_bm25_index(self) -> None:
        """从 collection 现有数据重建 BM25 索引（实验 collection 重建后调用）。"""
        if self._bm25_index is None:
            return
        self._bm25_index.clear()
        try:
            data = self._collection.get(include=["documents", "metadatas"])
        except Exception as ex:
            logger.warning(f"读取 collection 重建 BM25 失败: {ex}")
            return
        documents = data.get("documents") or []
        metadatas = data.get("metadatas") or [{} for _ in documents]
        ids = data.get("ids") or []
        for index, text in enumerate(documents):
            meta = metadatas[index] if index < len(metadatas) else {}
            chunk_id = ids[index] if index < len(ids) else meta.get("chunk_id") or f"chunk-{index}"
            self._bm25_index.add(chunk_id, text, {
                "chunk_id": chunk_id,
                "title": meta.get("title", ""),
                "content": text,
                "chunk": meta.get("chunk_index", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "source_id": meta.get("source_id", ""),
                "section_path": meta.get("section_path", ""),
                "total_chunks": meta.get("total_chunks", 0),
            })

    def _legacy_chunk_records(self, title: str, content: str):
        """回放旧生产实现：sentence 切块 + 旧式 md5 chunk_id + 稳定 source_id。

        chunk_id 保持旧行为以兼容已有前端；source_id 是新增可选字段，
        从标题派生，保证同一文档的所有 chunk 可被 source 级标注命中。
        """
        from core.retrieval import ChunkRecord, legacy_chunk_id

        records = []
        chunks = self._chunk_text(content, chunk_size=500)
        stable_source_id = source_id_for(title, content)
        for index, chunk in enumerate(chunks):
            records.append(ChunkRecord(
                text=chunk,
                chunk_id=legacy_chunk_id(title, index, chunk),
                chunk_index=index,
                source_id=stable_source_id,
                title=title,
                section_path="",
                total_chunks=len(chunks),
            ))
        return records

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（客服场景常见问题）。"""
        default_docs = [
            {
                "title": "退款政策",
                "content": (
                    "退款政策说明。"
                    "用户在购买后 7 天内可以申请无理由退款。"
                    "退款申请提交后，系统会在 1-3 个工作日内审核。"
                    "审核通过后，款项将在 5-7 个工作日内退回原支付账户。"
                    "如果商品已发货，需要先完成退货流程才能退款。"
                    "退货运费由用户承担，除非是商品质量问题。"
                    "超过 7 天但未超过 30 天的订单，需要提供商品质量问题的证据才能退款。"
                ),
            },
            {
                "title": "订单查询",
                "content": (
                    "订单查询指南。"
                    "用户可以通过订单号查询订单状态。"
                    "订单状态包括：待支付、已支付、已发货、运输中、已签收、已完成。"
                    "如果订单显示已发货但超过 7 天未收到，可以联系客服申请查件。"
                    "物流信息通常在发货后 24 小时内更新。"
                    "如果订单显示异常，请提供订单号联系客服处理。"
                ),
            },
            {
                "title": "账户安全",
                "content": (
                    "账户安全说明。"
                    "建议用户定期修改密码，密码长度至少 8 位，包含字母和数字。"
                    "如果忘记密码，可以通过绑定的手机号或邮箱重置。"
                    "发现账户异常登录时，系统会自动锁定账户并发送通知。"
                    "用户可以在安全设置中开启两步验证，提高账户安全性。"
                    "不要将密码分享给他人，客服人员不会索要用户密码。"
                ),
            },
            {
                "title": "技术故障排查",
                "content": (
                    "常见技术问题排查。"
                    "应用崩溃：请尝试清除缓存后重启应用，如果问题持续请更新到最新版本。"
                    "登录失败 401 错误：表示认证失败，请检查用户名密码是否正确，或尝试重置密码。"
                    "页面加载慢：检查网络连接，尝试切换 WiFi 或移动数据。"
                    "支付失败：确认银行卡余额充足，检查是否开启了网上支付功能。"
                    "500 服务器错误：这是服务端问题，请稍后重试，如果持续出现请联系技术支持。"
                ),
            },
            {
                "title": "会员与积分",
                "content": (
                    "会员积分规则。"
                    "每消费 1 元累积 1 积分。"
                    "积分可以在下次购物时抵扣，100 积分 = 1 元。"
                    "会员等级分为：普通会员、银卡会员（累计消费 1000 元）、金卡会员（累计消费 5000 元）。"
                    "银卡会员享受 95 折优惠，金卡会员享受 9 折优惠。"
                    "积分有效期为 1 年，过期自动清零。"
                    "生日当月消费可获得双倍积分。"
                ),
            },
            {
                "title": "配送说明",
                "content": (
                    "配送服务说明。"
                    "标准配送：3-5 个工作日送达，免运费（订单满 99 元）。"
                    "加急配送：1-2 个工作日送达，运费 15 元。"
                    "同城配送：当日达或次日达，运费 10 元。"
                    "偏远地区可能需要额外 2-3 天。"
                    "配送时间为每天 9:00-18:00，节假日可能延迟。"
                    "如果需要修改收货地址，请在发货前联系客服。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
