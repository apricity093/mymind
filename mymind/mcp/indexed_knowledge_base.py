"""生产知识库：独立主模型/MiniLM 双索引、原文保存和显式补齐。"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from contextlib import nullcontext
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

from core.document_chunking import BudgetChunker
from core.embedding import build_embedding
from core.retrieval import BM25Index, dedupe_items, source_id_for, weighted_rrf

logger = logging.getLogger(__name__)


def prepare_minilm(path=None, warm=True):
    """启动前准备 ONNX 模型，并用不截断、不填充的同模型 tokenizer 计量。"""
    from tokenizers import Tokenizer

    function = ONNXMiniLM_L6_V2()
    existing = Path(function.DOWNLOAD_PATH) / function.EXTRACTED_FOLDER_NAME / "model.onnx"
    if path or not existing.exists():
        function.DOWNLOAD_PATH = Path(path or "./data/onnx_models")
    if warm:
        function(["模型预热"])
    else:
        function._download_model_if_not_exists()
    tokenizer = Tokenizer.from_str(function.tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return function, lambda text: len(tokenizer.encode(text).ids)


class KnowledgeBase:
    BACKUP_BUDGET = 240
    BACKUP_OVERLAP = 0
    INDEX_VERSION = "dual-rag-v1"

    def __init__(self, chroma_host="localhost", chroma_port=8000, chroma_path="./data/chroma",
                 embedding=None, client=None, backup_function=None, backup_counter=None,
                 on_change=None, load_defaults=True, collection_prefix="knowledge"):
        self.embedding = embedding or build_embedding()
        self._main_space = "qwen" if self.embedding.settings.model == "qwen3.7-text-embedding-flash" else "gemini"
        self._failure_outcome = "provider_failed" if self._main_space == "qwen" else "google_failed"
        self.telemetry = self.embedding.telemetry
        self.on_change = on_change
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._vector_weights = {"main": float(os.getenv("RAG_MAIN_VECTOR_WEIGHT", "0.5")),
                                "backup": float(os.getenv("RAG_BACKUP_VECTOR_WEIGHT", "0.5"))}
        self.query_budget_s = float(os.getenv("RAG_EMBEDDING_QUERY_BUDGET_S", "8"))
        if not 0 < self.query_budget_s <= 20:
            raise ValueError("主 embedding 查询预算须大于 0 且不超过 20 秒，工具保留至少 10 秒")
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(Path(chroma_path) / f"{collection_prefix}_sources.sqlite3"),
                                   check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("""CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY, title TEXT NOT NULL, content TEXT NOT NULL,
            format TEXT NOT NULL, main_ready INTEGER NOT NULL DEFAULT 0,
            backup_ready INTEGER NOT NULL DEFAULT 0, main_config TEXT NOT NULL DEFAULT '')""")
        self._db.commit()
        if client is None:
            try:
                client = chromadb.HttpClient(host=chroma_host, port=chroma_port,
                                              settings=chromadb.Settings(anonymized_telemetry=False))
                client.heartbeat()
            except Exception:
                client = chromadb.PersistentClient(path=chroma_path,
                                                   settings=chromadb.Settings(anonymized_telemetry=False))
        self._client = client
        if backup_function is None or backup_counter is None:
            backup_function, backup_counter = prepare_minilm(os.getenv("RAG_ONNX_PATH") or None)
        self._backup_function = backup_function
        self._backup_chunker = BudgetChunker(backup_counter, self.BACKUP_BUDGET,
                                            self.BACKUP_OVERLAP, "minilm", reserve=0)
        self._main_chunker = BudgetChunker(self.embedding.count_tokens,
                                          int(os.getenv("RAG_CHUNK_TOKENS", "512")),
                                          int(os.getenv("RAG_OVERLAP_TOKENS", "0")), self._main_space)
        if self._main_chunker.budget > self.embedding.INPUT_LIMIT:
            raise ValueError("RAG_CHUNK_TOKENS 超过主 embedding 输入上限")
        self._signature = json.dumps({"embedding": self.embedding.identity,
                                      "counting": self.embedding.counting_mode if self._main_space == "qwen" else "gemini_auto_v1",
                                      **self._main_chunker.config}, sort_keys=True)
        invalidated = self._db.execute("UPDATE sources SET main_ready=0 WHERE main_config != ?", (self._signature,)).rowcount
        self._db.commit()
        # 维度不同必须使用不同 collection；分块变更由原文状态要求显式补齐。
        main_name = "qwen3_7_text_embedding_flash" if self._main_space == "qwen" else "gemini_001"
        self._main = client.get_or_create_collection(
            name=f"{collection_prefix}_{main_name}_{self.embedding.settings.dimensions}_v1",
            embedding_function=None, metadata={"hnsw:space": "cosine", "model": self.embedding.settings.model or "disabled"})
        self._backup = client.get_or_create_collection(
            name=f"{collection_prefix}_minilm_v1", embedding_function=backup_function,
            metadata={"hnsw:space": "cosine", "model": "all-MiniLM-L6-v2", "budget": self.BACKUP_BUDGET})
        self._bm25 = {"main": BM25Index(), "backup": BM25Index()}
        for route in ("main", "backup"):
            present = self._reload_bm25(route)
            for row in self._rows():
                if row[f"{route}_ready"] and row["content"].strip() and row["source_id"] not in present:
                    self._db.execute(f"UPDATE sources SET {route}_ready=0 WHERE source_id=?", (row["source_id"],))
                    invalidated += 1
        self._db.commit()
        # HTTP 服务转本地存储时，collection 可能为空；先由保留原文准备完整备用。
        for row in self._rows():
            if not row["backup_ready"]:
                try:
                    self._write_index("backup", dict(row))
                except Exception as ex:
                    self.telemetry.record("prepare_backup", row["source_id"], "failed", type(ex).__name__)
        if self._main_space == "qwen":
            legacy_name = re.compile(rf"{re.escape(collection_prefix)}_gemini_001_\d+_v1")
            for collection in client.list_collections():
                if legacy_name.fullmatch(collection.name):
                    client.delete_collection(collection.name)
                    self.telemetry.record("cleanup", collection.name, "removed")
                    invalidated += 1
        self._route = "main" if self._main_complete else "backup"
        self.last_import = {}
        if invalidated:
            self._changed()
        if load_defaults and not self._rows():
            # 旧 collection 保存了完整正文时可重新导入；零碎旧块不能充当原文。
            from mcp.knowledge_base import KnowledgeBase as BaselineKnowledgeBase
            BaselineKnowledgeBase._load_default_docs(self)

    def _rows(self):
        return self._db.execute("SELECT * FROM sources ORDER BY source_id").fetchall()

    @property
    def _main_complete(self):
        return self.embedding.enabled and not self._db.execute(
            "SELECT 1 FROM sources WHERE main_ready=0 OR backup_ready=0 LIMIT 1").fetchone()

    @property
    def index_version(self):
        return self.INDEX_VERSION

    @property
    def chunk_config(self):
        return {"main": self._main_chunker.config, "backup": self._backup_chunker.config,
                "main_counting": self.embedding.counting_mode}

    @property
    def doc_count(self):
        with self._lock:
            return self._main.count() if self._main_complete else self._backup.count()

    def stats(self):
        with self._lock:
            rows = self._rows()
            return {"documents": len(rows), "main_chunks": self._main.count(),
                    "backup_chunks": self._backup.count(),
                    "pending_main": sum(not r["main_ready"] for r in rows),
                    "pending_backup": sum(not r["backup_ready"] for r in rows),
                    "main_complete": bool(self._main_complete), "active_index": self._route,
                    "embedding_model": self.embedding.settings.model,
                    "telemetry": self.telemetry.snapshot()}

    def _changed(self):
        if self.on_change is not None and self.on_change() < 0:
            raise RuntimeError("知识索引已更新，检索缓存刷新失败")

    def _set_route(self, route, identity=""):
        if route != self._route:
            self._route = route
            self._changed()
            self.telemetry.record("route", identity, "recovered" if route == "main" else "degraded")

    def _reload_bm25(self, route):
        collection = self._main if route == "main" else self._backup
        index = self._bm25[route]
        index.clear()
        data = collection.get(include=["documents", "metadatas"])
        ready_ids = {r["source_id"] for r in self._rows() if r[f'{route}_ready']}
        present = {meta["source_id"] for meta in data["metadatas"]}
        for ident, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
            if meta["source_id"] not in ready_ids:
                continue
            item = {**meta, "content": text, "chunk": meta["chunk_index"], "chunk_id": ident}
            title = BudgetChunker.embedding_title(meta["title"], meta["section_path"])
            index.add(ident, f"{title}\n{text}", item)
        return present

    def _prepare_index(self, route, doc):
        chunker = self._main_chunker if route == "main" else self._backup_chunker
        records = chunker.chunk_document(doc["title"], doc["content"], doc["source_id"],
                                         markdown=doc["format"] != "txt")
        titles = [chunker.embedding_title(r.title, r.section_path) for r in records]
        texts = [r.text for r in records]
        if route == "main":
            vectors = self.embedding.embed(texts, "RETRIEVAL_DOCUMENT", titles)
        else:
            vectors = self._backup_function([f"{title}\n{text}" for title, text in zip(titles, texts)])
        return records, titles, texts, vectors

    def _write_index(self, route, doc, prepared=None):
        records, titles, texts, vectors = prepared or self._prepare_index(route, doc)
        collection = self._main if route == "main" else self._backup
        previous = collection.get(where={"source_id": doc["source_id"]}, include=[])["ids"]
        ids = [r.chunk_id for r in records]
        if ids:
            collection.upsert(ids=ids, documents=texts, embeddings=vectors,
                              metadatas=[{**r.metadata, "index_version": self.INDEX_VERSION,
                                          "embedding_model": self.embedding.settings.model if route == "main" else "all-MiniLM-L6-v2"}
                                         for r in records])
        obsolete = [ident for ident in previous if ident not in ids]
        if obsolete:
            collection.delete(ids=obsolete)
        self._db.execute(f"UPDATE sources SET {route}_ready=1" +
                         (", main_config=?" if route == "main" else "") + " WHERE source_id=?",
                         (self._signature, doc["source_id"]) if route == "main" else (doc["source_id"],))
        self._db.commit()
        # 仅替换变化文档的 BM25 条目。
        index = self._bm25[route]
        for ident in previous:
            index.remove(ident)
        for r, title in zip(records, titles):
            index.add(r.chunk_id, f"{title}\n{r.text}",
                      {**r.metadata, "content": r.text, "chunk": r.chunk_index})
        return len(records)

    def add_documents(self, documents):
        """返回实际备用入库片段数，详细结果见 import_documents。"""
        return self.import_documents(documents)["processed_chunks"]

    def import_documents(self, documents):
        outcomes, count = [], 0
        titles = Counter(d.get("title", "") for d in documents if not d.get("source_id"))
        occurrences = Counter()
        # 写入串行化不妨碍统计或查询读取完整的已提交状态。
        with self._write_lock:
            for document in documents:
                title, content = document.get("title", ""), document.get("content", "")
                source_id = document.get("source_id") or source_id_for(title, content)
                if not document.get("source_id") and titles[title] > 1:
                    source_id = f"{source_id}:entry:{occurrences[title]}"
                    occurrences[title] += 1
                doc = {"title": title, "content": content, "source_id": source_id,
                       "format": document.get("format", "md")}
                prepared, status, reason = {}, "degraded", ""
                try:
                    prepared["backup"] = self._prepare_index("backup", doc)
                except Exception as ex:
                    status, reason = "failed", self._reason(ex)
                if status != "failed" and self.embedding.enabled:
                    try:
                        prepared["main"] = self._prepare_index("main", doc)
                        status = "success"
                    except Exception as ex:
                        self.telemetry.record("import", source_id, self._failure_outcome, self._reason(ex))
                with self._lock:
                    self._db.execute("INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?, 0, 0, '')",
                                     (source_id, title, content, doc["format"]))
                    self._db.commit()
                    for index in self._bm25.values():
                        for ident, item in list(index.items.items()):
                            if item.get("source_id") == source_id:
                                index.remove(ident)
                    try:
                        if "backup" in prepared:
                            count += self._write_index("backup", doc, prepared["backup"])
                        if "main" in prepared:
                            self._write_index("main", doc, prepared["main"])
                    except Exception as ex:
                        status, reason = "failed", self._reason(ex)
                    self._set_route("main" if self._main_complete else "backup")
                    self._changed()
                    self.telemetry.record("import", source_id, status, reason)
                    outcomes.append({"source_id": source_id, "status": status, **({"reason": reason} if reason else {})})
            with self._lock:
                failed = sum(o["status"] == "failed" for o in outcomes)
                outcome = {"status": "failed" if failed == len(outcomes) and failed else
                           "partial" if failed else "degraded" if any(o["status"] == "degraded" for o in outcomes) else "success",
                           "degraded": any(o["status"] == "degraded" for o in outcomes),
                           "processed_chunks": count, "documents": outcomes,
                           "pending_main": sum(not r["main_ready"] for r in self._rows())}
                self.last_import = outcome
                return outcome

    @staticmethod
    def _reason(ex):
        from core.embedding import EmbeddingError
        return str(ex) if isinstance(ex, EmbeddingError) else type(ex).__name__

    def repair_main(self):
        outcomes = []
        with self._write_lock:
            with self._lock:
                rows = self._rows()
            for row in rows:
                doc = dict(row)
                try:
                    prepared = {}
                    if not row["backup_ready"]:
                        prepared["backup"] = self._prepare_index("backup", doc)
                    if not row["main_ready"]:
                        if not self.embedding.enabled:
                            raise ValueError("embedding_not_configured")
                        prepared["main"] = self._prepare_index("main", doc)
                    with self._lock:
                        for route, data in prepared.items():
                            self._write_index(route, doc, data)
                        self._changed()
                    outcomes.append({"source_id": row["source_id"], "status": "success"})
                    self.telemetry.record("repair", row["source_id"], "recovered")
                except Exception as ex:
                    self.telemetry.record("repair", row["source_id"], self._failure_outcome, self._reason(ex))
                    outcomes.append({"source_id": row["source_id"], "status": "failed", "reason": self._reason(ex)})
            with self._lock:
                self._set_route("main" if self._main_complete else "backup")
                return {"status": "success" if self._main_complete else "partial",
                        "documents": outcomes, **self.stats()}

    def delete_document(self, source_id):
        with self._write_lock, self._lock:
            # 删除两套索引及原文，存储失败向调用者报告。
            for collection in (self._main, self._backup):
                collection.delete(where={"source_id": source_id})
            self._db.execute("DELETE FROM sources WHERE source_id=?", (source_id,))
            self._db.commit()
            for route in ("main", "backup"):
                self._reload_bm25(route)
            self._set_route("main" if self._main_complete else "backup")
            self._changed()
            return self.stats()

    def search(self, query, top_k=5, mode="hybrid"):
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError("检索模式须为 vector、bm25 或 hybrid")
        identity = uuid.uuid4().hex[:12]
        route, vector = "backup", None
        with self._lock:
            complete = self._main_complete
        if complete:
            try:
                budget = self.embedding.query_budget(self.query_budget_s) if hasattr(self.embedding, "query_budget") else nullcontext()
                with budget:
                    vector = self.embedding.embed([query], "RETRIEVAL_QUERY")[0] if mode != "bm25" else None
                route = "main"
            except Exception as ex:
                self.telemetry.record("query", identity, self._failure_outcome, self._reason(ex))
        with self._lock:
            # 编码期间的更新可能使主库 pending；只查询当前完整的向量空间。
            if not self._main_complete:
                route = "backup"
            self._set_route(route, identity)
            collection = self._main if route == "main" else self._backup
            index = self._bm25[route]
            recall_k = max(20, top_k)
            vector_items = []
            ready_ids = [r["source_id"] for r in self._rows() if r[f'{route}_ready']]
            if mode != "bm25" and ready_ids and collection.count():
                kwargs = {"query_embeddings": [vector]} if route == "main" else {"query_texts": [query]}
                try:
                    data = collection.query(**kwargs, n_results=min(collection.count(), recall_k),
                                            where={"source_id": {"$in": ready_ids}})
                    vector_items = [{**meta, "content": text, "score": 1 - distance, "chunk_id": ident,
                                     "chunk": meta["chunk_index"], "retrieval_sources": ["vector"]}
                                    for ident, text, meta, distance in zip(data["ids"][0], data["documents"][0],
                                                                         data["metadatas"][0], data["distances"][0])]
                except Exception as ex:
                    self.telemetry.record("query", identity, "failed", type(ex).__name__)
                    raise
            bm25_items = [{**index.items[ident], "score": score, "retrieval_sources": ["bm25"]}
                          for ident, score in index.search(query, recall_k)] if mode != "vector" else []
            weight = self._vector_weights[route]
            items = weighted_rrf(vector_items, bm25_items, vector_weight=weight, bm25_weight=1-weight) if mode == "hybrid" else vector_items if mode == "vector" else bm25_items
            result = dedupe_items(items)[:top_k]
            if route == "backup":
                self.telemetry.record("query", identity, "degraded", f"evidence_count={len(result)}")
            return [{**item, "index_route": route, "degraded": route == "backup",
                     "pending_main": self.stats()["pending_main"], "index_version": self.INDEX_VERSION}
                    for item in result]

    def vector_search(self, query, top_k=5):
        return self.search(query, top_k, "vector")

    async def search_handler(self, params, context):
        return await asyncio.to_thread(self.search, params.get("query", ""), params.get("top_k", 5))

    def close(self):
        self._db.close()
