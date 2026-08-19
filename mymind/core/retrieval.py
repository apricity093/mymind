"""
RAG 检索实验核心构件（R0-R4 及消融变体）。

设计目标：让切块、稳定 ID、去重、BM25 混合召回、查询改写与 LLM 重排
成为可独立开关、可归因的检索变体，同时不破坏生产链路。

本模块只依赖标准库。BM25 为纯 Python 实现，避免为实验引入额外依赖；
中文按单字 + 相邻二元组切词，英文按单词切词。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ── 变体配置 ──────────────────────────────────────────────────────────────────

CHUNK_SIZE_DEFAULT = 500
OVERLAP_DEFAULT = 80
RRF_K = 60
RRF_VECTOR_WEIGHT = 0.5
RRF_BM25_WEIGHT = 0.5
RECALL_CANDIDATE_K = 20

_VARIANT_FACTORIES: Dict[str, Callable[[], "VariantConfig"]] = {}


def variant_config(name: str) -> "VariantConfig":
    """返回不可变变体配置；未知变体名直接报错，避免静默回退掩盖实验污染。"""
    normalized = name.strip().lower()
    if normalized not in _VARIANT_FACTORIES:
        raise KeyError(f"未知检索变体: {name}，可选: {sorted(_VARIANT_FACTORIES)}")
    return _VARIANT_FACTORIES[normalized]()


def register_variant(name: str) -> Callable[[Callable[[], "VariantConfig"]], Callable[[], "VariantConfig"]]:
    def decorator(factory: Callable[[], "VariantConfig"]) -> Callable[[], "VariantConfig"]:
        _VARIANT_FACTORIES[name.lower()] = factory
        return factory
    return decorator


@register_variant("r0")
def _r0() -> "VariantConfig":
    return VariantConfig(
        name="r0",
        label="当前生产实现",
        chunk_mode="sentence",
        chunk_size=CHUNK_SIZE_DEFAULT,
        overlap=0,
        cosine=False,
        stable_ids=False,
        dedup_mode="legacy_repr",
        hybrid=False,
        rewrite=True,
        llm_rerank=True,
        deterministic_rerank_fallback=False,
    )


@register_variant("r1")
def _r1() -> "VariantConfig":
    return VariantConfig(
        name="r1",
        label="修正基线",
        chunk_mode="sentence",
        chunk_size=CHUNK_SIZE_DEFAULT,
        overlap=0,
        cosine=True,
        stable_ids=True,
        dedup_mode="chunk_id",
        hybrid=False,
        rewrite=True,
        llm_rerank=True,
        deterministic_rerank_fallback=True,
    )


@register_variant("r2")
def _r2() -> "VariantConfig":
    return VariantConfig(
        name="r2",
        label="Markdown/段落感知切块 + overlap",
        chunk_mode="markdown",
        chunk_size=CHUNK_SIZE_DEFAULT,
        overlap=OVERLAP_DEFAULT,
        cosine=True,
        stable_ids=True,
        dedup_mode="chunk_id",
        hybrid=False,
        rewrite=True,
        llm_rerank=True,
        deterministic_rerank_fallback=True,
    )


@register_variant("r3")
def _r3() -> "VariantConfig":
    return VariantConfig(
        name="r3",
        label="BM25/向量加权 RRF",
        chunk_mode="markdown",
        chunk_size=CHUNK_SIZE_DEFAULT,
        overlap=OVERLAP_DEFAULT,
        cosine=True,
        stable_ids=True,
        dedup_mode="chunk_id",
        hybrid=True,
        rewrite=False,
        llm_rerank=False,
        deterministic_rerank_fallback=True,
    )


@register_variant("r4")
def _r4() -> "VariantConfig":
    return VariantConfig(
        name="r4",
        label="完整候选方案",
        chunk_mode="markdown",
        chunk_size=CHUNK_SIZE_DEFAULT,
        overlap=OVERLAP_DEFAULT,
        cosine=True,
        stable_ids=True,
        dedup_mode="chunk_id",
        hybrid=True,
        rewrite=True,
        llm_rerank=True,
        deterministic_rerank_fallback=True,
    )


@register_variant("r4-no-rewrite")
def _r4_no_rewrite() -> "VariantConfig":
    config = _r4()
    return config.replace(name="r4-no-rewrite", label="R4 - rewrite", rewrite=False)


@register_variant("r4-no-rerank")
def _r4_no_rerank() -> "VariantConfig":
    config = _r4()
    return config.replace(name="r4-no-rerank", label="R4 - rerank", llm_rerank=False)


@register_variant("r4-no-bm25")
def _r4_no_bm25() -> "VariantConfig":
    config = _r4()
    return config.replace(name="r4-no-bm25", label="R4 - BM25", hybrid=False)


@register_variant("r4-no-overlap")
def _r4_no_overlap() -> "VariantConfig":
    config = _r4()
    return config.replace(name="r4-no-overlap", label="R4 - overlap", overlap=0)


@dataclass(frozen=True)
class VariantConfig:
    """一个检索变体的完整开关。实验固定语料/embedding/回答模型，只变这些开关。"""

    name: str
    label: str = ""
    chunk_mode: str = "sentence"            # sentence | markdown
    chunk_size: int = CHUNK_SIZE_DEFAULT
    overlap: int = 0
    cosine: bool = True
    stable_ids: bool = True
    dedup_mode: str = "chunk_id"            # chunk_id | content_hash | legacy_repr
    hybrid: bool = False                    # BM25 + 向量 RRF
    rewrite: bool = True                    # 查询改写
    llm_rerank: bool = True                 # LLM 重排
    deterministic_rerank_fallback: bool = True
    collection_version: int = 1
    index_version: str = "rag-index-v1"

    @property
    def collection_name(self) -> str:
        return f"mymind_rag_{self.name}_v{self.collection_version}"

    @property
    def chunk_config(self) -> Dict[str, Any]:
        return {
            "chunk_mode": self.chunk_mode,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "index_version": self.index_version,
            "collection_version": self.collection_version,
        }

    def replace(self, **changes: Any) -> "VariantConfig":
        values = {
            "name": self.name,
            "label": self.label,
            "chunk_mode": self.chunk_mode,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "cosine": self.cosine,
            "stable_ids": self.stable_ids,
            "dedup_mode": self.dedup_mode,
            "hybrid": self.hybrid,
            "rewrite": self.rewrite,
            "llm_rerank": self.llm_rerank,
            "deterministic_rerank_fallback": self.deterministic_rerank_fallback,
            "collection_version": self.collection_version,
            "index_version": self.index_version,
        }
        values.update(changes)
        return VariantConfig(**values)


# ── 稳定标识 ──────────────────────────────────────────────────────────────────

def source_id_for(title: str, content: str) -> str:
    """稳定 source_id：优先由规范化标题派生，标题缺失时由全文派生。"""
    normalized = re.sub(r"\s+", " ", (title or "").strip()).casefold()
    if normalized:
        return "src-" + hashlib.sha1(f"title:{normalized}".encode("utf-8")).hexdigest()[:20]
    return "src-" + hashlib.sha1(content.encode("utf-8")).hexdigest()[:20]


def legacy_chunk_id(title: str, chunk_index: int, text: str) -> str:
    """R0 生产实现的旧 chunk id，用于基线回放与兼容。"""
    return hashlib.md5(f"{title}_{chunk_index}_{text[:50]}".encode("utf-8")).hexdigest()


def stable_chunk_id(source_id: str, section_path: str, chunk_index: int) -> str:
    """与内容无关的稳定 chunk id：重建索引、重排文档顺序都不会改变。"""
    raw = f"{source_id}::{section_path}::{chunk_index}::v1"
    return "chk-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def normalize_content_key(text: str) -> str:
    """按规范化内容去重：忽略空白差异，避免同一内容被重复注入。"""
    normalized = re.sub(r"\s+", "", text or "").casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


# ── 切块 ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkRecord:
    """切块结果。metadata 可直接写入 Chroma collection。"""

    text: str
    chunk_id: str
    chunk_index: int
    source_id: str
    title: str
    section_path: str = ""
    total_chunks: int = 1

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_id": self.chunk_id,
            "source_id": self.source_id,
            "section_path": self.section_path,
        }


class Chunker:
    """支持 sentence（R0/R1）与 markdown/段落感知（R2+）两种切块模式。"""

    def __init__(self, config: VariantConfig):
        self.config = config

    def chunk_document(self, title: str, content: str) -> List[ChunkRecord]:
        source_id = source_id_for(title, content)
        text = content or ""
        if self.config.chunk_mode == "markdown":
            pieces = self._markdown_pieces(text)
        else:
            pieces = self._sentence_pieces(text)

        chunks = self._pack_pieces(pieces)
        if not chunks and text.strip():
            chunks = [text.strip()]
        if not chunks:
            return []

        total = len(chunks)
        return [
            ChunkRecord(
                text=chunk_text,
                chunk_id=self._chunk_id(source_id, title, chunk_text, index),
                chunk_index=index,
                source_id=source_id,
                title=title,
                section_path=self._section_for(index),
                total_chunks=total,
            )
            for index, chunk_text in enumerate(chunks)
        ]

    # 内部实现：markdown pieces 为 (section_path, text)；sentence pieces 的 section 为空。
    _sections: List[str] = field(default_factory=list, init=False)

    def _chunk_id(self, source_id: str, title: str, text: str, index: int) -> str:
        if not self.config.stable_ids:
            return legacy_chunk_id(title, index, text)
        return stable_chunk_id(source_id, self._section_for(index), index)

    def _section_for(self, index: int) -> str:
        if index < len(self._sections):
            return self._sections[index]
        return self._sections[-1] if self._sections else ""

    @staticmethod
    def _sentence_pieces(text: str) -> List[Tuple[str, str]]:
        # 保持与当前生产实现一致的语义：换行视为句号。
        sentences = text.replace("\n", "。").split("。")
        pieces: List[Tuple[str, str]] = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                pieces.append(("", sentence))
        return pieces

    @staticmethod
    def _markdown_pieces(text: str) -> List[Tuple[str, str]]:
        """按 Markdown 标题维护 section_path，并把标题行并入段落首行。

        标题只维护路径，不单独成块；正文按空行分段，长段再按句子拆开。
        """
        pieces: List[Tuple[str, str]] = []
        section_stack: List[str] = []
        paragraph: List[str] = []
        pending_heading: Optional[str] = None

        def section_path() -> str:
            return "/".join(section_stack) if section_stack else ""

        def flush_paragraph() -> None:
            nonlocal pending_heading
            if not paragraph:
                pending_heading = None
                return
            body = "\n".join(part for part in paragraph if part.strip()).strip()
            paragraph.clear()
            if not body:
                pending_heading = None
                return
            if pending_heading:
                body = f"{pending_heading}\n{body}"
                pending_heading = None
            pieces.append((section_path(), body))

        for raw_line in text.splitlines():
            line = raw_line.strip()
            heading = re.match(r"^(#{1,6})\s+(.+)$", line)
            if heading:
                flush_paragraph()
                level = len(heading.group(1))
                section_stack = section_stack[: level - 1]
                section_stack.append(heading.group(2).strip())
                pending_heading = line
                continue
            if not line:
                flush_paragraph()
                continue
            paragraph.append(line)

        flush_paragraph()

        # 长段按句子二次拆分，保持同一 section_path。
        expanded: List[Tuple[str, str]] = []
        for section, piece in pieces:
            if len(piece) <= CHUNK_SIZE_DEFAULT:
                expanded.append((section, piece))
                continue
            current = ""
            for sentence in re.split(r"(?<=[。！？!?])", piece):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current) + len(sentence) + 1 > CHUNK_SIZE_DEFAULT:
                    if current:
                        expanded.append((section, current))
                    current = sentence
                else:
                    current = f"{current} {sentence}" if current else sentence
            if current:
                expanded.append((section, current))
        return expanded

    def _pack_pieces(self, pieces: Sequence[Tuple[str, str]]) -> List[str]:
        """把 (section_path, text) 打包成 chunk_size 块，并实现 overlap。"""
        chunk_size = max(1, self.config.chunk_size)
        overlap = max(0, min(self.config.overlap, chunk_size - 1))
        chunks: List[str] = []
        sections: List[str] = []
        current = ""
        current_section = ""

        def flush() -> None:
            nonlocal current, current_section
            if current:
                chunks.append(current)
                sections.append(current_section)
            current = ""
            current_section = ""

        for section, piece in pieces:
            if section != current_section:
                flush()
                current_section = section
            if not current:
                current = piece
                continue
            joined = f"{current}\n{piece}"
            if len(joined) <= chunk_size:
                current = joined
                continue
            flush()
            # overlap：新块继承上一块的尾部，而不是从零开始。
            current = current_tail(chunks[-1], overlap) if chunks and overlap else ""
            current = f"{current}\n{piece}".strip("\n") if current else piece
            if section != sections[-1]:
                sections[-1] = section
        flush()
        self._sections = sections
        return chunks


def current_tail(chunk: str, overlap: int) -> str:
    if overlap <= 0:
        return ""
    return chunk[-overlap:]


# ── BM25 ──────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """中文单字 + 相邻二元组，英文/数字整词小写。无第三方分词依赖。"""
    tokens: List[str] = []
    cjk: List[str] = []
    for match in _TOKEN_RE.finditer(text or ""):
        token = match.group(0)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            cjk.extend(token)
        else:
            if cjk:
                tokens.extend(cjk)
                tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
                cjk = []
            tokens.append(token.casefold())
    if cjk:
        tokens.extend(cjk)
        tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
    return tokens


class BM25Index:
    """轻量 Okapi BM25，支持增量建索引和查询。"""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: Dict[str, str] = {}
        self.items: Dict[str, Dict[str, Any]] = {}
        self.frequencies: Dict[str, Dict[str, int]] = {}
        self.document_frequency: Dict[str, int] = {}
        self.average_document_length = 0.0

    def clear(self) -> None:
        self.documents.clear()
        self.items.clear()
        self.frequencies.clear()
        self.document_frequency.clear()
        self.average_document_length = 0.0

    def add(self, doc_id: str, text: str, item: Optional[Dict[str, Any]] = None) -> None:
        if doc_id in self.documents:
            self.remove(doc_id)
        tokens = tokenize(text)
        self.documents[doc_id] = text
        self.items[doc_id] = dict(item or {})
        if not tokens:
            return
        counts: Dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        self.frequencies[doc_id] = counts
        for token in counts:
            self.document_frequency[token] = self.document_frequency.get(token, 0) + 1
        self.average_document_length = statistics.mean(
            [len(self.frequencies[d]) for d in self.frequencies]
        ) if self.frequencies else 0.0

    def remove(self, doc_id: str) -> None:
        counts = self.frequencies.pop(doc_id, None)
        if counts:
            for token in counts:
                if self.document_frequency.get(token, 0) <= 1:
                    self.document_frequency.pop(token, None)
                else:
                    self.document_frequency[token] -= 1
        self.documents.pop(doc_id, None)
        self.items.pop(doc_id, None)
        self.average_document_length = statistics.mean(
            [len(self.frequencies[d]) for d in self.frequencies]
        ) if self.frequencies else 0.0

    def score(self, doc_id: str, query: str) -> float:
        counts = self.frequencies.get(doc_id)
        if not counts:
            return 0.0
        document_length = len(counts)
        average_length = self.average_document_length or document_length
        denominator = self.k1 * ((1 - self.b) + self.b * document_length / average_length)
        score = 0.0
        document_count = len(self.documents)
        for token in set(tokenize(query)):
            document_frequency = self.document_frequency.get(token, 0)
            if document_frequency == 0:
                continue
            term_frequency = counts.get(token, 0)
            idf = math.log(1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5))
            score += idf * (term_frequency * (self.k1 + 1)) / (term_frequency + denominator)
        return score

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.documents:
            return []
        scored = sorted(
            ((doc_id, self.score(doc_id, query)) for doc_id in self.documents),
            key=lambda item: (-item[1], item[0]),
        )
        return [item for item in scored if item[1] > 0][:max(0, top_k)]


# ── RRF 与去重 ────────────────────────────────────────────────────────────────

def weighted_rrf(
    vector_ranked: Sequence[Dict[str, Any]],
    bm25_ranked: Sequence[Dict[str, Any]],
    k: int = RRF_K,
    vector_weight: float = RRF_VECTOR_WEIGHT,
    bm25_weight: float = RRF_BM25_WEIGHT,
) -> List[Dict[str, Any]]:
    """按 chunk_id 做加权倒数排名融合，返回 items 并附带 fusion_score。"""
    vector_ranks = {item_key(item): rank for rank, item in enumerate(vector_ranked, start=1)}
    bm25_ranks = {item_key(item): rank for rank, item in enumerate(bm25_ranked, start=1)}
    fused: Dict[str, Tuple[Dict[str, Any], float, List[str]]] = {}
    for item in vector_ranked:
        key = item_key(item)
        sources = ["vector", "bm25"] if key in bm25_ranks else ["vector"]
        fused[key] = (item, rrf_part(vector_ranks, key, vector_weight, k), sources)
    for item in bm25_ranked:
        key = item_key(item)
        if key in fused:
            item, score, sources = fused[key]
            fused[key] = (item, score + rrf_part(bm25_ranks, key, bm25_weight, k), sources)
        else:
            fused[key] = (
                item,
                rrf_part(bm25_ranks, key, bm25_weight, k),
                ["bm25"],
            )
    ordered = sorted(fused.values(), key=lambda row: (-row[1], item_key(row[0])))
    return [decorate(item, score, sources) for item, score, sources in ordered]


def rrf_part(ranks: Dict[str, int], key: str, weight: float, k: int) -> float:
    rank = ranks.get(key)
    if rank is None:
        return 0.0
    return weight / (k + rank)


def item_strength(item: Dict[str, Any]) -> float:
    """无答案过滤用的统一相关性强度：有 RRF 融合分数时用融合分数，否则用原始 score。"""
    for key in ("fusion_score", "score"):
        try:
            value = float(item.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return 0.0


def decorate(item: Dict[str, Any], fusion_score: Optional[float], sources: Sequence[str]) -> Dict[str, Any]:
    decorated = dict(item)
    decorated["fusion_score"] = round(fusion_score, 6) if fusion_score is not None else None
    decorated["retrieval_sources"] = list(sources)
    return decorated


def item_key(item: Dict[str, Any]) -> str:
    chunk_id = item.get("chunk_id") or item.get("chunkId") or item.get("id")
    if chunk_id:
        return str(chunk_id)
    return normalize_content_key(str(item.get("content", "")))


def dedupe_items(items: Iterable[Dict[str, Any]], mode: str = "chunk_id") -> List[Dict[str, Any]]:
    """正确去重：优先 chunk_id，其次规范化内容哈希；保留最高分先出现的位置。"""
    selected: List[Dict[str, Any]] = []
    seen: set = set()
    for item in items:
        if mode == "legacy_repr":
            # R0 回放：与旧实现一致，按整条 dict 的字符串表示去重。
            key = hashlib.md5(str(item).encode("utf-8")).hexdigest()
        elif mode == "chunk_id":
            key = item_key(item)
        else:
            key = normalize_content_key(str(item.get("content", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
    return selected


def deterministic_rank(items: Sequence[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """确定性重排回退：fusion_score → score → chunk_id → chunk_index。

    相同输入必然得到相同顺序，且不会把无关高分结果靠标题偶然前置。
    """

    def sort_key(item: Dict[str, Any]) -> Tuple[float, float, str, int]:
        fusion = item.get("fusion_score")
        fusion_value = float(fusion) if isinstance(fusion, (int, float)) else float("-inf")
        try:
            score_value = float(item.get("score", float("-inf")))
        except (TypeError, ValueError):
            score_value = float("-inf")
        return (
            -fusion_value if fusion_value != float("-inf") else 1.0,
            -score_value if score_value != float("-inf") else 1.0,
            str(item.get("chunk_id") or item.get("id") or item.get("title") or ""),
            int(item.get("chunk") or item.get("chunk_index") or 0),
        )

    return sorted(items, key=sort_key)[:max(0, top_k)]


# ── 检索管道 ──────────────────────────────────────────────────────────────────

@dataclass
class SearchRunStats:
    """守护指标：调用次数、失败与 fallback 率、分阶段延迟。"""

    vector_calls: int = 0
    bm25_calls: int = 0
    rewrite_calls: int = 0
    rewrite_failures: int = 0
    rerank_calls: int = 0
    rerank_failures: int = 0
    rerank_fallbacks: int = 0
    total_latency_ms: float = 0.0
    rewrite_latency_ms: float = 0.0
    vector_latency_ms: float = 0.0
    bm25_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0

    @property
    def rerank_failure_rate(self) -> float:
        return self.rerank_failures / self.rerank_calls if self.rerank_calls else 0.0

    @property
    def rerank_fallback_rate(self) -> float:
        return self.rerank_fallbacks / self.rerank_calls if self.rerank_calls else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "vector_calls": self.vector_calls,
            "bm25_calls": self.bm25_calls,
            "rewrite_calls": self.rewrite_calls,
            "rewrite_failures": self.rewrite_failures,
            "rerank_calls": self.rerank_calls,
            "rerank_failures": self.rerank_failures,
            "rerank_fallbacks": self.rerank_fallbacks,
            "total_latency_ms": round(self.total_latency_ms, 3),
            "rewrite_latency_ms": round(self.rewrite_latency_ms, 3),
            "vector_latency_ms": round(self.vector_latency_ms, 3),
            "bm25_latency_ms": round(self.bm25_latency_ms, 3),
            "rerank_latency_ms": round(self.rerank_latency_ms, 3),
            "rerank_failure_rate": round(self.rerank_failure_rate, 4),
            "rerank_fallback_rate": round(self.rerank_fallback_rate, 4),
        }


@dataclass
class SearchOutcome:
    items: List[Dict[str, Any]]
    stats: SearchRunStats
    reranked: bool = False
    rewritten: bool = False


VectorSearcher = Callable[[str, int], Any]
QueryRewriter = Callable[[str, int], Any]
IndexReranker = Callable[[str, List[Dict[str, Any]], int], Any]


class RetrievalPipeline:
    """可配置检索管道：改写 → 并行召回 → (BM25 + RRF) → 去重 → 重排/确定性回退。"""

    def __init__(
        self,
        config: VariantConfig,
        vector_searcher: VectorSearcher,
        bm25_index: Optional[BM25Index] = None,
        rewriter: Optional[QueryRewriter] = None,
        reranker: Optional[IndexReranker] = None,
        min_score: float = 0.0,
    ):
        self.config = config
        self._vector_searcher = vector_searcher
        self._bm25_index = bm25_index
        self._rewriter = rewriter
        self._reranker = reranker
        self.min_score = min_score
        self.observers: List[Callable[[SearchRunStats], None]] = []

    async def search(self, query: str, top_k: int = 5) -> SearchOutcome:
        started = time.perf_counter()
        stats = SearchRunStats()
        rewritten = False
        sub_queries = [query]

        if self.config.rewrite and self._rewriter is not None:
            stats.rewrite_calls += 1
            phase_started = time.perf_counter()
            try:
                value = self._rewriter(query, 3)
                if asyncio.iscoroutine(value):
                    value = await value
                sub_queries = await self._normalize_queries(query, value)
                rewritten = len(sub_queries) > 1
            except Exception:
                stats.rewrite_failures += 1
                sub_queries = [query]
            stats.rewrite_latency_ms += (time.perf_counter() - phase_started) * 1000

        recall_k = max(top_k, RECALL_CANDIDATE_K)
        vector_ranked = await self._vector_recall(sub_queries, recall_k, stats)
        bm25_ranked: List[Dict[str, Any]] = []
        if self.config.hybrid and self._bm25_index is not None:
            bm25_ranked = self._bm25_recall(sub_queries, recall_k, stats)

        if self.config.hybrid:
            merged = weighted_rrf(vector_ranked, bm25_ranked)
        else:
            merged = [decorate(item, None, ["vector"]) for item in vector_ranked]

        merged = dedupe_items(merged, self.config.dedup_mode)

        items, reranked = await self._rank(query, merged, top_k, stats)
        if self.min_score > 0:
            items = [item for item in items if item_strength(item) >= self.min_score]
        stats.total_latency_ms = (time.perf_counter() - started) * 1000
        for observer in self.observers:
            observer(stats)
        return SearchOutcome(items=items, stats=stats, reranked=reranked, rewritten=rewritten)

    async def _vector_recall(
        self,
        sub_queries: Sequence[str],
        recall_k: int,
        stats: SearchRunStats,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        phase_started = time.perf_counter()
        tasks = [self._maybe_await(self._vector_searcher(query, recall_k)) for query in sub_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            stats.vector_calls += 1
            if isinstance(result, BaseException):
                continue
            ranked.extend(result if isinstance(result, list) else [])
        stats.vector_latency_ms += (time.perf_counter() - phase_started) * 1000
        return ranked

    def _bm25_recall(
        self,
        sub_queries: Sequence[str],
        recall_k: int,
        stats: SearchRunStats,
    ) -> List[Dict[str, Any]]:
        assert self._bm25_index is not None
        ranked: List[Dict[str, Any]] = []
        phase_started = time.perf_counter()
        for query in sub_queries:
            stats.bm25_calls += 1
            for doc_id, score in self._bm25_index.search(query, recall_k):
                stored = dict(self._bm25_index.items.get(doc_id, {}))
                item = {
                    "chunk_id": stored.get("chunk_id") or str(doc_id),
                    "title": stored.get("title", ""),
                    "content": stored.get("content") or self._bm25_index.documents.get(doc_id, ""),
                    "score": round(score, 4),
                }
                for field in ("chunk", "chunk_index", "source_id", "section_path", "total_chunks"):
                    if field in stored:
                        item[field] = stored[field]
                ranked.append(item)
        stats.bm25_latency_ms += (time.perf_counter() - phase_started) * 1000
        return ranked

    async def _rank(
        self,
        query: str,
        merged: Sequence[Dict[str, Any]],
        top_k: int,
        stats: SearchRunStats,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        if len(merged) <= top_k:
            return deterministic_rank(merged, top_k), False

        if not self.config.llm_rerank or self._reranker is None:
            return deterministic_rank(merged, top_k), False

        stats.rerank_calls += 1
        phase_started = time.perf_counter()
        try:
            value = self._reranker(query, list(merged), top_k)
            if asyncio.iscoroutine(value):
                value = await value
            if isinstance(value, (list, tuple)) and all(
                isinstance(item, int) and 0 <= item < len(merged) for item in value
            ):
                ordered = [merged[index] for index in value]
                return ordered[:top_k], True
            raise ValueError("rerank 返回格式不是合法索引列表")
        except Exception:
            stats.rerank_failures += 1
            stats.rerank_fallbacks += 1
            if self.config.deterministic_rerank_fallback:
                return deterministic_rank(merged, top_k), False
            # R0 回放：保持合并后的原始顺序（旧实现 items[:top_k]）。
            return list(merged)[:top_k], False
        finally:
            stats.rerank_latency_ms += (time.perf_counter() - phase_started) * 1000

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if asyncio.iscoroutine(value):
            return await value
        return value

    @staticmethod
    async def _normalize_queries(query: str, value: Any) -> List[str]:
        if isinstance(value, str):
            queries = [value]
        elif isinstance(value, (list, tuple)):
            queries = [str(item) for item in value if str(item).strip()]
        else:
            queries = []
        return list(dict.fromkeys([query] + queries))[:4]


def parse_rerank_indices(raw: str, item_count: int, top_k: int) -> List[int]:
    """解析 LLM 重排输出；索引越界/重复直接拒绝，由调用方触发确定性回退。"""
    if not raw:
        raise ValueError("空 rerank 输出")
    text = raw.strip()
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("rerank 输出不是 JSON 数组")
    values = json.loads(text[start : end + 1])
    if not isinstance(values, list):
        raise ValueError("rerank 输出不是数组")
    indices: List[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("rerank 索引不是整数")
        if value < 0 or value >= item_count:
            raise ValueError("rerank 索引越界")
        if value in indices:
            raise ValueError("rerank 索引重复")
        indices.append(value)
    return indices[:top_k]
