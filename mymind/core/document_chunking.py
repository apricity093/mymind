"""共享文档结构解析，以及各模型独立的输入预算切分。"""
from __future__ import annotations

import re
from dataclasses import dataclass

from core.retrieval import ChunkRecord, source_id_for
from core.embedding import EmbeddingError


@dataclass(frozen=True)
class Section:
    path: str
    text: str


def parse_sections(content, markdown=True):
    sections, lines, stack = [], [], []
    fence = None

    def flush():
        body = "\n".join(lines).strip()
        if body:
            sections.append(Section("/".join(heading for _, heading in stack), body))
        lines.clear()

    for line in content.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if marker:
            kind = marker.group()[0]
            fence = None if fence == kind else kind if fence is None else fence
        heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line) if markdown and not fence else None
        if heading:
            flush()
            level = len(heading.group(1))
            stack = [(depth, name) for depth, name in stack if depth < level]
            stack.append((level, heading.group(2)))
        elif markdown and not fence and re.fullmatch(r"\s{0,3}(?:=+|-+)\s*", line) and lines and lines[-1].strip():
            title_lines = []
            while lines and lines[-1].strip():
                title_lines.append(lines.pop().strip())
            flush()
            level = 1 if line.strip().startswith("=") else 2
            stack = [(depth, name) for depth, name in stack if depth < level]
            stack.append((level, " ".join(reversed(title_lines))))
        else:
            lines.append(line)
    flush()
    return sections


class BudgetChunker:
    """章节内递归切分；标题、分隔符与固定余量共同占用输入预算。"""

    def __init__(self, counter, budget, overlap=0, space="gemini", reserve=16):
        if budget <= reserve or not 0 <= overlap < budget - reserve:
            raise ValueError("块长须大于余量，overlap 须小于正文预算")
        self.counter, self.budget, self.overlap = counter, budget, overlap
        self.space, self.reserve = space, reserve

    @property
    def config(self):
        return dict(mode="structure_recursive", budget=self.budget, overlap=self.overlap,
                    reserve=self.reserve, space=self.space, parser_version=1)

    @staticmethod
    def embedding_title(title, path):
        return " / ".join(part for part in (title, path) if part)

    def chunk_document(self, title, content, source_id=None, markdown=True, structural=True):
        source_id = source_id or source_id_for(title, content)
        sections = parse_sections(content, markdown) if structural else [Section("", content.strip())]
        pieces = []
        for section in sections:
            heading = self.embedding_title(title, section.path)
            if self.counter(heading + "\n") + self.reserve >= self.budget:
                raise ValueError("标题路径超过模型片段预算")

            def fits(body, with_overlap=False):
                try:
                    count = self.counter(f"{heading}\n{body}") + self.reserve
                    if with_overlap:
                        count += self.overlap + self.counter("\n\n")
                    return count <= self.budget
                except EmbeddingError as ex:
                    if str(ex) == 'input_budget_exceeded':
                        return False
                    raise

            def split_fits(body):
                return fits(body, with_overlap=True)

            # 超长单元给后续重叠留出空间；完整短章节仍原样保留。
            units = self._split(section.text, fits if fits(section.text) or not self.overlap else split_fits,
                                preserve_fits=fits)
            current = ""
            for unit in units:
                joined = f"{current}\n\n{unit}" if current else unit
                if fits(joined):
                    current = joined
                    continue
                pieces.append((section.path, current))
                independent_qa = re.match(r"^(?:Q[:：]|问[:：]|问题[:：])", unit)
                tail = self._tail(current) if self.overlap and not independent_qa else ""
                current = f"{tail}\n\n{unit}" if tail and fits(f"{tail}\n\n{unit}") else unit
            if current:
                pieces.append((section.path, current))
        return [ChunkRecord(text, f"{self.space}:{source_id}:{i}", i, source_id, title,
                            path, len(pieces)) for i, (path, text) in enumerate(pieces)]

    def _split(self, text, fits, depth=0, preserve_fits=None):
        atomic = re.match(r"^(?:Q[:：]|问[:：]|问题[:：]|\d+[.、)]\s|步骤\s*\d)", text)
        if fits(text) or (atomic and preserve_fits and preserve_fits(text)):
            return [text] if text.strip() else []
        # 明确的问答对与编号步骤在适配时完整保留，超长单元再按更细边界切分。
        boundaries = [r"\n(?=(?:Q[:：]|问[:：]|问题[:：]|\d+[.、)]\s|步骤\s*\d))",
                      r"\n\s*\n", r"(?<=[。！？.!?])", r"\n", r"\s+"]
        if depth < len(boundaries):
            parts = re.split(boundaries[depth], text)
            return [piece for part in parts if part.strip()
                    for piece in self._split(part.strip(), fits, depth + 1, preserve_fits)]
        result = []
        while text:
            low, high = 1, len(text)
            if not fits(text[:1]):
                raise ValueError("标题之后没有可用的正文预算")
            while low < high:
                middle = (low + high + 1) // 2
                if fits(text[:middle]):
                    low = middle
                else:
                    high = middle - 1
            result.append(text[:low])
            text = text[low:]
        return result

    def _tail(self, text):
        # 优先完整句/段；目标不足容纳一个自然单元时，保留满足预算的短尾部。
        parts = re.split(r"(?<=[。！？.!?])|\n\s*\n", text)
        tail = ""
        for part in reversed(parts):
            candidate = part + tail
            if self.counter(candidate) > self.overlap:
                break
            tail = candidate
        if tail.strip():
            return tail.strip()
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.counter(text[-middle:]) <= self.overlap:
                low = middle
            else:
                high = middle - 1
        return text[-low:] if low else ""
