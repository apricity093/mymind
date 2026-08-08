"""Bounded assembly of dynamic customer-support context."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ContextBuildResult:
    text: str
    metadata: Dict[str, Any]


class ContextBuilder:
    def __init__(self, max_chars: int = 8000):
        self.max_chars = max(500, int(max_chars))

    def build(self, memory: Any, knowledge: str = "", current_message: str = "") -> ContextBuildResult:
        episodic = [str(item).strip() for item in memory.relevant_history if str(item).strip()]
        recent = list(memory.recent_messages)
        profile = dict(memory.user_profile or {})
        summary = str(memory.summary or "").strip()
        knowledge_text = str(knowledge or "").strip()
        reductions: List[Dict[str, Any]] = []

        def render() -> tuple[str, Dict[str, int]]:
            sections: List[tuple[str, str]] = []
            if summary:
                sections.append(("summary", f"[会话摘要]\n{summary}"))
            if episodic:
                sections.append(("episodic", "[相关历史]\n" + "\n".join(f"- {item}" for item in episodic)))
            if profile:
                sections.append(("profile", "[用户画像]\n" + json.dumps(profile, ensure_ascii=True, sort_keys=True)))
            if recent:
                rows = [f"{item.role.value}: {item.content}" for item in recent]
                sections.append(("recent", "[最近对话]\n" + "\n".join(rows)))
            if knowledge_text:
                sections.append(("knowledge", knowledge_text))
            return "\n\n".join(text for _, text in sections), {name: len(text) for name, text in sections}

        text, section_chars = render()
        while len(text) > self.max_chars:
            before = len(text)
            section = ""
            if episodic:
                episodic.pop()
                section = "episodic"
            elif profile:
                key = next(reversed(profile))
                profile.pop(key, None)
                section = "profile"
            elif len(recent) > 2:
                recent.pop(0)
                section = "recent"
            elif len(summary) > 200:
                summary = summary[-max(200, len(summary) - (before - self.max_chars)):]
                section = "summary"
            elif knowledge_text:
                keep = max(0, len(knowledge_text) - (before - self.max_chars))
                knowledge_text = knowledge_text[:keep]
                section = "knowledge"
            else:
                text = text[: self.max_chars]
                section = "hard_limit"
                reductions.append({"section": section, "before_chars": before, "after_chars": len(text)})
                break
            text, section_chars = render()
            reductions.append({"section": section, "before_chars": before, "after_chars": len(text)})

        metadata = {
            "max_chars": self.max_chars,
            "rendered_chars": len(text),
            "over_budget": len(text) > self.max_chars,
            "current_request_preserved": current_message is not None,
            "current_request_chars": len(str(current_message)),
            "sections": section_chars,
            "reductions": reductions,
            "episodic_rendered": len(episodic),
            "recent_rendered": len(recent),
        }
        return ContextBuildResult(text=text, metadata=metadata)
