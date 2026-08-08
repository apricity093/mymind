"""Provider-aware construction of Anthropic prompt-cache boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class PromptCachePolicy:
    enabled: bool = False
    min_stable_chars: int = 4096

    def build_system(self, stable: str, dynamic: str = "") -> Tuple[Any, Dict[str, Any]]:
        stable = str(stable or "")
        dynamic = str(dynamic or "").strip()
        eligible = self.enabled and len(stable) >= self.min_stable_chars
        metadata = {
            "prompt_cache_enabled": self.enabled,
            "prompt_cache_eligible": eligible,
            "stable_prefix_chars": len(stable),
            "prompt_cache_min_chars": self.min_stable_chars,
        }
        if not eligible:
            combined = stable if not dynamic else f"{stable}\n\n{dynamic}"
            return combined, metadata
        blocks = [
            {
                "type": "text",
                "text": stable,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        if dynamic:
            blocks.append({"type": "text", "text": dynamic})
        return blocks, metadata
