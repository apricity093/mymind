"""Deterministic policy for deciding whether a chat request should use RAG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.intent_recognizer import IntentCategory


@dataclass(frozen=True)
class KnowledgeDecision:
    should_search: bool
    reason: str


class KnowledgePolicy:
    """Keep knowledge gating explainable and independent from HTTP/LLM code."""

    _ALWAYS_SEARCH = {
        IntentCategory.QUERY,
        IntentCategory.TECHNICAL,
        IntentCategory.BILLING,
        IntentCategory.ACCOUNT,
        IntentCategory.ORDER_STATUS,
        IntentCategory.LOGISTICS,
        IntentCategory.REFUND,
        IntentCategory.INVOICE,
        IntentCategory.PAYMENT_ISSUE,
        IntentCategory.ACCOUNT_SECURITY,
        IntentCategory.TECHNICAL_LOGIN,
        IntentCategory.TECHNICAL_CRASH,
    }
    _NEVER_SEARCH = {
        IntentCategory.GREETING,
        IntentCategory.FEEDBACK,
        IntentCategory.ESCALATION,
        IntentCategory.HUMAN_HANDOFF,
    }
    _BUSINESS_KEYWORDS = (
        "订单", "物流", "快递", "配送", "退款", "退货", "账单", "扣款", "支付",
        "发票", "账户", "账号", "登录", "密码", "验证码", "报错", "错误", "崩溃",
        "会员", "积分", "订阅", "order", "delivery", "shipping", "refund", "invoice",
        "payment", "account", "login", "error",
    )

    def __init__(self, fallback_min_chars: int = 4) -> None:
        self.fallback_min_chars = max(1, int(fallback_min_chars))

    def decide(self, message: str, intent: Optional[IntentCategory]) -> KnowledgeDecision:
        if intent in self._ALWAYS_SEARCH:
            return KnowledgeDecision(True, f"intent:{intent.value}")
        if intent in self._NEVER_SEARCH:
            return KnowledgeDecision(False, f"intent:{intent.value}")

        normalized = " ".join(str(message or "").casefold().split())
        if len(normalized) < self.fallback_min_chars:
            return KnowledgeDecision(False, "fallback:text_too_short")
        matched = next((keyword for keyword in self._BUSINESS_KEYWORDS if keyword in normalized), None)
        if matched:
            return KnowledgeDecision(True, f"fallback:keyword:{matched}")
        return KnowledgeDecision(False, "fallback:no_business_signal")
