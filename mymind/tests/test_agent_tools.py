from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.tools import billing_tools, check_billing_fields, validate_tool_input


@pytest.mark.parametrize("channel", ["未知", "unknown", "  未知  ", " UNKNOWN ", " ", ""])
def test_explicit_unknown_channel_is_missing(channel):
    request = SimpleNamespace(entities={})
    arguments = {"payment_channel": channel}
    validate_tool_input(billing_tools()["check_billing_fields"], arguments)
    result = check_billing_fields(request, arguments)
    assert result["fields"]["payment_channel"] is False
    assert "payment_channel" in result["missing_fields"]
    assert result["can_confirm_refund"] is False


@pytest.mark.parametrize("channel", ["银行卡", "线上支付渠道"])
def test_provided_channel_keeps_existing_field_check(channel):
    result = check_billing_fields(SimpleNamespace(entities={}), {"payment_channel": channel})
    assert result["fields"]["payment_channel"] is True
    assert result["missing_fields"] == ["order_id", "amount", "date"]
    assert result["can_confirm_refund"] is False


def test_unknown_channel_does_not_change_other_known_fields():
    request = SimpleNamespace(entities={"order_id": "ORD-1", "amount": "80元", "date": "2026-10-17"})
    result = check_billing_fields(request, {"payment_channel": "未知"})
    assert result["fields"] == {"order_id": True, "amount": True, "date": True, "payment_channel": False}
    assert result["missing_fields"] == ["payment_channel"]
    assert result["reason"] == "当前工具只做字段检查，不连接订单或支付系统"
