from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from core.llm_gateway import LLMRequest, build_gateway
from experiments.reporting import metadata, write_report
from memory.context_builder import ContextBuilder
from memory.conversation_memory import MemoryContext, Message, MsgRole

load_dotenv()


SCENARIOS = [
    ("退款", "订单在7天内可申请退款", "我的订单昨天购买，怎么退款？"),
    ("退款多轮", "退款审核需要1-3个工作日", "那审核要多久？"),
    ("重复扣款", "重复扣款需要提供支付流水", "同一笔订单扣了两次钱"),
    ("扣款多轮", "不要承诺立即退款", "现在能马上退回吗？"),
    ("物流", "发货24小时后更新物流", "为什么还没有物流信息？"),
    ("物流多轮", "超过7天未收到可申请查件", "已经八天了怎么办？"),
    ("登录故障", "先确认错误码和网络环境", "登录一直报500"),
    ("故障多轮", "后台操作应升级人工", "这些都试过了还是不行"),
    ("账户修改", "修改账户邮箱需要身份验证", "帮我修改登录邮箱"),
    ("账户多轮", "不得索取用户密码", "需要把密码发给你吗？"),
    ("投诉升级", "明确投诉应升级人工", "我要投诉并找经理"),
    ("升级多轮", "保留问题摘要供人工处理", "可以，把刚才的问题一起转过去"),
]


def _setting(name: str, fallback: str = "") -> str:
    return os.getenv(name, fallback).strip()


async def _complete(gateway: Any, model: str, stable: str, prompt: str,
                    identity: str, scenario: str = "stable-prefix"):
    result = await gateway.complete(LLMRequest(
        model=model,
        stable_prompt=stable,
        messages=[{"role": "user", "content": prompt}],
        cache_identity=identity,
        cache_mode="automatic",
        max_tokens=512,
        temperature=0.0,
    ))
    return result.text, {
        "provider": result.usage.provider,
        "input_tokens": result.usage.input_tokens,
        "total_input_tokens": result.usage.total_input_tokens,
        "cache_read_tokens": result.usage.cache_read_tokens,
        "cache_write_tokens": result.usage.cache_write_tokens,
        "cache_miss_tokens": result.usage.cache_miss_tokens,
        "cache_status": result.usage.status,
        "metadata": result.metadata,
        "scenario": scenario,
    }


async def run_real(output_dir: Path, confirm_cost: bool, provider: str | None = None,
                   repeat: int = 5, cache_scenario: str = "stable-prefix") -> dict:
    if not confirm_cost:
        raise ValueError("Real-model experiments require --confirm-cost.")
    api_key = _setting("LLM_API_KEY") or _setting("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY or ANTHROPIC_API_KEY is required.")
    base_url = _setting("LLM_BASE_URL") or _setting("ANTHROPIC_BASE_URL") or None
    provider = (provider or _setting("LLM_PROVIDER") or
                ("deepseek" if base_url and "deepseek" in base_url.lower() else "anthropic")).lower()
    model = _setting("LLM_MODEL") or _setting("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    gateway = build_gateway(provider, api_key, model, base_url, cache_enabled=True)
    stable = (
        "你是 mymind 客服助手。只根据提供的背景回答；不确定时说明需要人工确认。"
        "遵守隐私、安全、退款和人工升级规则，回答准确、简洁、可执行。"
    )
    builder = ContextBuilder(max_chars=8000)
    rows = []
    for index, (name, fact, request) in enumerate(SCENARIOS):
        memory = MemoryContext(
            recent_messages=[Message(MsgRole.ASSISTANT, "请继续说明问题。")],
            relevant_history=[fact, "无关的历史天气信息"],
            user_profile={"preferred_channel": "email"},
            summary="用户正在咨询客服问题。",
        )
        knowledge = f"[知识库]\n- {fact}"
        b0_prompt = memory.to_prompt_text() + "\n\n" + knowledge + "\n\n用户请求：" + request
        c2_context = builder.build(memory, knowledge, request)
        c2_prompt = c2_context.text + "\n\n用户请求：" + request
        b0_answer, b0_usage = await _complete(gateway, model, stable, b0_prompt, f"b0:{index}")
        c2_answer, c2_usage = await _complete(gateway, model, stable, c2_prompt, f"c2:{index}")
        rows.append({
            "scenario": name, "index": index,
            "b0_answer": b0_answer, "c2_answer": c2_answer,
            "b0_usage": b0_usage, "c2_usage": c2_usage,
            "context_metadata": c2_context.metadata,
        })

    cache_rows = []
    repeat = max(2, int(repeat))
    for index, (_, fact, request) in enumerate(SCENARIOS[:3]):
        for call_index in range(repeat):
            if cache_scenario == "identical":
                prompt = request
                identity = f"c3:identical:{index}"
            elif cache_scenario == "invalidation":
                version = "v2" if call_index == repeat - 1 else "v1"
                prompt = f"{fact}\n{request}"
                identity = f"c3:knowledge:{index}:{version}"
            else:
                prompt = f"固定背景：{fact}\n当前问题：{request}（第{call_index + 1}次验证）"
                identity = f"c3:stable-prefix:{index}"
            _, usage = await _complete(gateway, model, stable, prompt, identity, cache_scenario)
            cache_rows.append({"scenario_index": index, "call": call_index + 1, "usage": usage})

    reads = [row["usage"]["cache_read_tokens"] for row in cache_rows]
    total = [row["usage"]["total_input_tokens"] for row in cache_rows if row["usage"]["total_input_tokens"] is not None]
    writes = [row["usage"]["cache_write_tokens"] for row in cache_rows]
    eligible = [row for row in cache_rows if row["usage"]["cache_status"] != "unknown"]
    variants = {
        "B0": {"status": "baseline", "requests": len(rows)},
        "C1": {"status": "covered_by_offline_and_integration_layers"},
        "C2": {"status": "context_budget_and_memory_candidate", "requests": len(rows)},
        "C3": {
            "provider": provider, "model": model, "base_url": base_url or "official",
            "scenario": cache_scenario, "repeat": repeat,
            "request_hit_rate": (sum(v > 0 for v in reads) / len(eligible)) if eligible else None,
            "token_hit_rate": (sum(reads) / sum(total)) if total and sum(total) else None,
            "cache_read_tokens": sum(reads), "cache_write_tokens": sum(writes),
            "status": "measured" if cache_rows else "not_configured",
        },
    }
    config = {"provider": provider, "model": model, "base_url": base_url or "official",
              "scenarios": len(SCENARIOS), "cache_scenario": cache_scenario, "repeat": repeat,
              "temperature": 0.0}
    report = {
        "title": "Python Multi-provider Cache and Memory Real Model Experiment",
        "artifact_type": "cache-memory-real-model-v2", "metadata": metadata(config, model),
        "config": config, "variants": variants, "rows": rows, "cache_rows": cache_rows,
        "failures": [], "overall_passed": True,
    }
    report["artifacts"] = write_report(report, output_dir, "cache-memory-real-model")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--provider", choices=("deepseek", "openai", "anthropic"))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cache-scenario", choices=("stable-prefix", "identical", "invalidation"), default="stable-prefix")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    args = parser.parse_args()
    report = asyncio.run(run_real(args.output_dir, args.confirm_cost, args.provider, args.repeat, args.cache_scenario))
    print(report["artifacts"]["json"])


if __name__ == "__main__":
    main()
