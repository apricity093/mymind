from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content
from core.prompt_cache import PromptCachePolicy
from experiments.reporting import metadata, write_report
from memory.context_builder import ContextBuilder
from memory.conversation_memory import MemoryContext, Message, MsgRole


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


async def _complete(client, model, system, prompt):
    response = await client.messages.create(
        model=model, max_tokens=512, temperature=0.0,
        system=system, messages=[{"role": "user", "content": prompt}],
    )
    usage = getattr(response, "usage", None)
    return extract_text_content(response.content), {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    }


async def run_real(output_dir: Path, confirm_cost: bool) -> dict:
    if not confirm_cost:
        raise ValueError("Real-model experiments require --confirm-cost.")
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is required.")
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
    base_url = os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
    client = AsyncAnthropic(api_key=api_key, **({"base_url": base_url} if base_url else {}))
    system = "你是客服助手。只根据提供的背景回答，不确定时说明需要人工确认。"
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
        b0_answer, b0_usage = await _complete(client, model, system, b0_prompt)
        c2_answer, c2_usage = await _complete(client, model, system, c2_prompt)
        candidate_label = "B" if index % 2 == 0 else "A"
        answer_a, answer_b = (b0_answer, c2_answer) if candidate_label == "B" else (c2_answer, b0_answer)
        judge_prompt = (
            f"事实：{fact}\n请求：{request}\n答案A：{answer_a}\n答案B：{answer_b}\n"
            "返回JSON：{\"winner\":\"A|B|tie\",\"a_acceptable\":true,\"b_acceptable\":true}"
        )
        judge_text, judge_usage = await _complete(client, model, "你是严格的客服答案评审。", judge_prompt)
        try:
            start, end = judge_text.find("{"), judge_text.rfind("}") + 1
            judgement = json.loads(judge_text[start:end])
        except Exception:
            judgement = {"winner": "invalid", "a_acceptable": False, "b_acceptable": False}
        candidate_acceptable = judgement.get("b_acceptable") if candidate_label == "B" else judgement.get("a_acceptable")
        baseline_acceptable = judgement.get("a_acceptable") if candidate_label == "B" else judgement.get("b_acceptable")
        candidate_won = judgement.get("winner") == candidate_label
        rows.append({
            "scenario": name, "index": index, "b0_answer": b0_answer, "c2_answer": c2_answer,
            "judgement": judgement, "b0_usage": b0_usage, "c2_usage": c2_usage,
            "judge_usage": judge_usage, "context_metadata": c2_context.metadata,
            "candidate_label": candidate_label,
            "candidate_acceptable": bool(candidate_acceptable),
            "baseline_acceptable": bool(baseline_acceptable),
            "candidate_won": candidate_won,
        })

    non_inferior = sum(row["candidate_acceptable"] for row in rows) / len(rows)
    wins = sum(row["candidate_won"] for row in rows) / len(rows)
    cache_policy = PromptCachePolicy(enabled=base_url is None, min_stable_chars=4096)
    _, cache_meta = cache_policy.build_system(system)
    variants = {
        "B0": {"acceptable_rate": sum(row["baseline_acceptable"] for row in rows) / len(rows)},
        "C1": {"status": "covered_by_offline_and_integration_layers"},
        "C2": {"non_inferior_rate": non_inferior, "win_rate": wins},
        "C3": {
            "status": "eligible" if cache_meta["prompt_cache_eligible"] else "not_applicable",
            "reason": "stable_prefix_below_provider_minimum" if not cache_meta["prompt_cache_eligible"] else "eligible",
        },
    }
    failures = []
    if non_inferior < 0.9:
        failures.append("c2_non_inferior_rate")
    if wins < 0.6:
        failures.append("c2_win_rate")
    config = {"scenarios": len(SCENARIOS), "temperature": 0.0, "model": model, "base_url": base_url or "official"}
    report = {
        "title": "Python Cache and Memory Real Model Experiment",
        "artifact_type": "cache-memory-real-model-v1",
        "metadata": metadata(config, model), "config": config, "variants": variants,
        "rows": rows, "failures": failures, "overall_passed": not failures,
    }
    report["artifacts"] = write_report(report, output_dir, "cache-memory-real-model")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    args = parser.parse_args()
    report = asyncio.run(run_real(args.output_dir, args.confirm_cost))
    print(report["artifacts"]["json"])
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
