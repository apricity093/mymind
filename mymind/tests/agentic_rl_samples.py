"""协议与判分测试用的手工观察，不是教师真实轨迹。"""

from copy import deepcopy
from pathlib import Path

from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.tasks import (
    RESOURCES, load_policy_support, load_private_standards, load_public_tasks,
)


TASKS = load_public_tasks(RESOURCES / "dev_tasks.json")
STANDARDS = load_private_standards(RESOURCES / "dev_standards.json")
CLAUSES = load_policy_support(RESOURCES / "policy_support.json")
KNOWLEDGE = (Path(__file__).resolve().parents[1]
             / "agents/customer_policy/resources/knowledge.md").read_text(encoding="utf-8")
BILLING, TECHNICAL = KNOWLEDGE.split("### 接口认证、授权与故障咨询", 1)


def trajectory(task_id, billing_text=BILLING, technical_text=TECHNICAL):
    return VisibleTrajectory.model_validate({
        "task_id": task_id,
        "materials": [m.model_dump() for m in TASKS[task_id].materials()],
        "searches": [{"action": {"query": "客服政策"}, "observation": {
            "request_number": 1,
            "status": {"reference_id": "search:1:status", "state": "candidates"},
            "snippets": [
                {"reference_id": "search:1:1", "source_id": "billing-policy",
                 "title": "退款到账与扣款咨询", "section_path": [], "content": billing_text},
                {"reference_id": "search:1:2", "source_id": "technical-policy",
                 "title": "接口认证、授权与故障咨询", "section_path": [], "content": technical_text},
            ],
        }}],
    })


def accepted_result(task_id, optional=False):
    """将私有必要项转换为测试输入，用于检查全部资源的装载与一致性。"""
    standard = STANDARDS[task_id]
    result = {"disposition": standard.dispositions[0], "reasons": [], "facts": [],
              "next_steps": [], "missing_info": []}
    for expected in standard.expectations:
        if not optional and not expected.required:
            continue
        item = deepcopy(expected.alternatives[0])
        refs = []
        for group in expected.supports:
            support = group[0]
            ref = support if support in standard.materials else (
                "search:1:1" if CLAUSES[support].source_id == "billing-policy" else "search:1:2")
            if ref not in refs:
                refs.append(ref)
        item["evidence"] = refs
        if item not in result[expected.section]:
            result[expected.section].append(item)
    return result


def s04_result():
    """独立手写 S04 完整结果，对照银行卡 4 日与 5 日上限。"""
    user = ["user:1"]
    policy = ["search:1:1"]
    both = user + policy
    return {
        "disposition": "answer",
        "reasons": [{"target": "R1", "code": "within_refund_window", "evidence": both}],
        "facts": [
            {"target": "R1", "type": "payment_channel", "value": "bank_card", "evidence": user},
            {"target": "R1", "type": "approval_status", "value": "approved", "evidence": user},
            {"target": "R1", "type": "elapsed_workdays", "value": 4, "evidence": user},
            {"target": "R1", "type": "timer_start", "value": "approval", "evidence": policy},
            {"target": "R1", "type": "limit_workdays", "value": 5, "evidence": policy},
            {"target": "R1", "type": "within_window", "value": True, "evidence": both},
        ],
        "next_steps": [
            {"target": "R1", "code": "wait_within_window", "evidence": both},
            {"target": "R1", "code": "verify_refund_if_overdue_unreceived",
             "params": {"limit_workdays": 5}, "evidence": both},
        ],
        "missing_info": [],
    }
def retained_path(path):
    """历史轨迹复核需要本地记录；公开仓库仅运行随附输入的回归。"""
    import pytest

    if not path.is_file():
        pytest.skip(f"需要本地保留的实验记录：{path.name}")
    return path
