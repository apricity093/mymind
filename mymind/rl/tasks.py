"""分别装载公开任务、私有标准与实际正文支持映射。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from agents.customer_policy.protocol import ProtocolModel, PublicTask


RESOURCES = Path(__file__).resolve().parent / "resources"


class Expectation(ProtocolModel):
    section: Literal["reasons", "facts", "next_steps", "missing_info"]
    alternatives: list[dict] = Field(min_length=1)
    supports: list[list[str]] = Field(min_length=1)
    required: bool = True
    decisive: bool = False


class PrivateStandard(ProtocolModel):
    task_id: str
    group_id: str
    split: Literal["candidate", "dev", "train", "validation", "test"]
    dispositions: list[Literal["answer", "request_info", "escalate"]]
    materials: dict[str, str]
    expectations: list[Expectation]
    severe_items: dict[str, list[dict]] = Field(default_factory=dict)
    # 经题面核查确未提供的用户事实；不因此把无关 unknown 项加入允许输出。
    unprovided_user_facts: dict[str, list[str]] = Field(default_factory=dict)
    # 复合用户事实逐分量的来源状态；不扩展模型可见输出协议。
    user_fact_components: dict[str, dict[str, dict[str, dict]]] = Field(default_factory=dict)


class PolicyClause(ProtocolModel):
    clause_id: str
    source_id: str
    text: str = Field(min_length=1)


def _load(path: str | Path, model, key: str):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [model.model_validate(value) for value in values]
    result = {getattr(row, key): row for row in rows}
    if len(rows) != len(result):
        raise ValueError(f"{path} 包含重复 {key}")
    return result


def load_public_tasks(path: str | Path) -> dict[str, PublicTask]:
    return _load(path, PublicTask, "task_id")


def load_private_standards(path: str | Path) -> dict[str, PrivateStandard]:
    return _load(path, PrivateStandard, "task_id")


def load_policy_support(path: str | Path) -> dict[str, PolicyClause]:
    return _load(path, PolicyClause, "clause_id")


def validate_task_pairs(tasks, standards, clauses):
    if set(tasks) != set(standards):
        raise ValueError("公开任务与私有标准 task_id 不对应")
    for task_id, task in tasks.items():
        standard = standards[task_id]
        actual = {"user:1": task.user_text,
                  **{record.reference_id: record.content for record in task.records}}
        if any(standard.materials.get(ref) != text for ref, text in actual.items()):
            raise ValueError(f"{task_id} 私有标准与公开材料不一致")
        for expected in standard.expectations:
            if any(not group or any(s not in standard.materials and s not in clauses
                                    and s != "retrieval_unavailable" for s in group)
                   for group in expected.supports):
                raise ValueError(f"{task_id} 支持映射不完整")
