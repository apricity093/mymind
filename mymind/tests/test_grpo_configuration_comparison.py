"""有限配置对照的组尺度、归一化、来源与额度回归；不加载模型。"""
import json
from pathlib import Path
import shutil

import pytest

from experiments.agentic_rl.grpo_diagnostics import group_statistics, normalization_evidence
from experiments.agentic_rl.stage7 import RunBudget, check_inputs, read
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards
from tests.agentic_rl_samples import retained_path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "agentic_runs/grpo-standard-20261021"


def test_complete_cross_rank_group_uses_sample_standard_deviation():
    rewards = [0, -1, -1, -1]
    group = group_statistics(rewards, "group")
    assert group["reward_std"] == .5
    assert group["expected_advantages"] == pytest.approx([.75/.5001] + [-.25/.5001]*3)
    assert group_statistics(rewards, "none")["expected_advantages"] == [.75, -.25, -.25, -.25]
    for reward in [-1, 0, .3, 1]:
        assert group_statistics([reward]*4, "group")["expected_advantages"] == [0]*4


def test_grpo_and_dapo_denominators_exclude_tool_observations_and_padding():
    grpo = normalization_evidence("grpo", [2, 5], 20, 2, 1, 2)
    assert grpo["trajectory_denominators"] == [2, 5]
    assert grpo["sample_denominator"] == 2
    assert "dapo_normalizer" not in grpo
    dapo = normalization_evidence("dapo", [2, 5], 20, 2, 1, 2)
    assert dapo["dapo_normalizer"] == 5


def prepared(tmp_path):
    cfg = read(retained_path(RUN / "configs/preflight.json"))
    baseline = read(ROOT / "agentic_runs/stage7-preparation-20261014/baseline.json")
    selection = read(ROOT / "agentic_runs/stage7-preparation-20261014/formal/sft-selection.json")
    paths = [cfg["baseline"], cfg["configuration_comparison"], cfg["sft_selection"],
             baseline["common_pool"], selection["training_output"] + "/configuration.json"]
    for path in paths:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path.replace("runs/", "agentic_runs/", 1), destination)
    tasks = load_public_tasks(ROOT / cfg["tasks"])
    standards = load_private_standards(ROOT / cfg["standards"].replace("runs/", "agentic_runs/", 1))
    return cfg, tasks, standards


def test_comparison_preflight_uses_real_formal_sft_selection(tmp_path):
    cfg, tasks, standards = prepared(tmp_path)
    check_inputs(tmp_path, cfg, tasks, standards, "grpo")
    for changes in ({"limits_file": "runs/stage7-preparation-20261014/formal/limits.json"},
                    {"sft_adapter": "rl40"}, {"baseline": "new-baseline.json"}):
        with pytest.raises((ValueError, FileNotFoundError)):
            check_inputs(tmp_path, {**cfg, **changes}, tasks, standards, "grpo")


def test_retry_trajectories_have_independent_bound_and_count_api(tmp_path):
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(dict(limits=dict(student_trajectories=8,
        infrastructure_trajectories=4, api_request_allowance=144), consumed={}, events=[])), encoding="utf-8")
    ledger = RunBudget(tmp_path, {"limits_file": "limits.json"})
    ledger.reserve("student_trajectories", 8, "normal")
    ledger.reserve("infrastructure_trajectories", 4, "same-policy-retry")
    with pytest.raises(RuntimeError, match="额度不足"):
        ledger.reserve("infrastructure_trajectories", 4, "extra")
    assert read(path)["consumed"] == dict(student_trajectories=8,
        infrastructure_trajectories=4, api_request_allowance=144)


def test_summary_preserves_readable_disposition_on_historical_first_records(monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    from summarize_grpo_comparison import evaluation, disposition
    directory = ROOT / "agentic_runs/stage7-preparation-20261014/formal/grpo-validation-40"
    retained_path(directory / "first_records.jsonl")
    summary,_=evaluation(directory)
    assert summary["complete_success"]==49
    assert summary["primary_disposition_correct"]==67
    assert disposition({"raw_result":{"disposition":"request_info","facts":"invalid"}})=="request_info"
    assert disposition({"raw_result":'{"disposition":"escalate","facts":"invalid"}'})=="escalate"
