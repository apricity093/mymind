"""阶段8只消费新增授权，累计账本仍保留阶段7。"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.agentic_rl.stage7 import RunBudget


def test_stage8_does_not_spend_old_remaining(tmp_path):
    ledger = {"limits": {"student_trajectories": 20, "api_request_allowance": 240},
              "consumed": {"student_trajectories": 5, "api_request_allowance": 60},
              "events": [{"kind": "old", "amount": 5}]}
    (tmp_path / "limits.json").write_text(json.dumps(ledger))
    authorization = {"additional_limits": {"student_trajectories": 1, "api_request_allowance": 12},
                     "consumed_before": ledger["consumed"]}
    (tmp_path / "authorization.json").write_text(json.dumps(authorization))
    config = {"limits_file": "limits.json", "stage_authorization": "authorization.json"}
    RunBudget(tmp_path, config).reserve("student_trajectories", 1, "first")
    with pytest.raises(RuntimeError, match="阶段新增授权"):
        RunBudget(tmp_path, config).reserve("student_trajectories", 1, "second")
    after = json.loads((tmp_path / "limits.json").read_text())
    assert after["consumed"] == {"student_trajectories": 6, "api_request_allowance": 72}
    assert after["events"][0] == ledger["events"][0]


def test_stage8_api_and_zero_updates_are_independent_limits(tmp_path):
    (tmp_path / "limits.json").write_text(json.dumps({"limits": {"student_trajectories": 100, "api_request_allowance": 1200, "grpo_updates": 100}, "consumed": {}, "events": []}))
    (tmp_path / "authorization.json").write_text(json.dumps({"additional_limits": {"student_trajectories": 10, "api_request_allowance": 0, "grpo_updates": 0}, "consumed_before": {}}))
    budget = RunBudget(tmp_path, {"limits_file": "limits.json", "stage_authorization": "authorization.json"})
    for key in ("student_trajectories", "grpo_updates"):
        with pytest.raises(RuntimeError, match="阶段新增授权"):
            budget.reserve(key, 1, "not-authorized")
    assert not json.loads((tmp_path / "limits.json").read_text())["events"]
