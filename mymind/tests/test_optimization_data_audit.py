"""定向审计的分母、时间边界、双rank语义和原件缺失口径。"""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_optimization_data import (aggregate_tools, audit_group, candidate_coverage,
    deduplicate_groups, distribution, expected_bucket, fraction, lines, select_prefix, tool_summary)


def test_distribution_uses_task_denominators_and_bucket_selection_rate():
    before = [{"role": "billing"}]*3 + [{"role": "technical"}]
    after = [{"role": "billing"}, {"role": "technical"}]
    a, b = distribution(before, after, "role")
    assert a["train_share"] == fraction(3, 4)
    assert a["selected_share"] == fraction(1, 2)
    assert a["selection_rate"] == fraction(1, 3)
    assert a["share_difference_pp"] == -25
    assert b["selection_rate"] == fraction(1, 1)


def test_allowed_disposition_set_is_one_bucket():
    assert expected_bucket({"dispositions": ["escalate", "answer"]}) == "answer|escalate"


def test_zero_selected_bucket_is_retained():
    a, b = distribution([{"g": "a"}, {"g": "b"}], [{"g": "a"}], "g")
    assert b["selected_count"] == 0
    assert b["selection_rate"] == fraction(0, 1)
    assert fraction(0, 0)["fraction"] is None


def test_prefix_is_policy_before_checkpoint_and_excludes_faults():
    def group(step, failed=False):
        return {0: (None, {"policy_step": step, "infrastructure_failure": failed})}
    groups = [group(0), group(38), group(40), group(78), group(80), group(10, True)]
    assert select_prefix(groups, 40) == groups[:2]
    assert select_prefix(groups, 80) == groups[:4]


def rank_pair(tmp_path):
    base = dict(policy_step=0, attempt=0, task_ids=["task"]*4, rows=[{"task_id": "task"}]*4,
                rewards=[0, 1, 0, 1], advantages=[-.5, .5, -.5, .5], effective_model_tokens=4,
                optimization_model_tokens=4, infrastructure_failure=False, nonzero_advantage=True)
    paths = [tmp_path / f"group-step0-task-attempt0-rank{rank}.json" for rank in (0, 1)]
    for rank, path in enumerate(paths):
        path.write_text(json.dumps(dict(base, completion_ids=[[rank]], completion_mask=[[1]])))
    return paths


def test_rank_copies_deduplicate_rows_without_equating_local_tokens(tmp_path):
    groups = deduplicate_groups(reversed(rank_pair(tmp_path)))
    assert len(groups) == 1 and len(groups[0][0][1]["rows"]) == 4
    assert groups[0][0][1]["completion_ids"] != groups[0][1][1]["completion_ids"]


def test_missing_rank_does_not_silently_report_half_tokens(tmp_path):
    with pytest.raises(AssertionError, match="rank"):
        deduplicate_groups(rank_pair(tmp_path)[:1])


def test_conflicting_rank_rewards_fail(tmp_path):
    paths = rank_pair(tmp_path)
    row = json.loads(paths[1].read_text())
    row["rewards"] = [1]*4
    paths[1].write_text(json.dumps(row))
    with pytest.raises(AssertionError):
        deduplicate_groups(paths)


def test_unavailable_candidate_is_unknown_not_failed_or_uncollected():
    result = candidate_coverage([{"task_id": "a"}, {"task_id": "b"}],
        [{"task_id": "a", "candidate": 2}], [{"task_id": "a", "candidate": 2}], {"candidates": 3})
    assert result["original_records_unavailable"] == 2
    assert result["failed_candidates_recomputed"] is None
    assert result["qualified_candidates_total_recomputed"] is None
    assert result["tasks"][1]["uncollected"] is None
    assert not result["rejection_funnel_complete"]


def sample(after=False):
    call = {"function": {"name": "search", "arguments": {"query": "x"}}}
    observation = {"status": {"state": "candidates"}, "snippets": []}
    tool = {"role": "tool", "content": json.dumps(observation)}
    action = {"role": "assistant", "content": "", "tool_calls": [call]}
    messages = ([action, tool, action, tool] if after else
                [dict(action, tool_calls=[call, call]), tool, tool])
    messages.append({"role": "assistant", "content": '{"disposition":"answer"}', "tool_calls": None})
    return dict(completion=messages, trajectory={"searches": [{"observation": observation}]*2})


def test_batched_searches_are_not_observation_driven():
    assert tool_summary(sample())["after_observation_requests"] == 0
    assert tool_summary(sample(True))["after_observation_requests"] == 1
    assert aggregate_tools([sample(), sample(True)])["after_observation_trajectories"] == fraction(1, 2)


def test_terminal_parse_failure_does_not_erase_tools():
    row = sample()
    row["completion"][-1]["content"] = "broken json"
    result = tool_summary(row)
    assert result["search_requests"] == result["actual_observations"] == 2
    assert result["terminal_disposition"] is None


def test_missing_required_log_is_not_an_empty_log(tmp_path):
    with pytest.raises(FileNotFoundError):
        lines(tmp_path / "missing.jsonl")


def test_actual_token_slices_are_rank_order_independent():
    directory = ROOT / "agentic_runs/stage7-preparation-20261014/formal/grpo"
    paths = sorted(directory.glob("group-step0-*-rank*.json"))
    if not paths:
        pytest.skip("正式本地训练原件未随公开代码提供")
    a = audit_group(deduplicate_groups(paths)[0], directory)
    b = audit_group(deduplicate_groups(reversed(paths))[0], directory)
    for key in ("effective_policy_tokens", "nonzero_advantage_policy_tokens", "token_slices"):
        assert a[key] == b[key]
    assert a["effective_policy_tokens"] == 1932
    assert a["completed_steps"] == [1, 2]


def test_actual_training_prefixes_exclude_eight_post_training_trajectories():
    directory = ROOT / "agentic_runs/stage7-preparation-20261014/formal/grpo"
    paths = list(directory.glob("group-step*-rank*.json"))
    if not paths:
        pytest.skip("正式本地训练原件未随公开代码提供")
    groups = deduplicate_groups(paths)
    assert len(groups) == 42
    for stop, count in ((40, 20), (80, 40)):
        prefix = select_prefix(groups, stop)
        assert len(prefix) == count
        assert len({r["task_id"] for g in prefix for r in g[0][1]["rows"]}) == count
        assert sum(len(g[0][1]["rows"]) for g in prefix) == count*4
    assert sum(len(g[0][1]["rows"]) for g in groups if g[0][1]["policy_step"] == 80) == 8
