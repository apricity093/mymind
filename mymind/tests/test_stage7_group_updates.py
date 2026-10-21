"""受控更新记录验证完整同题组的暂停判断，不加载或调用模型。"""
import pytest

from experiments.agentic_rl.grpo import BatchPaused, GroupUpdateEvidence
from experiments.agentic_rl.stage7 import RunBudget, read
from experiments.agentic_rl.runtime import write_json


def group(step=0, nonzero=True):
    return dict(policy_step=step, attempt=0, task_ids=["task"]*4,
        rewards=[-1,0,0,-1] if nonzero else [0]*4,
        advantages=[-.5,.5,.5,-.5] if nonzero else [0]*4, nonzero_advantage=nonzero)


def updates(step, gradient=0, delta=0):
    return [dict(step=step, gradient_norm=gradient, gradient_finite=True, parameter_delta_l1=delta)
        for _ in range(2)]


@pytest.mark.parametrize("first,second", [((0,0),(1,.1)), ((1,.1),(0,0)), ((1,0),(0,.1))])
def test_zero_slice_does_not_pause_a_complete_updated_group(tmp_path,first,second):
    monitor=GroupUpdateEvidence(tmp_path,0)
    monitor.begin(group(),2)
    monitor.record(updates(1,*first))
    assert not list(tmp_path.glob("group-updates-*.json"))
    monitor.record(updates(2,*second))
    result=read(tmp_path / "group-updates-step0-rank0.json")
    assert result["completed_steps"]==[1,2] and not result["paused"]
    assert result["nonzero_gradient"] and result["parameter_changed"]


@pytest.mark.parametrize("gradient,delta,missing", [(0,0,2),(1,0,1),(0,.1,1)])
def test_nonzero_advantage_without_group_update_pauses_after_two_steps(tmp_path,gradient,delta,missing):
    ledger_path=tmp_path / "limits.json"
    write_json(ledger_path,dict(limits=dict(student_trajectories=4,api_request_allowance=48,
        grpo_updates=2,gpu_seconds_reserved=100),consumed={},events=[]))
    ledger=RunBudget(tmp_path,dict(limits_file="limits.json"))
    ledger.reserve("gpu_seconds_reserved",100,"command")
    ledger.reserve("student_trajectories",4,"group")
    monitor=GroupUpdateEvidence(tmp_path,0)
    monitor.begin(group(),2)
    ledger.reserve("grpo_updates",1,"optimizer-step-1")
    monitor.record(updates(1,gradient,delta))
    ledger.reserve("grpo_updates",1,"optimizer-step-2")
    before=ledger_path.read_bytes()
    with pytest.raises(BatchPaused,match="完整组"):
        monitor.record(updates(2,gradient,delta))
    assert ledger_path.read_bytes()==before
    assert read(ledger_path)["consumed"]==dict(student_trajectories=4,api_request_allowance=48,
        grpo_updates=2,gpu_seconds_reserved=100)
    result=read(tmp_path / "group-updates-step0-rank0.json")
    assert result["paused"] and len(result["missing"])==missing and result["completed_steps"]==[1,2]
    assert len(result["rank_updates"])==2
    with pytest.raises(RuntimeError,match="额度不足"):
        RunBudget(tmp_path,dict(limits_file="limits.json")).reserve("grpo_updates",1,"retry")


def test_group_uses_both_ranks_and_does_not_reuse_previous_update(tmp_path):
    monitor=GroupUpdateEvidence(tmp_path,1)
    monitor.begin(group(40),2)
    rows=updates(41)
    rows[1].update(gradient_norm=1,parameter_delta_l1=.1)
    monitor.record(rows)
    monitor.record(updates(42))
    assert not read(tmp_path / "group-updates-step40-rank1.json")["paused"]
    monitor.begin(group(42),2)
    monitor.record(updates(43))
    with pytest.raises(BatchPaused):
        monitor.record(updates(44))
    assert read(tmp_path / "group-updates-step42-rank1.json")["completed_steps"]==[43,44]


def test_zero_advantage_group_is_recorded_without_new_update_pause(tmp_path):
    monitor=GroupUpdateEvidence(tmp_path,0)
    monitor.begin(group(nonzero=False),2)
    monitor.record(updates(1))
    monitor.record(updates(2))
    result=read(tmp_path / "group-updates-step0-rank0.json")
    assert not result["paused"] and not result["nonzero_gradient"] and not result["parameter_changed"]
