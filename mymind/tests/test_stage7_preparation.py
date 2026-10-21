"""正式入口的划分、公平分母、选模和一次检索回归。"""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mymind"))
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import parse_qwen
from agents.customer_policy.protocol import PublicTask, SearchObservation
from agents.customer_policy.runner import run_task
from experiments.agentic_rl.metrics import summarize
from experiments.agentic_rl.stage7 import check_inputs, read, RunBudget, select_checkpoint
from experiments.agentic_rl.runtime import write_json, apply_resume
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards
from tests.agentic_rl_samples import retained_path

WORK = Path(os.environ.get("MYMIND_WORK_ROOT", str(ROOT)))
RUNS = WORK / ("runs" if os.environ.get("MYMIND_WORK_ROOT") else "agentic_runs")
PREP = RUNS / "stage7-preparation-20261014"


@pytest.fixture
def inputs(tmp_path):
    cfg = read(retained_path(PREP / "configs/formal-sft.json"))
    baseline = read(PREP / "baseline.json")
    for target, source in ((cfg["baseline"], PREP / "baseline.json"),
        (baseline["common_pool"], RUNS / "pre-stage7-fixes-20261014/common_pool.json")):
        dest = tmp_path / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    tasks = load_public_tasks(WORK / "data/agentic_rl/formal/public_tasks.json")
    standards = load_private_standards(RUNS / "pre-stage7-fixes-20261014/standards/formal/private_standards.v5.json")
    return tmp_path, cfg, tasks, standards


def test_formal_pool_and_train_split_are_checked(inputs):
    root, cfg, tasks, standards = inputs
    check_inputs(root,cfg,tasks,standards,"sft")
    for bad in (cfg["task_ids"][:-1], [next(t for t,s in standards.items() if s.split == "validation")]):
        with pytest.raises(ValueError):
            check_inputs(root,{**cfg,"task_ids":bad},tasks,standards,"sft")


@pytest.mark.parametrize("change", [{"standards":"private_standards.v4.json"}, {"sft_adapter":"old-dev/adapter"},
    {"run_purpose":"grpo"}, {"turn_tokens":1024}, {"tasks":"other.json"}, {"run_kind":"dev"}, {"limits_file":"reset.json"}])
def test_wrong_baseline_or_origin_rejected(inputs,change):
    root,cfg,tasks,standards = inputs
    with pytest.raises(ValueError):
        check_inputs(root,{**cfg,**change},tasks,standards,"sft")


def test_rl_requires_selected_formal_sft(inputs):
    root,_,tasks,standards = inputs
    cfg = read(PREP / "configs/formal-grpo.json")
    cfg.pop("sft_adapter", None)
    cfg.pop("sft_selection", None)
    with pytest.raises(ValueError,match="显式"):
        check_inputs(root,cfg,tasks,standards,"grpo")
    cfg.update(sft_adapter="preflight/adapter",sft_selection="selection.json")
    write_json(root / "selection.json",dict(phase="sft",baseline=cfg["baseline"], selected_adapter="preflight/adapter",training_output="preflight"))
    (root / "preflight").mkdir()
    write_json(root / "preflight/configuration.json",dict(run_kind="preflight",run_purpose="sft",task_ids=cfg["task_ids"]))
    with pytest.raises(ValueError,match="正式起点"):
        check_inputs(root,cfg,tasks,standards,"grpo")


def test_equal_group_metric_retains_faults_and_unfinished():
    tasks={t: SimpleNamespace(role="billing") for t in ("a","b","c","d")}
    standards={t: SimpleNamespace(group_id="R01" if t in {"a","b","c"} else "R02",dispositions=["answer"]) for t in tasks}
    rows=[{"task_id":"a","grade":{"complete_success":True,"reward":1}},
          {"task_id":"b","grade":{"complete_success":True,"reward":1},"infrastructure_failure":True},
          {"task_id":"d","grade":{"severe_error":True,"reward":-1}}]
    metrics=summarize(list(tasks),rows,tasks,standards)
    assert metrics["combination_equal_success_rate"] == pytest.approx(1/6)
    assert metrics["instance_success_rate"] == .25
    assert metrics["combination_equal_severe_error_rate"] == .5
    assert metrics["unfinished_tasks"] == ["c"] and metrics["infrastructure_failures"] == 1
    assert metrics["all_task_cost"]["tasks"] == 4


def test_selection_is_success_then_severe_then_earlier_and_updated(tmp_path):
    directory=tmp_path / "train"
    directory.mkdir()
    write_json(directory / "configuration.json",dict(run_kind="formal",run_purpose="sft",baseline="baseline.json"))
    (directory / "updates-rank0.jsonl").write_text(json.dumps(dict(step=1,parameter_delta_l1=2))+"\n")
    nodes=[]
    for step,success,severe in ((40,.5,.2),(80,.5,.1),(120,.5,.1)):
        evaluation=f"eval-{step}"
        (tmp_path / evaluation).mkdir()
        write_json(tmp_path / evaluation / "metrics.json",dict(all_tasks_finished=True,infrastructure_failures=0,combination_equal_success_rate=success,combination_equal_severe_error_rate=severe))
        write_json(tmp_path / evaluation / "configuration.json",dict(adapter_path=f"train/checkpoint-{step}",baseline="baseline.json",evaluation_split="validation",search_limit=3))
        (directory / f"checkpoint-{step}").mkdir()
        write_json(directory / f"checkpoint-{step}/trainer_state.json",dict(global_step=step))
        nodes.append(dict(step=step,evaluation=evaluation))
    plan=dict(phase="sft",training_output="train",nodes=nodes,selection_output="selection.json")
    write_json(tmp_path / "plan.json",plan)
    assert select_checkpoint(tmp_path,tmp_path / "plan.json")["selected_step"] == 80
    (directory / "updates-rank0.jsonl").write_text(json.dumps(dict(step=1,parameter_delta_l1=0))+"\n")
    with pytest.raises(ValueError,match="实际参数更新"):
        select_checkpoint(tmp_path,tmp_path / "plan.json")


@pytest.mark.parametrize("calls", [0,1,2])
def test_one_search_counts_requests_and_preserves_final(calls):
    class Retrieval:
        async def search(self,query,number):
            return SearchObservation(request_number=number,status={"reference_id":"search:1:status","state":"empty"},snippets=[]),{}
    class Generation:
        def __init__(self): self.messages=[]
        async def generate(self,messages,tools_enabled=True):
            self.messages.append((deepcopy(messages),tools_enabled))
            if len(self.messages)==1 and calls:
                return parse_qwen('<tool_call>{"name":"search","arguments":{"query":"政策"}}</tool_call>'*calls)
            return parse_qwen('{"disposition":"answer","reasons":[],"facts":[],"next_steps":[],"missing_info":[]}')
    env=PolicyEnvironment(Retrieval()); env.max_searches=1
    generation=Generation()
    task=PublicTask(task_id="x",role="billing",user_text="问题")
    row=asyncio.run(run_task(task,generation,env))
    assert generation.messages[0][0] == task.messages(max_searches=1)
    if calls==0:
        assert not env.searches and len(generation.messages)==1 and row["runtime_error"] is None
    elif calls==1:
        assert len(generation.messages)==2 and generation.messages[1][1] is False
        assert generation.messages[1][0][-2]["role"]=="tool" and row["runtime_error"] is None
    else:
        assert not env.searches and row["runtime_error"]["kind"]=="protocol_error"


def test_search_prompt_matches_limit_and_frozen_student_inputs():
    rows=read(retained_path(RUNS / "pre-stage7-fixes-20261014/stage6/student_data.json"))
    tasks=load_public_tasks(WORK / "data/agentic_rl/formal/public_tasks.json")
    assert len(rows)==158
    for row in rows:
        task=tasks[row["task_id"]]
        three=task.messages()
        assert three == row["messages"][:len(three)]
        env=PolicyEnvironment(None)
        assert env.reset(task)==three
        env.max_searches=1
        one=env.reset(task)
        assert "最多一次检索" in one[0]["content"]
        assert one[0]["content"].replace("最多一次检索", "最多三次检索") == three[0]["content"]
        assert one[1:]==three[1:]


def test_shared_budget_relaunch_does_not_reset(tmp_path):
    write_json(tmp_path / "limits.json",dict(limits={"student_trajectories":4,"api_request_allowance":48},consumed={},events=[]))
    RunBudget(tmp_path,{"limits_file":"limits.json"}).reserve("student_trajectories",4,"first")
    with pytest.raises(RuntimeError):
        RunBudget(tmp_path,{"limits_file":"limits.json"}).reserve("student_trajectories",1,"retry")
    assert read(tmp_path / "limits.json")["consumed"]["student_trajectories"]==4


def test_platform_reason_missing_standard_is_not_implicitly_unknown():
    from experiments.agentic_rl.grading import _reason_conflicts
    from tests.agentic_rl_samples import STANDARDS
    standard=STANDARDS["S07"].model_copy(deep=True)
    target=next(a["target"] for e in standard.expectations for a in e.alternatives if a.get("type")=="request_scope")
    standard.expectations=[e for e in standard.expectations if not any(a.get("type")=="feature_enabled" for a in e.alternatives)]
    standard.unprovided_user_facts={}
    assert not _reason_conflicts({"target":target,"code":"platform_enablement_required"},standard)


def test_resume_stays_in_current_job_without_mutating_frozen_config(tmp_path):
    output=tmp_path / "formal/sft"; checkpoint=output / "checkpoint-40"
    checkpoint.mkdir(parents=True)
    write_json(checkpoint / "trainer_state.json",dict(global_step=40))
    cfg=dict(run_purpose="sft")
    assert apply_resume(tmp_path,cfg,output,"formal/sft/checkpoint-40")["resume_from_checkpoint"]=="formal/sft/checkpoint-40"
    assert "resume_from_checkpoint" not in cfg
    with pytest.raises(ValueError,match="同一作业"):
        apply_resume(tmp_path,cfg,tmp_path / "other","formal/sft/checkpoint-40")


def test_infrastructure_diagnostic_keeps_first_table_and_rejects_normal_low_score(inputs):
    root,_,tasks,standards=inputs
    cfg=read(PREP / "configs/formal-base.json")
    cfg={**cfg,"output_dir":"first"}
    source=root / "first"; source.mkdir()
    write_json(source / "configuration.json",cfg)
    tid=cfg["task_ids"][0]
    diagnostic={**cfg,"run_kind":"diagnostic","task_ids":[tid],"diagnostic_of":"first","output_dir":"separate"}
    first=source / "first_records.jsonl"
    first.write_text(json.dumps(dict(task_id=tid,infrastructure_failure=False,grade={"reward":0}))+"\n")
    with pytest.raises(ValueError,match="正常低分"):
        check_inputs(root,diagnostic,tasks,standards,"evaluate")
    first.write_text(json.dumps(dict(task_id=tid,infrastructure_failure=True))+"\n")
    check_inputs(root,diagnostic,tasks,standards,"evaluate")
    with pytest.raises(ValueError):
        check_inputs(root,{**diagnostic,"output_dir":"first"},tasks,standards,"evaluate")
