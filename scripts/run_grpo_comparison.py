"""本次配置对照的有限命令、来源核验、独立记账和执行记录。"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/"mymind"))
from experiments.agentic_rl.runtime import write_json
from experiments.agentic_rl.stage7 import read, RunBudget, check_inputs, select_checkpoint
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards
from stage8_command import balance

ROOT=Path(os.environ.get("MYMIND_WORK_ROOT", SOURCE))
RUN=ROOT/"runs/grpo-standard-20261021"


def inspect_environment():
    from trl import GRPOTrainer, GRPOConfig
    from trl.trainer.grpo_trainer import nanstd, use_adapter
    directory=RUN/"implementation"
    directory.mkdir(exist_ok=True)
    shutil.copy2(inspect.getfile(GRPOTrainer),directory/"installed_grpo_trainer.py")
    shutil.copy2(inspect.getfile(GRPOConfig),directory/"installed_grpo_config.py")
    (directory/"installed-statistics-and-adapter.txt").write_text(inspect.getsource(nanstd)+"\n"+inspect.getsource(use_adapter),encoding="utf-8")
    config=read(RUN/"configs/train.json")
    historical=ROOT/"runs/stage7-preparation-20261014/formal/grpo"
    versions={n:importlib.metadata.version(n) for n in ("torch","trl","transformers","peft","accelerate")}
    source_comparison=[]
    for relative in (SOURCE/"scripts/grpo_comparison_sync_files.txt").read_text(encoding="utf-8").splitlines():
        if not relative or relative.startswith("#"):
            continue
        before=historical/"source"/relative
        current=SOURCE/relative
        source_comparison.append(dict(path=relative,historical_available=before.exists(),
            equal=before.read_bytes()==current.read_bytes() if before.exists() else None))
    # 对公开输入、标准和政策逐字节核对冻结产物；不新增摘要替代读取。
    input_comparison={}
    for key in ("tasks","standards","policy_support"):
        current=ROOT/config[key]
        equal=(historical/"inputs"/current.name).read_bytes()==current.read_bytes()
        input_comparison[key]=equal
        if not equal:
            raise ValueError(f"冻结输入已变化: {key}")
    adapter=ROOT/config["sft_adapter"]
    if not (adapter/"adapter_model.safetensors").is_file():
        raise ValueError("指定 SFT120 目录没有完整 adapter 权重")
    record=dict(versions=versions,sft_adapter=str(adapter),sft_weight_bytes=(adapter/"adapter_model.safetensors").stat().st_size,
        base_revision=read(ROOT/config["model_dir"]/"base_revision.json"),input_comparison=input_comparison,
        source_comparison=source_comparison,defaults={k:getattr(GRPOConfig(output_dir=str(RUN/"unused")),k) for k in
            ("importance_sampling_level","use_bias_correction_kl","sync_ref_model","entropy_coef","multi_objective_aggregation","delta","top_entropy_quantile")},
        std_correction=1,advantage_epsilon=1e-4,kl_estimator="exp(logp_ref-logp_policy)-(logp_ref-logp_policy)-1; multiplied by exp(logp_policy-logp_old)")
    write_json(RUN/"environment.json",record)
    print(json.dumps(record,ensure_ascii=False,indent=2))


def account_balance():
    path=RUN/"control-api.json"
    rows=read(path) if path.exists() else []
    if len(rows)>=12:
        raise RuntimeError("本次余额查询额度耗尽")
    result=balance()
    rows.append(result)
    write_json(path,rows)
    if result.get("error") or not result["response"].get("is_available"):
        raise RuntimeError("余额核对失败，停止启动")
    return result


def preflight_acceptance():
    output=RUN/"preflight"
    groups=sorted([read(p) for p in output.glob("group-step*-rank0.json")],key=lambda g:g["policy_step"])
    if len(groups)!=2 or sorted(g["policy_step"] for g in groups)!=[0,2]:
        raise RuntimeError("预检必须恰好两组、八条轨迹")
    natural_update=False
    for rank in (0,1):
        if not read(output/f"reference-final-rank{rank}.json")["reference_unchanged"]:
            raise RuntimeError("预检参考参数变化")
        losses=[json.loads(s) for s in (output/f"loss-rank{rank}.jsonl").read_text(encoding="utf-8").splitlines()]
        updates=[json.loads(s) for s in (output/f"updates-rank{rank}.jsonl").read_text(encoding="utf-8").splitlines()]
        if len(losses)!=4 or len(updates)!=4 or any(x["excluded_token_gradient_l1"]!=0 for x in losses):
            raise RuntimeError("预检损失、mask或步数验证失败")
        natural_update=natural_update or any(x["parameter_delta_l1"]>0 and x["gradient_norm"]>0 for x in updates)
    replay_path=RUN/"kl-replay/verification.json"
    replay=read(replay_path) if replay_path.exists() else None
    if not natural_update and not (replay and replay["passed"] and replay["new_trajectories"]==0
            and replay["reference_unchanged"] and replay["optimizer_parameter_delta_l1"]>0):
        raise RuntimeError("预检自然同分尚无策略更新；须在原轨迹上单列受控KL回放验证，不新增采样")
    if not read(RUN/"kernel-verification.json")["passed"]:
        raise RuntimeError("安装版 TRL kernel 验证未通过")
    write_json(RUN/"preflight-acceptance.json",dict(passed=True,trajectories=8,
        reward_std=[x["reward_std"] for x in groups],initial_policy_reference_logp_max_difference=groups[0]["initial_policy_reference_logp_max_difference"],
        reference_unchanged=True,natural_policy_updated=natural_update,
        controlled_kl_policy_updated=bool(replay and replay["passed"]),
        zero_reward_kl_case="kl-replay/verification.json" if replay else "kernel-verification.json",
        formal_start="reload original SFT120; reset seed=42 and optimizer"))


def run(name):
    config=read(RUN/f"configs/{name}.json")
    directory=RUN/"commands"
    directory.mkdir(exist_ok=True)
    path=directory/f"{name}.json"
    if path.exists():
        raise RuntimeError("该命令已有尝试记录，不覆盖或自动重跑")
    if name!="preflight":
        preflight_acceptance()
    check_inputs(ROOT,config,load_public_tasks(ROOT/config["tasks"]),load_private_standards(ROOT/config["standards"]),config["run_purpose"])
    gpus=2 if name in {"preflight","train"} else 1
    devices=os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices)<gpus:
        raise RuntimeError("GPU数量不足")
    child_env=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(devices[:gpus]))
    command=[sys.executable]
    if gpus==2:
        command += ["-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=2","--module","experiments.agentic_rl.grpo"]
    else:
        command += ["-m","experiments.agentic_rl.evaluate"]
    command += ["--config",str(RUN/f"configs/{name}.json")]
    before=account_balance()
    RunBudget(ROOT,config).reserve("gpu_seconds_reserved",config["timeout_s"]*gpus,name)
    record=dict(name=name,argv=command,physical_gpus=devices[:gpus],hard_timeout_s=config["timeout_s"],
        started_utc=datetime.now(timezone.utc).isoformat(),status="running",balance_before=before,
        consumed_before=read(ROOT/config["limits_file"])["consumed"])
    write_json(path,record)
    output=ROOT/config["output_dir"]
    output.mkdir(parents=True,exist_ok=True)
    gpu_csv=directory/f"{name}-gpu.csv"
    start=time.monotonic()
    with (directory/f"{name}.log").open("w",encoding="utf-8") as log, gpu_csv.open("w",encoding="utf-8") as gpu_log:
        process=subprocess.Popen(["timeout","--signal=TERM","--kill-after=60s",str(config["timeout_s"])+"s",*command],
            cwd=SOURCE/"mymind",env=child_env,stdout=log,stderr=subprocess.STDOUT)
        record["pid"]=process.pid
        write_json(path,record)
        while process.poll() is None:
            snapshot=subprocess.run(["nvidia-smi","--query-gpu=timestamp,index,memory.used,utilization.gpu","--format=csv,noheader"],capture_output=True,text=True)
            gpu_log.write(snapshot.stdout)
            gpu_log.flush()
            time.sleep(10)
    record.update(exit_code=process.returncode,elapsed_s=time.monotonic()-start,status="exited",
        finished_utc=datetime.now(timezone.utc).isoformat(),consumed_after=read(ROOT/config["limits_file"])["consumed"],balance_after=account_balance())
    write_json(path,record)
    write_json(output/"execution.json",record)
    if process.returncode:
        raise RuntimeError(f"{name} 退出码 {process.returncode}，保留首次记录并停止")
    if name=="preflight":
        preflight_acceptance()
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=["inspect","preflight","kl-replay","sequence"])
    args=parser.parse_args()
    if args.action=="inspect":
        inspect_environment()
    elif args.action=="preflight":
        run("preflight")
    elif args.action=="kl-replay":
        # 回放仍在原预检GPU资源上限内；不增加轨迹、API或正式更新额度。
        execution=read(RUN/"commands/preflight.json")
        remaining=1800-execution["elapsed_s"]
        if remaining<600 or (RUN/"kl-replay").exists():
            raise RuntimeError("预检时间预算不足或已有回放尝试，停止")
        ledger=read(RUN/"limits.json")
        if "diagnostic_updates" not in ledger["limits"]:
            ledger["limits"]["diagnostic_updates"]=1
            ledger["events"].append(dict(kind="diagnostic_accounting",amount=1,
                reason="one controlled KL replay inside original preflight timeout/GPU allocation; zero new trajectories/API"))
            write_json(RUN/"limits.json",ledger)
        command=["timeout","--signal=TERM","--kill-after=60s","600s",sys.executable,str(SOURCE/"scripts/verify_grpo_reference_replay.py")]
        started=time.monotonic()
        record=dict(argv=command,hard_timeout_s=600,status="running",new_trajectories=0,
                    resource_allocation="remaining original preflight reservation",physical_gpus=os.environ["CUDA_VISIBLE_DEVICES"].split(",")[:1])
        path=RUN/"commands/kl-replay.json"
        if path.exists():
            raise RuntimeError("已有回放命令记录，不自动重跑")
        write_json(path,record)
        environment=dict(os.environ,CUDA_VISIBLE_DEVICES=",".join(record["physical_gpus"]))
        with (RUN/"commands/kl-replay.log").open("w",encoding="utf-8") as stream:
            result=subprocess.run(command,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        record.update(status="exited",exit_code=result.returncode,elapsed_s=time.monotonic()-started)
        write_json(path,record)
        if result.returncode:
            raise RuntimeError("KL回放验证失败；不新增轨迹")
        preflight_acceptance()
    else:
        for name in ("train","validation-20","validation-40","validation-80"):
            run(name)
        print(json.dumps(select_checkpoint(ROOT,RUN/"selection-plan.json"),ensure_ascii=False))


if __name__=="__main__":
    main()
