"""阶段8单命令的配置检查、额度预留、余额与退出记录。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "mymind"))
from experiments.agentic_rl.stage7 import check_inputs, read, RunBudget
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards
from experiments.agentic_rl.runtime import write_json


def balance():
    try:
        request = Request("https://api.deepseek.com/user/balance",
                          headers={"Authorization": "Bearer " + os.environ["DEEPSEEK_API_KEY"]})
        with urlopen(request, timeout=30) as response:
            return {"time_utc": datetime.now(timezone.utc).isoformat(), "response": json.load(response)}
    except Exception as error:
        return {"time_utc": datetime.now(timezone.utc).isoformat(), "error": type(error).__name__, "amount": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("action", choices=["before", "after"])
    parser.add_argument("--exit-code", type=int)
    parser.add_argument("--elapsed-s", type=int)
    args = parser.parse_args()
    root = Path(os.environ["MYMIND_WORK_ROOT"])
    run = root / "runs/stage8-20261016"
    cfg = read(run / "configs" / f"{args.name}.json")
    path = run / "commands" / f"{args.name}.json"
    path.parent.mkdir(exist_ok=True)
    if args.action == "before":
        if path.exists():
            raise RuntimeError("命令已尝试，不自动重启")
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(devices) != 1 or not devices[0].isdigit():
            raise ValueError("阶段8仅允许一张物理GPU")
        check_inputs(root, cfg, load_public_tasks(root / cfg["tasks"]),
                     load_private_standards(root / cfg["standards"]), "evaluate")
        if cfg.get("model_selection"):
            selected = read(root / cfg["model_selection"])
            if selected["selected_adapter"] != cfg["adapter_path"]:
                raise ValueError("模型与正式selection不同")
        maximum = (7200 if args.name in {"test-base", "test-sft", "test-rl"} else
                   3600 if args.name == "test-one_search" else
                   1800 if args.name.startswith("regression-") else 900)
        if cfg["timeout_s"] != maximum:
            raise ValueError("命令硬上限与授权不符")
        account = balance()
        if "error" in account or not account["response"].get("is_available"):
            write_json(path, {"balance_before": account, "status": "balance_check_failed_no_launch"})
            raise RuntimeError("余额检查失败，未启动模型")
        RunBudget(root, cfg).reserve("gpu_seconds_reserved", maximum, cfg["output_dir"])
        write_json(path, {"name": args.name, "configuration": str(run / "configs" / f"{args.name}.json"),
            "hard_timeout_s": maximum, "physical_gpus": devices, "gpu_seconds_reserved": maximum,
            "balance_before": account, "consumed_before": read(root / cfg["limits_file"])["consumed"],
            "status": "running", "module": "experiments.agentic_rl.evaluate"})
        print(maximum)
    else:
        record = read(path)
        record.update(exit_code=args.exit_code, elapsed_s=args.elapsed_s, balance_after=balance(),
                      consumed_after=read(root / cfg["limits_file"])["consumed"], status="exited")
        write_json(path, record)
        output = root / cfg["output_dir"]
        if output.exists():
            write_json(output / "execution.json", record)


if __name__ == "__main__":
    main()
