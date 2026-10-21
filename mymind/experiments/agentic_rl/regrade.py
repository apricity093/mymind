"""按修正后的开发标准重评原始结果，保留原评分和生成源码。"""
import argparse
import shutil
from pathlib import Path

from agents.customer_policy.configuration import load_configuration
from agents.customer_policy.protocol import VisibleTrajectory
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.tasks import load_private_standards, load_policy_support
from experiments.agentic_rl.runtime import write_json
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root, config, output = load_configuration(args.config)
    standards = load_private_standards(root / config["standards"])
    clauses = load_policy_support(root / config["policy_support"])
    inputs = output / "grading_inputs"
    inputs.mkdir(exist_ok=True)
    for key in ("standards", "policy_support"):
        source = root / config[key]
        shutil.copy2(source, inputs / source.name)
    shutil.copy2(__file__, inputs / "regrade.py")
    for name in ("grading.py", "tasks.py"):
        shutil.copy2(Path(__file__).with_name(name), inputs / name)
    summary_path = output / "summary.json"
    shutil.copy2(summary_path, output / "summary-before-stage2-audit.json")
    previous = json.loads(summary_path.read_text(encoding="utf-8"))
    rows, qualified = [], []
    for path in sorted(output.glob("*-candidate-*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["grade_before_stage2_audit"] = row["grade"]
        row["grade"] = grade_result(standards[row["task_id"]], row["raw_result"],
            VisibleTrajectory.model_validate(row["trajectory"]), clauses).to_dict()
        write_json(path, row)
        rows.append({"task_id": row["task_id"], "candidate": row["candidate"], "grade": row["grade"],
                     "search_count": len(row["trajectory"]["searches"]),
                     "infrastructure_failure": row["infrastructure_failure"], "runtime_error": row["runtime_error"]})
        if row["grade"]["complete_success"] and not row["infrastructure_failure"]:
            qualified.append(row)
    accepted = {}
    for row in sorted(qualified, key=lambda r: r["candidate"]):
        accepted.setdefault(row["task_id"], row)
    write_json(summary_path, {"mode": previous["mode"], "attempts": rows,
        "accepted_task_ids": sorted(accepted) if previous["mode"] == "teacher" else [],
        "formal_evaluation": False, "regraded_after_stage2_audit": True})
    if previous["mode"] == "teacher":
        write_json(output / "accepted_trajectories.json", list(accepted.values()))
    print(f"regraded {len(rows)} attempts; complete tasks: {sorted(accepted)}")
