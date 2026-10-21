"""从已核对的候选建立本轮配置；引用既有大文件，不重做示范导出。"""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agentic_runs/stage7-preparation-20261014"
RUN = "runs/stage7-preparation-20261014"
CANDIDATE = "runs/pre-stage7-fixes-20261014"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if (OUT / "baseline.json").exists():
        raise RuntimeError("本轮配置已初始化；不能覆盖基线或重置累计额度")
    template = read(ROOT / "mymind/experiments/agentic_rl/configs/stage3.json")
    pool = read(ROOT / "agentic_runs/pre-stage7-fixes-20261014/common_pool.json")["effective_sft_rl_pool"]
    standards = read(ROOT / "agentic_runs/pre-stage7-fixes-20261014/standards/formal/private_standards.v5.json")
    # 在任何模型采样前固定两题，不按学生预检结果择题。
    ids = ["M6-B01-001", "M6-F02-011"]
    baseline = {k: deepcopy(template[k]) for k in ("model_id", "model_revision", "model_dir", "retrieval", "context_tokens", "turn_tokens")}
    baseline.update(revision="stage7-baseline-20261014-v13-v5-158", grading_revision="pre-stage7-grading-v13-20261014",
        standards=f"{CANDIDATE}/standards/formal/private_standards.v5.json",
        common_pool=f"{CANDIDATE}/common_pool.json",
        demonstrations=f"{CANDIDATE}/stage6/accepted_trajectories.json",
        student_data=f"{CANDIDATE}/stage6/student_data.json",
        public_tasks="data/agentic_rl/formal/public_tasks.json", policy_support="data/agentic_rl/formal/policy_support.json",
        knowledge="src/mymind/agents/customer_policy/resources/knowledge.md",
        public_protocol="src/mymind/agents/customer_policy/protocol.py",
        preflight_limits=f"{RUN}/preflight/limits.json", formal_limits=f"{RUN}/formal/limits.json", preflight_task_ids=ids,
        tokenizer="models/Qwen3-4B-Instruct-2507", tokenizer_template=f"{CANDIDATE}/chat_template.jinja",
        budget={"context_tokens": 16384, "completion_tokens": 8192, "turn_tokens": 2048},
        evaluation_generation={"temperature": 0, "top_p": 1, "max_new_tokens": 2048, "context_tokens": 16384,
            "model_id": template["model_id"], "model_revision": template["model_revision"]},
        review={"F2_F3": "此前独立复审源码/离线通过", "F1": "本会话有限复核通过", "new_independent_review": False},
        source_manifest="scripts/stage7_sync_files.txt", historical_freeze="stage6-frozen-20261014-v4-158")
    write(OUT / "baseline.json", baseline)
    common = {k: deepcopy(baseline[k]) for k in ("model_id", "model_revision", "model_dir", "retrieval", "context_tokens", "turn_tokens", "budget", "grading_revision", "standards")}
    common.update(tasks=baseline["public_tasks"], policy_support=baseline["policy_support"], baseline=f"{RUN}/baseline.json",
        source_manifest="scripts/stage7_sync_files.txt", seed=42)
    students = read(ROOT / "agentic_runs/pre-stage7-fixes-20261014/stage6/student_data.json")
    write(OUT / "preflight/student_data.json", [r for r in students if r["task_id"] in ids])
    for kind, target_ids, sft_steps, rl_steps, accumulation in (("preflight", ids, 2, 4, 1), ("formal", pool, 120, 80, 2)):
        base = {**deepcopy(common), "run_kind": kind, "limits_file": f"{RUN}/{kind}/limits.json", "task_ids": target_ids}
        limits = {"student_trajectories": 48 if kind == "preflight" else 1000,
            "sft_updates": sft_steps, "grpo_updates": rl_steps,
            "api_request_allowance": 576 if kind == "preflight" else 12000,
            "gpu_seconds_reserved": 8*3600 if kind == "preflight" else 32*3600}
        write(OUT / kind / "limits.json", {"limits": limits, "consumed": {}, "events": [],
            "accounting": "跨进程/重跑同一文件；失败和中断的已预留额度不返还；GPU秒按物理卡数×超时时长保守预留"})
        training = deepcopy(template["training"])
        training.update(max_steps=sft_steps, gradient_accumulation_steps=accumulation,
            save_steps=2 if kind == "preflight" else 40, learning_rate=1e-4)
        sft = {**deepcopy(base), "run_purpose": "sft", "output_dir": f"{RUN}/{kind}/sft",
            "student_data": f"{RUN}/preflight/student_data.json" if kind == "preflight" else baseline["student_data"],
            "lora": template["lora"], "training": training, "timeout_s": 1800 if kind == "preflight" else 5400}
        write(OUT / "configs" / f"{kind}-sft.json", sft)
        grpo_training = {**deepcopy(training), "max_steps": rl_steps, "gradient_accumulation_steps": 1,
            "save_steps": 4 if kind == "preflight" else 20, "learning_rate": 5e-6}
        grpo = {**deepcopy(base), "run_purpose": "grpo", "output_dir": f"{RUN}/{kind}/grpo",
            "training": grpo_training, "grpo": {**template["grpo"], "generation_batch_size": 4},
            "post_update_task_ids": ids, "stop_after_zero_signal_groups": 8,
            "timeout_s": 3600 if kind == "preflight" else 10800}
        if kind == "preflight":
            grpo["sft_adapter"] = f"{RUN}/preflight/sft/adapter"
        # 正式起点由验证选模写入；在交接前此配置不可执行。
        write(OUT / "configs" / f"{kind}-grpo.json", grpo)
        validation_ids = [r["task_id"] for r in standards if r["split"] == "validation"]
        def evaluation(name, adapter=None, searches=3):
            cfg = {**deepcopy(base), "run_purpose": "evaluate", "output_dir": f"{RUN}/{kind}/{name}",
                "task_ids": ids if kind == "preflight" else validation_ids,
                "evaluation_split": "train" if kind == "preflight" else "validation",
                "generation": baseline["evaluation_generation"], "search_limit": searches,
                "samples_per_task": 1, "timeout_s": 1200 if kind == "preflight" else 3600}
            if adapter:
                cfg["adapter_path"] = adapter
            write(OUT / "configs" / f"{kind}-{name}.json", cfg)
            return cfg
        if kind == "preflight":
            evaluation("evaluate", f"{RUN}/preflight/grpo/adapter")
            evaluation("sft-evaluate", f"{RUN}/preflight/sft/adapter")
            evaluation("one-search", f"{RUN}/preflight/sft/adapter", 1)
        else:
            evaluation("base")
            for phase, nodes in (("sft", [40, 80, 120]), ("grpo", [20, 40, 80])):
                entries = []
                for step in nodes:
                    name = f"{phase}-validation-{step}"
                    cfg = evaluation(name, f"{RUN}/formal/{phase}/checkpoint-{step}")
                    entries.append({"step": step, "evaluation": cfg["output_dir"]})
                plan = {"phase": phase, "training_output": f"{RUN}/formal/{phase}", "nodes": entries,
                    "selection_output": f"{RUN}/formal/{phase}-selection.json"}
                if phase == "sft":
                    plan["rl_config"] = f"{RUN}/configs/formal-grpo.json"
                write(OUT / "configs" / f"select-{phase}.json", plan)
            one = evaluation("one-search", searches=1)
            one["adapter_selection"] = f"{RUN}/formal/sft-selection.json"
            write(OUT / "configs/formal-one-search.json", one)
    write(OUT / "budget_plan.json", {"seed": 42, "sft_steps": 120, "sft_example_presentations": 480,
        "sft_validation_nodes": [40,80,120], "rl_steps": 80, "rl_generation_groups": 40,
        "rl_training_trajectories_without_faults": 160, "rl_post_update_trajectories": 8,
        "rl_validation_nodes": [20,40,80], "validation_tasks_each": 80,
        "validation_total_first_trajectories": 640, "planned_student_trajectories": 808,
        "total_trajectory_cap": 1000, "gpu_hours_reserved_cap": 32, "api_request_allowance": 12000,
        "prices": None, "optimality_proven": False, "full_formal_run_executed": False})


if __name__ == "__main__":
    main()
