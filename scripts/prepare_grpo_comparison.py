"""建立一次独立 GRPO 式配置对照；不修改冻结基线与旧账本。"""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "agentic_runs/grpo-standard-20261021"
REMOTE = "runs/grpo-standard-20261021"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    if (RUN / "authorization.json").exists():
        raise RuntimeError("本次实验已建立；不重置额度或覆盖配置")
    original = read(ROOT / "agentic_runs/stage7-preparation-20261014/formal/grpo/configuration.json")
    baseline = read(ROOT / "agentic_runs/stage7-preparation-20261014/baseline.json")
    config = deepcopy(original)
    config.update(configuration_comparison=f"{REMOTE}/authorization.json", limits_file=f"{REMOTE}/limits.json",
                  source_manifest="scripts/grpo_comparison_sync_files.txt", output_dir=f"{REMOTE}/train", timeout_s=14400)
    config["grpo"].update(loss_type="grpo", scale_rewards="group", epsilon=.2, epsilon_high=.2,
                          beta=.04, importance_sampling_level="token", sync_ref_model=False, entropy_coef=0.,
                          use_bias_correction_kl=True)
    preflight = deepcopy(config)
    preflight.update(run_kind="preflight", task_ids=baseline["preflight_task_ids"],
                     output_dir=f"{REMOTE}/preflight", timeout_s=1800, post_update_task_ids=[])
    preflight["training"].update(max_steps=4, save_steps=4)
    configs = {"preflight": preflight, "train": config}
    for step in (20, 40, 80):
        evaluation = read(ROOT / f"agentic_runs/stage7-preparation-20261014/configs/formal-grpo-validation-{step}.json")
        evaluation.update(configuration_comparison=f"{REMOTE}/authorization.json", limits_file=f"{REMOTE}/limits.json",
                          source_manifest="scripts/grpo_comparison_sync_files.txt", timeout_s=7200,
                          output_dir=f"{REMOTE}/validation-{step}", adapter_path=f"{REMOTE}/train/checkpoint-{step}")
        configs[f"validation-{step}"] = evaluation
    authorization = dict(experiment="GRPO式配置对照", date="2026-10-21", seed=42,
        baseline=original["baseline"], sft_selection=original["sft_selection"], sft_adapter=original["sft_adapter"],
        limits_file=f"{REMOTE}/limits.json", preflight_task_ids=baseline["preflight_task_ids"],
        outputs=[c["output_dir"] for c in configs.values()],
        normal_trajectories=dict(preflight=8, training=160, validation=240, post_update=8),
        infrastructure_trajectories=16, api_request_allowance=5184,
        control_api_requests=12, gpu_seconds_reserved=54000,
        hard_timeouts_s={name: c["timeout_s"] for name,c in configs.items()},
        source="2026-10-21用户授权：最多两组预检、B臂80步、20/40/80各80验证、固定8后采样；基础设施整组最多补采一次，另列16条封顶。",
        model_selection_rule="组合等权完整成功率最高；完全同分严重错误率更低；再同分步数更早",
        forbidden=["sft_training", "rl40_continuation", "test120", "extra_arm", "hyperparameter_search", "commit", "push", "replace_formal_pointer"])
    write(RUN / "authorization.json", authorization)
    write(RUN / "limits.json", dict(limits=dict(student_trajectories=416, infrastructure_trajectories=16,
        api_request_allowance=5184, grpo_updates=84, diagnostic_updates=1, gpu_seconds_reserved=54000), consumed={}, events=[]))
    for name,c in configs.items():
        write(RUN / "configs" / f"{name}.json", c)
    write(RUN / "selection-plan.json", dict(phase="grpo", training_output=f"{REMOTE}/train",
        selection_output=f"{REMOTE}/selection.json", nodes=[dict(step=s,evaluation=f"{REMOTE}/validation-{s}") for s in (20,40,80)]))
    # 原正式 manifest 的全部实际依赖继续保留，新增本次入口及测试。
    manifest = (ROOT / "scripts/stage7_sync_files.txt").read_text(encoding="utf-8").splitlines()
    manifest += ["mymind/experiments/agentic_rl/grpo_diagnostics.py", "mymind/tests/test_grpo_configuration_comparison.py",
                 "scripts/prepare_grpo_comparison.py", "scripts/run_grpo_comparison.py", "scripts/verify_grpo_kernels.py",
                 "scripts/summarize_grpo_comparison.py", "scripts/grpo_comparison_sync_files.txt", "scripts/stage8_command.py",
                 "scripts/verify_grpo_reference_replay.py"]
    (ROOT / "scripts/grpo_comparison_sync_files.txt").write_text("\n".join(dict.fromkeys(manifest))+"\n", encoding="utf-8")
    print(json.dumps(authorization, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
