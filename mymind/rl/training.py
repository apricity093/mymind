"""训练入口共用的装载、更新证据及恢复记录。"""
import json
import math
import os
from pathlib import Path
import time

from agents.customer_policy.configuration import load_configuration
from rl.runtime import initialize, write_json, apply_resume
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support


def training_inputs():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    root, config, output = load_configuration(args.config)
    if int(os.environ.get("RANK", "0")) == 0:
        return initialize()
    config = apply_resume(root, config, output, args.resume_from_checkpoint)
    return (root, config, output, load_public_tasks(root / config["tasks"]),
            load_private_standards(root / config["standards"]), load_policy_support(root / config["policy_support"]))


def load_trainable_model(root, config, *, sft=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    provenance = json.loads((root / config["model_dir"] / "base_revision.json").read_text(encoding="utf-8"))
    if provenance["revision"] != config["model_revision"] or provenance["repo_id"] != config["model_id"]:
        raise ValueError("训练基座与固定 revision 不一致")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    tokenizer = AutoTokenizer.from_pretrained(root / config["model_dir"], local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(root / config["model_dir"], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    if sft:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **config["lora"]),
                               autocast_adapter_dtype=False)
    else:
        model = PeftModel.from_pretrained(model, root / config["sft_adapter"], is_trainable=True,
                                          autocast_adapter_dtype=False)
    if config.get("run_kind"):
        from peft import get_peft_model_state_dict
        loaded = get_peft_model_state_dict(model)
        matched = None
        if not sft:
            from safetensors.torch import load_file
            saved = load_file(str(root / config["sft_adapter"] / "adapter_model.safetensors"))
            matched = sum(torch.equal(loaded[n].detach().cpu(), value) for n,value in saved.items() if n in loaded)
            if set(saved) != set(loaded) or matched != len(saved):
                raise ValueError("RL 实际初始LoRA张量与指定SFT adapter 不一致")
        write_json(root / config["output_dir"] / f"model-load-rank{os.environ['RANK']}.json", {
            "base": provenance, "tokenizer_path": str(root / config["model_dir"]),
            "origin": "new_lora_from_original_qwen" if sft else config["sft_adapter"],
            "matched_sft_tensors": matched, "adapter_tensors": len(loaded),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")})
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model, tokenizer


def append_record(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def optimizer_summary(optimizer):
    states = list(optimizer.state.values())
    return {"state_entries": len(states), "steps": sorted(set(float(s["step"]) for s in states if "step" in s)),
            "exp_avg_l1": sum(float(s["exp_avg"].float().abs().sum()) for s in states if "exp_avg" in s),
            "exp_avg_sq_l1": sum(float(s["exp_avg_sq"].float().abs().sum()) for s in states if "exp_avg_sq" in s)}


class ResumeEvidence:
    def _load_from_checkpoint(self, checkpoint, model=None):
        super()._load_from_checkpoint(checkpoint, model)
        # PEFT load_adapter 默认将适配参数提升为 FP32，恢复后保持本入口的 BF16 参数精度。
        if self.args.bf16:
            import torch
            for parameter in (model or self.model).parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.bfloat16)

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint:
            import torch
            saved = torch.load(Path(checkpoint) / "optimizer.pt", map_location="cpu", weights_only=True)
            expected = optimizer_summary(type("Saved", (), {"state": saved["state"]})())
            actual = optimizer_summary(self.optimizer)
            write_json(self.evidence_output / f"resume-rank{self.accelerator.process_index}.json",
                {"checkpoint": str(checkpoint), "saved_optimizer": expected, "loaded_optimizer": actual,
                 "trainer_global_step": self.state.global_step, "scheduler": self.lr_scheduler.state_dict(),
                 "rng_file_exists": (Path(checkpoint) / f"rng_state_{self.accelerator.process_index}.pth").exists()})
            if expected["state_entries"] != actual["state_entries"] or expected["steps"] != actual["steps"]:
                raise RuntimeError("优化器状态未正确恢复")
            if any(not math.isclose(expected[key], actual[key], rel_tol=1e-6, abs_tol=1e-9)
                   for key in ("exp_avg_l1", "exp_avg_sq_l1")):
                raise RuntimeError("优化器动量未正确恢复")


def evidence_callback(output, root=None, config=None, on_update=None):
    import torch
    from transformers import TrainerCallback

    class EvidenceCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if config and config.get("run_kind"):
                from rl.run_control import reserve_distributed
                reserve_distributed(root, config, config["run_purpose"] + "_updates", 1, f"optimizer-step-{state.global_step+1}")

        def on_train_begin(self, args, state, control, model, **kwargs):
            self.path = output / f"updates-rank{args.process_index}.jsonl"
            self.parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
            self.previous = {n: p.detach().cpu().clone() for n, p in self.parameters.items()}
            self.start = time.monotonic()
            torch.cuda.reset_peak_memory_stats()

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            gradients = [p.grad for p in self.parameters.values() if p.grad is not None]
            self.gradient_norm = sum(float(g.float().square().sum()) for g in gradients) ** 0.5
            self.gradient_finite = all(bool(torch.isfinite(g).all()) for g in gradients)
            if not self.gradient_finite:
                raise RuntimeError("训练梯度非有限")

        def on_step_end(self, args, state, control, optimizer, **kwargs):
            delta = sum(float((p.detach().cpu().float() - self.previous[n].float()).abs().sum())
                        for n, p in self.parameters.items())
            self.previous = {n: p.detach().cpu().clone() for n, p in self.parameters.items()}
            torch.cuda.synchronize()
            now = time.monotonic()
            record = {"step": state.global_step, "gradient_norm": self.gradient_norm,
                "gradient_finite": self.gradient_finite, "parameter_delta_l1": delta,
                "elapsed_step_s": now - self.start, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "optimizer": optimizer_summary(optimizer)}
            append_record(self.path, record)
            self.start = now
            if on_update:
                on_update(record)
            return control

    return EvidenceCallback()


def training_arguments(config, output):
    return dict(output_dir=str(output), **config["training"], bf16=True, optim="adamw_torch",
        torch_compile=False, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, report_to="none", save_strategy="steps", logging_steps=1,
        remove_unused_columns=False, save_only_model=False, dataloader_num_workers=0)


def save_training_result(trainer, tokenizer, output):
    trainer.save_model(str(output / "adapter"))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "adapter")
        write_json(output / "training_result.json", {"global_step": trainer.state.global_step,
            "world_size": trainer.accelerator.num_processes, "effective_args": trainer.args.to_dict(),
            "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
            "trainable_dtypes": sorted(set(str(p.dtype) for p in trainer.model.parameters() if p.requires_grad)),
            "optimizer": optimizer_summary(trainer.optimizer), "log_history": trainer.state.log_history})
