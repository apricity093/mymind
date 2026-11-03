"""正式 Qwen BF16 LoRA 双卡短程 SFT。"""
import json

from rl.training import (ResumeEvidence, training_inputs, load_trainable_model,
    evidence_callback, training_arguments, save_training_result)


def main():
    import torch
    from transformers import Trainer, TrainingArguments, set_seed
    root, config, output, tasks, standards, _ = training_inputs()
    if config.get("run_purpose"):
        from rl.run_control import check_inputs
        check_inputs(root, config, tasks, standards, "sft")
    set_seed(config["training"]["seed"])
    model, tokenizer = load_trainable_model(root, config, sft=True)
    rows = json.loads((root / config["student_data"]).read_text(encoding="utf-8"))
    if len(rows) != len(config["task_ids"]) or [r["task_id"] for r in rows] != config["task_ids"]:
        raise ValueError("SFT 数据与配置任务池不一致")
    dataset = [{k: row[k] for k in ("input_ids", "labels", "attention_mask")} for row in rows]

    def collate(batch):
        length = max(len(row["input_ids"]) for row in batch)
        return {key: torch.tensor([row[key] + [pad] * (length-len(row[key])) for row in batch])
                for key, pad in (("input_ids", tokenizer.pad_token_id), ("labels", -100), ("attention_mask", 0))}

    class SFTTrainer(ResumeEvidence, Trainer):
        pass

    trainer = SFTTrainer(model=model, args=TrainingArguments(**training_arguments(config, output)),
        train_dataset=dataset, data_collator=collate, callbacks=[evidence_callback(output, root, config)])
    trainer.evidence_output = output
    trainer.train(resume_from_checkpoint=str(root / config["resume_from_checkpoint"])
                  if config.get("resume_from_checkpoint") else None)
    save_training_result(trainer, tokenizer, output)


if __name__ == "__main__":
    main()
