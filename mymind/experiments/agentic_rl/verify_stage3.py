"""汇总正式双卡闭环的实际更新、mask、恢复和故障证据。"""
import argparse
import json
import math
import os
from pathlib import Path

from experiments.agentic_rl.runtime import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def verify(run, tokenizer):
    stages = {}
    for name, expected_step in (("sft", 2), ("sft-resume", 4), ("grpo", 6), ("grpo-resume", 8)):
        directory = run / name
        result = read(directory / "training_result.json")
        assert result["global_step"] == expected_step and result["world_size"] == 2
        assert result["trainable_dtypes"] == ["torch.bfloat16"]
        ranks = {}
        for rank in range(2):
            updates = jsonl(directory / f"updates-rank{rank}.jsonl")
            assert updates and all(row["gradient_finite"] and math.isfinite(row["gradient_norm"]) for row in updates)
            assert any(row["gradient_norm"] > 0 and row["parameter_delta_l1"] > 0 for row in updates)
            ranks[str(rank)] = {"update_steps": [r["step"] for r in updates],
                "nonzero_gradient_steps": sum(r["gradient_norm"] > 0 for r in updates),
                "parameter_delta_l1_sum": sum(r["parameter_delta_l1"] for r in updates),
                "peak_allocated_gib": max(r["peak_allocated_bytes"] for r in updates)/2**30,
                "peak_reserved_gib": max(r["peak_reserved_bytes"] for r in updates)/2**30,
                "elapsed_steps_s": sum(r["elapsed_step_s"] for r in updates)}
            if name.endswith("resume"):
                resume = read(directory / f"resume-rank{rank}.json")
                saved, loaded = resume["saved_optimizer"], resume["loaded_optimizer"]
                assert saved["state_entries"] == loaded["state_entries"] and saved["steps"] == loaded["steps"]
                assert all(math.isclose(saved[key], loaded[key], rel_tol=1e-6, abs_tol=1e-9)
                           for key in ("exp_avg_l1", "exp_avg_sq_l1")) and resume["rng_file_exists"]
                assert updates[0]["step"] == int(saved["steps"][0]) + 1
                ranks[str(rank)]["resume"] = resume
        checkpoint = directory / f"checkpoint-{expected_step}"
        assert all((checkpoint / file).exists() for file in ("optimizer.pt", "scheduler.pt", "trainer_state.json",
            "adapter_model.safetensors", "rng_state_0.pth", "rng_state_1.pth"))
        stages[name] = {"ranks": ranks, "trainable_parameters": result["trainable_parameters"]}

    audit = read(run / "student-data" / "mask_audit.json")
    for row in read(run / "student-data" / "student_data.json"):
        expected = set()
        for span in row["spans"]:
            expected.update(range(span["start"], span["end"]))
        assert {i for i, label in enumerate(row["labels"]) if label != -100} == expected
        assert all(row["labels"][i] == row["input_ids"][i] for i in expected)

    groups, mask_observations = [], 0
    for stage in ("grpo", "grpo-resume"):
        directory = run / stage
        assert read(directory / "post_update_sampling.json")["model_source"] == "current_in_memory_updated_policy"
        for path in sorted(directory.glob("group-*-rank0.json")):
            group = read(path)
            partner = read(Path(str(path).replace("rank0", "rank1")))
            assert len(group["rows"]) == 4 and len(set(group["task_ids"])) == 1
            assert not group["infrastructure_failure"]
            mean = sum(group["rewards"])/4
            assert all(abs(a-(r-mean)) < 1e-5 for a, r in zip(group["advantages"], group["rewards"]))
            token_sum = 0
            for rank, record in enumerate((group, partner)):
                local_count = len(record["completion_ids"])
                for index, (ids, mask, tool_mask) in enumerate(zip(record["completion_ids"],
                                                record["completion_mask"], record["tool_mask"])):
                    token_sum += sum(m*t for m, t in zip(mask, tool_mask))
                    external = tokenizer.decode([i for i, m, t in zip(ids, mask, tool_mask) if m and not t])
                    row = record["rows"][rank*local_count+index]
                    for message in row["completion"]:
                        if message.get("role") == "tool":
                            assert message["content"] in external
                            mask_observations += 1
            assert token_sum == group["effective_model_tokens"]
            groups.append({"stage": stage, "policy_step": group["policy_step"], "task_id": group["task_ids"][0],
                "rewards": group["rewards"], "advantages": group["advantages"],
                "effective_model_tokens": token_sum, "nonzero_advantage": group["nonzero_advantage"],
                "actual_searches": sum(r["actual_search_count"] for r in group["rows"]),
                "prompt_tokens": group["prompt_lengths"]+partner["prompt_lengths"],
                "completion_tokens": group["completion_lengths"]+partner["completion_lengths"],
                "generation_s_by_rank": [sum(t["generation_s"] for t in record["generation"]) for record in (group, partner)]})
    assert mask_observations > 0 and any(group["nonzero_advantage"] for group in groups if group["policy_step"] < 6)
    for name in ("fault-recover", "fault-repeat"):
        directory = run / name
        for rank in range(2):
            first = read(directory / f"group-step0-S10-attempt0-rank{rank}.json")
            second = read(directory / f"group-step0-S10-attempt1-rank{rank}.json")
            assert first["optimization_contribution"] == "excluded_before_backward"
            assert first["optimization_model_tokens"] == 0 and first["controlled_fault"]
            assert first["policy_step"] == second["policy_step"] == 0
            if name == "fault-repeat":
                assert second["optimization_contribution"] == "excluded_before_backward"
                assert read(directory / f"paused-rank{rank}.json")["policy_step"] == 0
                assert not (directory / f"updates-rank{rank}.jsonl").exists()
            else:
                assert second["optimization_contribution"] == "accepted"
                assert jsonl(directory / f"updates-rank{rank}.jsonl")[0]["step"] == 1
    return {"passed": True, "stages": stages, "sft_mask_audit": audit, "rl_groups": groups,
        "masked_tool_observations_checked": mask_observations,
        "fault_handling": "整组在 backward 前排除，同版本补采一次，再次异常在 step 0 暂停",
        "context_tokens": 16384, "completion_tokens_including_tools": 8192,
        "formal_evaluation": False, "stage4_started": False}


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(os.environ["MYMIND_WORK_ROOT"])
    tokenizer = AutoTokenizer.from_pretrained(root / "models/Qwen3-4B-Instruct-2507", local_files_only=True)
    summary = verify(root / args.run, tokenizer)
    write_json(root / args.run / "stage3_acceptance.json", summary)
    print("stage 3 acceptance passed", flush=True)


if __name__ == "__main__":
    main()
