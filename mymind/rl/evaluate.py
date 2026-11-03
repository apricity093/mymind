"""在开发任务上运行真实策略，保存原始消息、工具诊断和业务判分。"""
import asyncio
import json
import os
from pathlib import Path
import time

from agents.customer_policy.configuration import rerank_gateway
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import DeepSeekGeneration, QwenGeneration
from agents.customer_policy.protocol import VisibleTrajectory
from agents.customer_policy.retrieval import build_retrieval
from agents.customer_policy.runner import run_task
from rl.grading import grade_result
from rl.runtime import initialize, write_json


def process_errors(row):
    """相同有效结果上的原样重复搜索不作为完整教师示范。"""
    seen, errors = {}, []
    for turn in row["trajectory"]["searches"]:
        query = "".join(turn["action"].get("query", "").split())
        observation = turn["observation"]
        content = [(s["source_id"], s["content"]) for s in observation["snippets"]]
        if query in seen and content and seen[query] == content:
            errors.append("相同查询重复返回相同正文，无新增信息")
        if observation["status"]["state"] == "candidates":
            seen[query] = content
    return errors


async def main(*, teacher=False):
    root, config, output, tasks, standards, clauses = initialize()
    if config.get("run_purpose") == "evaluate":
        if teacher:
            raise ValueError("策略评测不能调用教师")
        from rl.formal_evaluate import evaluate
        return await evaluate(root, config, output, tasks, standards, clauses)
    gateway = rerank_gateway(config["retrieval"])
    retrieval = build_retrieval(root, config["retrieval"], gateway)
    if retrieval.knowledge.stats()["pending_main"] or not retrieval.knowledge._main.count():
        raise RuntimeError("须先 prepare 完整 Qwen 实验索引")
    generation = (DeepSeekGeneration(os.environ["DEEPSEEK_API_KEY"], config["generation"]) if teacher
                  else QwenGeneration(root / config["model_dir"], config["generation"],
                      root / config["adapter_path"] if config.get("adapter_path") else None))
    if not teacher:
        generation.torch.manual_seed(config.get("seed", 42))
        write_json(output / "model_load.json", generation.load_info)
    if not teacher:
        ordinary = await generation.generate([{"role": "user", "content": "请用一句中文说明你可以回答问题。"}], False)
        write_json(output / "ordinary_generation.json", {"text": ordinary.text, **ordinary.diagnostics})
        (output / "chat_template.jinja").write_text(generation.tokenizer.chat_template, encoding="utf-8")
    rows, accepted = [], []
    formal = config.get("dataset_mode") == "formal"
    paused = False
    async def collect(task_id):
        nonlocal paused
        if paused:
            return
        if standards[task_id].split != ("train" if config.get("dataset_mode") in {"calibration", "formal"} else "dev"):
            raise ValueError("任务不属于当前采集允许的划分")
        count = config["candidates_per_task"] if teacher else config.get("samples_per_task", 1)
        succeeded = False
        for candidate in range(count):
            path = output / f"{task_id}-candidate-{candidate+1}.json"
            partial = output / f"{task_id}-candidate-{candidate+1}.partial.json"
            previous = root / config["resume_from"] if config.get("resume_from") else output
            source = path if path.exists() else previous / path.name
            partial_source = partial if partial.exists() else previous / partial.name
            if succeeded and not source.exists() and not partial_source.exists():
                break
            executed_now = False
            start = time.monotonic()
            if formal and source.exists():
                row = json.loads(source.read_text(encoding="utf-8"))
                if source != path:
                    row["source_record"] = str(source.relative_to(root))
                    row["original_grade"] = row["grade"]
            elif formal and partial_source.exists():
                row = json.loads(partial_source.read_text(encoding="utf-8"))
                row.update(runtime_error={"kind": "interrupted", "detail": "中断候选按已耗候选保留，不重复调用"},
                           infrastructure_failure=True, elapsed_s=None)
                if partial_source != partial:
                    row["source_record"] = str(partial_source.relative_to(root))
            else:
                executed_now = True
                row = await run_task(tasks[task_id], generation, PolicyEnvironment(retrieval),
                    budget=config.get("budget") if not teacher else None,
                    on_progress=(lambda state: write_json(partial, state)) if formal else None)
                row["elapsed_s"] = time.monotonic() - start
            row["candidate"] = candidate + 1
            grade = grade_result(standards[task_id], row["raw_result"],
                                 VisibleTrajectory.model_validate(row["trajectory"]), clauses)
            row["grade"] = grade.to_dict()
            row["process_errors"] = process_errors(row) if formal else []
            if not formal or not path.exists():
                write_json(path, row)
            rows.append({"task_id": task_id, "candidate": candidate+1, "grade": row["grade"],
                         "search_count": len(row["trajectory"]["searches"]),
                         "infrastructure_failure": row["infrastructure_failure"], "runtime_error": row["runtime_error"],
                         "process_errors": row["process_errors"]})
            print(f"{task_id} candidate={candidate+1} reward={grade.reward} elapsed={row['elapsed_s']}s", flush=True)
            if executed_now and config.get("dataset_mode") in {"calibration", "formal"} and (row["runtime_error"] or {}).get("status_code") == 402:
                paused = True
                break
            if teacher and not succeeded and grade.complete_success and not row["infrastructure_failure"] and not row["runtime_error"] and not row["process_errors"]:
                accepted.append(row)
                succeeded = True
                if not formal:
                    break
    semaphore = asyncio.Semaphore(config.get("concurrency", 1) if teacher else 1)
    async def limited(task_id):
        async with semaphore:
            await collect(task_id)
    await asyncio.gather(*(limited(task_id) for task_id in config["task_ids"]))
    mode = "teacher" if teacher else ("student-sampling" if config.get("dataset_mode") == "calibration" else "base-dev")
    write_json(output / "summary.json", {"mode": mode, "attempts": rows,
        "accepted_task_ids": [row["task_id"] for row in accepted], "formal_evaluation": False,
        "paused_for_service_balance": paused,
        "unstarted_task_ids": [t for t in config["task_ids"] if not any(r["task_id"] == t for r in rows)]})
    if teacher:
        write_json(output / "accepted_trajectories.json", accepted)
        await generation.close()
    retrieval.knowledge.close()
    retrieval.knowledge.embedding.close()
    await gateway.client.close()
    label = config.get("dataset_mode", "development")
    print(f"completed {len(rows)} {label} attempts; accepted {len(accepted)} teacher trajectories")


if __name__ == "__main__":
    asyncio.run(main())
