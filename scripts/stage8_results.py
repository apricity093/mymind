"""读取阶段8首次原记录，汇总成本和行为；不生成轨迹或重新判分。"""
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

CST = timezone(timedelta(hours=8))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def usage(rows):
    total, tariffs = Counter(), Counter()
    unknown = []
    for row in rows:
        cost = row.get("cost", {})
        for key in ("embedding_requests", "rerank_http_requests"):
            total[key] += cost.get(key, 0)
        events = cost.get("usage", [])
        times = cost.get("rerank_request_started_unix_s", [])
        periods = ["peak" if (d := datetime.fromtimestamp(t, CST)).weekday() < 5 and
                   (9 <= d.hour < 12 or 14 <= d.hour < 18) else "off_peak" for t in times]
        for index, event in enumerate(events):
            for embedding in event.get("embedding") or []:
                total["embedding_usage_events"] += 1
                tokens = embedding.get("reported_tokens")
                if tokens is None:
                    unknown.append({"task_id": row["task_id"], "kind": "embedding_tokens"})
                else:
                    total["embedding_input_tokens"] += tokens
            rerank = event.get("rerank")
            if not rerank:
                continue
            total["rerank_usage_events"] += 1
            hit = rerank.get("cache_read_tokens")
            miss = rerank.get("cache_miss_tokens")
            if hit is None and miss is not None:
                hit = rerank["input_tokens"] - miss
            if miss is None and hit is not None:
                miss = rerank["input_tokens"] - hit
            total["rerank_input_tokens"] += rerank["input_tokens"]
            total["rerank_output_tokens"] += rerank["output_tokens"]
            period = periods[index] if len(periods) == len(events) else (periods[0] if len(set(periods)) == 1 else None)
            if hit is None or miss is None or period is None:
                unknown.append({"task_id": row["task_id"], "kind": "cache_or_tariff"})
                continue
            tariffs[f"{period}_hit_tokens"] += hit
            tariffs[f"{period}_miss_tokens"] += miss
            tariffs[f"{period}_output_tokens"] += rerank["output_tokens"]
        reported = sum(bool(e.get("rerank")) for e in events)
        if cost.get("rerank_http_requests", 0) != reported:
            unknown.append({"task_id": row["task_id"], "kind": "http_without_usage",
                            "requests": cost.get("rerank_http_requests"), "usage_events": reported})
    # 2026-10-16中文官方价：Flash谷时缓存命中/未命中/输出每百万0.02/1/4元，峰时双倍。
    known = sum(multiplier * (tariffs[f"{p}_hit_tokens"]*.02 + tariffs[f"{p}_miss_tokens"] +
                             tariffs[f"{p}_output_tokens"]*4)/1e6 for p, multiplier in (("off_peak", 1), ("peak", 2)))
    return {**dict(total), "tariffs": dict(tariffs), "known_deepseek_usage_price_cny": known,
            "embedding_usage_estimate_cny": total["embedding_input_tokens"]*.125/1e6,
            "unknown": unknown, "actual_attributable_bill_cny": None,
            "embedding_actual_bill_cny": None, "gpu_monetary_cost_cny": None}


def behavior(rows):
    details = []
    for row in rows:
        after = [g for g in row.get("generations", []) if any(m["role"] == "tool" for m in g["input_messages"])]
        details.append({"task_id": row["task_id"], "requested_searches": row.get("search_requests", 0),
                        "actual_searches": len(row["trajectory"]["searches"]),
                        "post_observation_search": any(g["assistant_message"].get("tool_calls") for g in after),
                        "post_observation_final": bool(after and not after[-1]["assistant_message"].get("tool_calls")),
                        "over_budget": (row.get("runtime_error") or {}).get("detail") == "模型请求超过剩余检索预算"})
    return {"requested_searches": sum(r["requested_searches"] for r in details),
            "actual_searches": sum(r["actual_searches"] for r in details),
            "actual_search_distribution": dict(Counter(str(r["actual_searches"]) for r in details)),
            "post_observation_search_tasks": [r["task_id"] for r in details if r["post_observation_search"]],
            "post_observation_final_tasks": [r["task_id"] for r in details if r["post_observation_final"]],
            "over_budget_tasks": [r["task_id"] for r in details if r["over_budget"]], "details": details}


def add_breakdown_counts(metrics, tasks, standards):
    for dimension, items in metrics["breakdown"].items():
        for item, part in items.items():
            selected = []
            for row in metrics["rows"]:
                tid = row["task_id"]
                standard = standards[tid]
                group = standard["group_id"]
                topic = "refund" if group.startswith(("R", "D")) else {"B": "duplicate_payment", "T": "authentication", "P": "authorization", "F": "fault"}.get(group[0], "basic")
                actual = {"domain": tasks[tid]["role"], "topic": topic, "group": group,
                          "primary_disposition": "/".join(standard["dispositions"])}[dimension]
                if actual == item:
                    selected.append(row)
            assert len(selected) == part["tasks"]
            part.update(complete_success=sum(r["success"] for r in selected),
                        severe_errors=sum(r["severe"] for r in selected),
                        protocol_failures=sum(r["protocol_failure"] for r in selected))
    return metrics["breakdown"]


def describe(directory):
    metrics = read(directory / "metrics.json")
    rows = records(directory / "first_records.jsonl")
    config = read(directory / "configuration.json")
    tasks = {r["task_id"]: r for r in read(directory / "inputs" / Path(config["tasks"]).name)}
    standards = {r["task_id"]: r for r in read(directory / "inputs" / Path(config["standards"]).name)}
    add_breakdown_counts(metrics, tasks, standards)
    successes = {r["task_id"] for r in metrics["rows"] if r["success"]}
    result = {k: v for k, v in metrics.items() if k not in {"rows", "all_task_cost", "successful_task_cost"}}
    result.update(complete_success=len(successes), severe_errors=sum(r["severe"] for r in metrics["rows"]),
                  mean_reward=sum(r["grade"]["reward"] for r in rows)/len(rows) if rows else None,
                  retrieval=behavior(rows))
    for key, subset in (("all_task_cost", rows), ("successful_task_cost", [r for r in rows if r["task_id"] in successes])):
        old = metrics[key]
        result[key] = {k: v for k, v in old.items() if k not in {"usage", "missing_cost_information"}}
        result[key].update(usage=usage(subset), missing_cost_information=["精确归属账单、Embedding实际账单和GPU货币费用未知；失败无usage费用不记0"])
    result["all_cost_per_success"] = ({"known_deepseek_usage_price_cny": result["all_task_cost"]["usage"]["known_deepseek_usage_price_cny"]/len(successes),
        "generation_s": metrics["all_task_cost"]["gpu_generation_s"]/len(successes),
        "requested_searches": metrics["all_task_cost"]["search_requests"]/len(successes)} if successes else None)
    return result


def check_command(root, name):
    run = root / "runs/stage8-20261016"
    command = read(run / "commands" / f"{name}.json")
    config = read(run / "configs" / f"{name}.json")
    directory = root / config["output_dir"]
    result = describe(directory)
    added = {k: command["consumed_after"].get(k, 0)-command["consumed_before"].get(k, 0) for k in command["consumed_after"]}
    summary = {"command": command, "result": result, "consumption_after_gpu_reservation": added}
    (run / "commands" / f"{name}-checked.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    assert command["exit_code"] == 0, f"命令退出{command['exit_code']}，暂停"
    assert result["all_tasks_finished"] and not result["unresolved_infrastructure_failures"], "首次表不完整或基础设施异常，暂停"
    assert added["student_trajectories"] == len(config["task_ids"])
    assert added["api_request_allowance"] == 12*len(config["task_ids"])
    assert added["sft_updates"] == added["grpo_updates"] == 0
    assert command["balance_after"].get("response", {}).get("is_available"), "命令后余额检查失败"
    info = read(directory / "model_load.json")
    if config.get("adapter_path"):
        assert info["adapter_path"] == str(root / config["adapter_path"])
        assert info["adapter_weights"]["saved_tensors"] == info["adapter_weights"]["matched_tensors"] == 504
    assert not info["training"] and info["trainable_parameters"] == 0
    return {"name": name, "exit_code": command["exit_code"], "finished": result["finished_tasks"],
            "success": result["complete_success"], "severe": result["severe_errors"],
            "protocol": result["protocol_failures"], "usage": result["all_task_cost"]["usage"],
            "balance_after": command["balance_after"]}


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-command", required=True)
    args = parser.parse_args()
    print(json.dumps(check_command(Path(os.environ["MYMIND_WORK_ROOT"]), args.check_command), ensure_ascii=False, indent=2))
