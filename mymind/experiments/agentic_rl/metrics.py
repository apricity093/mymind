"""首次策略评测的完整分母、组合等权指标及实际可得成本。"""
from collections import Counter, defaultdict


def rates(rows):
    n = len(rows)
    return {"tasks": n, "success_rate": sum(r["success"] for r in rows)/n,
            "severe_error_rate": sum(r["severe"] for r in rows)/n,
            "infrastructure_failures": sum(r["infrastructure_failure"] for r in rows)}


def summarize(task_ids, records, tasks, standards):
    by_id = {r["task_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id)-set(task_ids):
        raise ValueError("首次主表存在重复或非预定题目")
    groups = {key: defaultdict(list) for key in ("domain", "topic", "group", "primary_disposition")}
    rows = []
    for tid in task_ids:
        row = by_id.get(tid, {})
        grade = row.get("grade", {})
        failure = row.get("infrastructure_failure", False)
        result = {"task_id": tid, "finished": tid in by_id,
            "success": bool(grade.get("complete_success")) and not failure and not row.get("runtime_error"),
            "severe": bool(grade.get("severe_error")), "reward": grade.get("reward"),
            "protocol_failure": bool(grade.get("protocol_error")) or (row.get("runtime_error") or {}).get("kind") in {"protocol_error", "budget_exhausted"},
            "infrastructure_failure": failure, "elapsed_s": row.get("elapsed_s"),
            "search_requests": row.get("search_requests", len(row.get("trajectory", {}).get("searches", []))),
            "missing_conditions": grade.get("missing", []), "errors": grade.get("errors", []),
            "cost": row.get("cost", {})}
        rows.append(result)
        group = standards[tid].group_id
        topic = "refund" if group.startswith(("R", "D")) else {"B": "duplicate_payment", "T": "authentication", "P": "authorization", "F": "fault"}.get(group[0], "basic")
        for key, value in (("domain", tasks[tid].role), ("topic", topic), ("group", group),
                           ("primary_disposition", "/".join(standards[tid].dispositions))):
            groups[key][value].append(result)
    breakdown = {key: {name: rates(values) for name,values in mapping.items()} for key,mapping in groups.items()}
    group_rates = list(breakdown["group"].values())
    return {"planned_tasks": len(task_ids), "finished_tasks": len(by_id),
        "all_tasks_finished": len(by_id) == len(task_ids),
        "combination_equal_success_rate": sum(r["success_rate"] for r in group_rates)/len(group_rates),
        "combination_equal_severe_error_rate": sum(r["severe_error_rate"] for r in group_rates)/len(group_rates),
        "instance_success_rate": rates(rows)["success_rate"],
        "infrastructure_failures": sum(r["infrastructure_failure"] for r in rows),
        "protocol_failures": sum(r["protocol_failure"] for r in rows),
        "unfinished_tasks": [r["task_id"] for r in rows if not r["finished"]],
        "reward_distribution": dict(Counter(str(r["reward"]) for r in rows)),
        "breakdown": breakdown, "rows": rows,
        "all_task_cost": cost_summary(rows), "successful_task_cost": cost_summary([r for r in rows if r["success"]]),
        "mean_reward_is_success_rate": False}


def cost_summary(rows):
    return {"tasks": len(rows), "search_requests": sum(r["search_requests"] for r in rows),
        "elapsed_s": sum(r["elapsed_s"] or 0 for r in rows),
        "elapsed_missing": sum(r["elapsed_s"] is None for r in rows),
        "embedding_requests": sum(r["cost"].get("embedding_requests", 0) for r in rows),
        "rerank_http_requests": sum(r["cost"].get("rerank_http_requests", 0) for r in rows),
        "usage": [r["cost"].get("usage", []) for r in rows],
        "missing_cost_information": ["货币价格/账单未查询", "请求失败或供应商未返回的usage不可得", "未启动题目成本未知"],
        "gpu_generation_s": sum(r["cost"].get("gpu_generation_s", 0) for r in rows)}
