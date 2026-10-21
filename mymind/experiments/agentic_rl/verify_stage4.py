"""独立 HTTP 验收及事后判分；不由应用导入。"""
import argparse
import json
from pathlib import Path
import time
from dataclasses import asdict

import requests

from agents.customer_policy.protocol import VisibleTrajectory
from agents.customer_policy.rendering import render_result
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support
from experiments.agentic_rl.grading import grade_result


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resources", required=True)
    parser.add_argument("--saved-responses", action="store_true", help="只核查已有响应，不再次采样")
    args = parser.parse_args()
    output, resources = Path(args.output), Path(args.resources)
    output.mkdir(parents=True, exist_ok=True)
    if args.saved_responses:
        status = json.loads((output / "status.json").read_text(encoding="utf-8"))
        original = json.loads((output / "http_acceptance.json").read_text(encoding="utf-8"))
    else:
        status_response = requests.get(args.url + "/status", timeout=30)
        status_response.raise_for_status()
        status = status_response.json()
        write(output / "status.json", status)
    tasks = load_public_tasks(resources / "dev_tasks.json")
    standards = load_private_standards(resources / "dev_standards.json")
    support = load_policy_support(resources / "policy_support.json")
    results = []
    for task_id in ("S10", "S13"):
        task = tasks[task_id]
        request = {key: value for key, value in task.model_dump().items() if key != "task_id"}
        start = time.monotonic()
        if args.saved_responses:
            row = json.loads((output / f"{task_id}-response.json").read_text(encoding="utf-8"))
            previous = next(r for r in original["results"] if r["task_id"] == task_id)
        else:
            write(output / f"{task_id}-request.json", request)
            response = requests.post(args.url + "/requests", json=request, timeout=300)
            response.raise_for_status()
            row = response.json()
            write(output / f"{task_id}-response.json", row)
        searches = row["trajectory"]["searches"]
        generations = row["generations"]
        observed_then_generated = bool(searches) and any(
            any(m["role"] == "tool" and m["content"] == json.dumps(turn["observation"],
                ensure_ascii=False, separators=(",", ":")) for m in g["input_messages"])
            for turn in searches for g in generations[1:])
        visible_contents_in_prompt = all(any(json.dumps(turn["observation"], ensure_ascii=False,
            separators=(",", ":")) in g.get("prompt", "") for g in generations[1:]) for turn in searches)
        actual_queries = [turn["action"].get("query") for turn in searches]
        generated_queries = []
        for generation in generations:
            for call in generation["assistant_message"].get("tool_calls", []):
                arguments = call["function"]["arguments"]
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                generated_queries.append(arguments.get("query"))
        template_matches = isinstance(row["raw_result"], dict) and row["rendered"] == render_result(row["raw_result"])
        grading_trajectory = {**row["trajectory"], "task_id": task_id}
        score = grade_result(standards[task_id], row["raw_result"], VisibleTrajectory.model_validate(grading_trajectory), support)
        if args.saved_responses:
            http_200, readback = previous["checks"]["http_200"], previous["checks"]["request_readback"]
            latency = previous["latency_s"]
        else:
            retrieved = requests.get(args.url + "/requests/" + row["request_id"], timeout=30)
            retrieved.raise_for_status()
            http_200, readback = response.status_code == 200, retrieved.json() == row
            latency = time.monotonic() - start
        checks = {"http_200": http_200, "request_readback": readback,
            "runtime_ok": row["runtime_error"] is None, "real_search": bool(searches),
            "observed_then_generated": observed_then_generated, "visible_contents_in_prompt": visible_contents_in_prompt,
            "queries_from_policy": actual_queries == generated_queries,
            "fresh_context": not any(m["role"] == "tool" for m in generations[0]["input_messages"]),
            "template_matches_model": template_matches, "fixed_rerank_success": all(
                d.get("diagnostics", {}).get("rerank_status") == "llm_success" for d in row["tool_diagnostics"])}
        results.append({"task_id": task_id, "request_id": row["request_id"], "checks": checks,
            "latency_s": latency, "queries": actual_queries,
            "grade": asdict(score), "runtime_error": row["runtime_error"]})
    model = status["model"]
    weights = model["adapter_weights"]
    checks = {"adapter_loaded": model["model_class"] == "PeftModelForCausalLM" and weights["matched_tensors"] == weights["saved_tensors"] > 0,
        "bf16": set(model["parameter_dtypes"]) == {"torch.bfloat16"},
        "inference": not model["training"] and model["trainable_parameters"] == 0,
        "requests": all(all(r["checks"].values()) for r in results)}
    write(output / "http_acceptance.json", {"passed": all(checks.values()), "checks": checks, "results": results})
    print(json.dumps({"passed": all(checks.values()), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
