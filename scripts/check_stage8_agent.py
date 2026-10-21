"""真实加载独立API入口，用固定公开S01请求验证所选adapter推理。"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "mymind"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["sft", "rl"])
    args = parser.parse_args()
    root = Path(os.environ["MYMIND_WORK_ROOT"])
    run = root / "runs/stage8-20261016"
    export = root / "exports/stage8-20261016"
    output = run / "demo" / args.group
    output.mkdir(parents=True, exist_ok=True)
    config_path = export / f"agent-{args.group}.json"
    os.environ["MYMIND_AGENT_CONFIG"] = str(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    opened = set()
    def audit(event, arguments):
        if event == "open" and isinstance(arguments[0], (str, bytes)):
            opened.add(os.fsdecode(arguments[0]))
    sys.addaudithook(audit)
    from fastapi.testclient import TestClient
    from api import policy_app
    from experiments.agentic_rl.stage7 import RunBudget
    http_times = []
    original = policy_app.rerank_gateway
    def gateway(retrieval_config):
        result = original(retrieval_config)
        async def on_request(request):
            http_times.append(time.time())
        result.client._client.event_hooks["request"].append(on_request)
        return result
    policy_app.rerank_gateway = gateway
    public = json.loads((export / "demo-request.json").read_text(encoding="utf-8"))
    budget_config = {"limits_file": "runs/stage7-preparation-20261014/formal/limits.json",
                     "stage_authorization": "runs/stage8-20261016/authorization.json"}
    completed = []
    for path in (run / "demo").glob("*/app/shutdown.json"):
        completed.append(json.loads(path.read_text())["requests_completed"])
    if sum(completed) >= 4:
        raise RuntimeError("演示轨迹上限4条已用完")
    with TestClient(policy_app.app) as client:
        status = client.get("/status")
        status.raise_for_status()
        model = status.json()["model"]
        assert model["adapter_path"] == str(root / config["adapter_dir"])
        assert model["adapter_weights"]["saved_tensors"] == model["adapter_weights"]["matched_tensors"] == 504
        assert not model["training"] and model["trainable_parameters"] == 0
        policy_app.app.state.generation.torch.manual_seed(42)
        embedding_before = policy_app.app.state.retrieval.knowledge.embedding.usage["requests"]
        RunBudget(root, budget_config).reserve("student_trajectories", 1, f"stage8-independent-demo-{args.group}-S01")
        start = time.monotonic()
        response = client.post("/requests", json=public)
        response.raise_for_status()
        row = response.json()
        row["elapsed_s"] = time.monotonic()-start
        row["search_requests"] = sum(len(g["assistant_message"].get("tool_calls", [])) for g in row["generations"])
        row["cost"] = {"embedding_requests": policy_app.app.state.retrieval.knowledge.embedding.usage["requests"]-embedding_before,
                       "rerank_http_requests": len(http_times), "rerank_request_started_unix_s": http_times,
                       "usage": [{"embedding": d.get("embedding_usage"), "rerank": d.get("diagnostics", {}).get("rerank_usage")} for d in row["tool_diagnostics"]],
                       "gpu_generation_s": sum(g.get("latency_ms", 0)/1000 for g in row["generations"])}
        expected = policy_app.PolicyRequest.model_validate(public).public_task(row["request_id"]).messages()
        assert row["generations"][0]["input_messages"] == expected
        stored = client.get(f"/requests/{row['request_id']}")
        stored.raise_for_status()
        assert stored.json()["raw_result"] == row["raw_result"]
    forbidden = [p for p in opened if any(name in Path(p).name.lower() for name in
                 ("private_standard", "dev_standard", "trainer_state", "optimizer", "rng_state", "scheduler.pt"))
                 and not p.endswith((".py", ".pyc"))]
    assert not forbidden, forbidden
    proof = {"group": args.group, "public_request": "S01", "new_trajectories": 1,
             "actual_api_entry": "api.policy_app:app", "time_utc": datetime.now(timezone.utc).isoformat(),
             "model": model, "private_or_trainer_state_files_opened": forbidden,
             "initial_policy_messages_equal_public_request": True,
             "infrastructure_failure": row["infrastructure_failure"],
             "runtime_error": row["runtime_error"], "protocol_valid": row["raw_result"] is not None and not row["runtime_error"],
             "business_grade": "演示入口不加载标准，不在策略进程内判分",
             "requests_completed": 1, "shutdown_record": str(output / "app/shutdown.json")}
    for name, value in (("trajectory.json", row), ("verification.json", proof), ("opened-files.json", sorted(opened))):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: proof[k] for k in ("group", "new_trajectories", "protocol_valid", "private_or_trainer_state_files_opened")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
