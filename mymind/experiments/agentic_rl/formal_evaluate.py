"""同一公共 runner 的正式首次评测，故障和未完成题目保留在分母。"""
import json
import time
from copy import deepcopy
from datetime import datetime, timezone

from agents.customer_policy.configuration import rerank_gateway
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import QwenGeneration
from agents.customer_policy.protocol import VisibleTrajectory
from agents.customer_policy.retrieval import build_retrieval
from agents.customer_policy.runner import run_task
from experiments.agentic_rl.grading import grade_result
from experiments.agentic_rl.metrics import summarize
from experiments.agentic_rl.runtime import write_json
from experiments.agentic_rl.stage7 import check_inputs, RunBudget, processed_infrastructure_faults


def install_http_diagnostics(gateway, path):
    """独立诊断保存实际HTTP阶段及异常链，不写认证头或改变请求设置。"""
    start = time.monotonic()
    sequence = [0]
    def emit(event, **fields):
        row = {"time_utc": datetime.now(timezone.utc).isoformat(),
               "elapsed_s": time.monotonic()-start, "event": event, **fields}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False)+"\n")
    def exceptions(error):
        chain = []
        while error is not None:
            chain.append({"type": type(error).__name__, "message": str(error)})
            error = error.__cause__
        return chain
    async def on_request(request):
        sequence[0] += 1
        number = sequence[0]
        emit("http.request", request_number=number, method=request.method, url=str(request.url),
             body=json.loads(request.content.decode("utf-8")), timeout=request.extensions.get("timeout"),
             retry_count=request.headers.get("x-stainless-retry-count"))
        async def trace(event, info):
            fields = {k: (v.decode() if isinstance(v, bytes) else v)
                      for k in ("host", "port", "timeout", "local_address")
                      if (v := info.get(k)) is not None}
            if info.get("exception") is not None:
                fields["exception_chain"] = exceptions(info["exception"])
            connection = info.get("return_value")
            if event in {"connection.connect_tcp.complete", "connection.start_tls.complete"} and hasattr(connection, "get_extra_info"):
                fields["server_addr"] = connection.get_extra_info("server_addr")
                fields["client_addr"] = connection.get_extra_info("client_addr")
            emit(event, request_number=number, **fields)
        request.extensions["trace"] = trace
    async def on_response(response):
        emit("http.response", request_number=sequence[0], status=response.status_code,
             headers={key: response.headers[key] for key in
                      ("date", "server", "content-type", "x-request-id", "x-ds-request-id", "cf-ray")
                      if key in response.headers})
    original = gateway.complete
    async def complete(request):
        emit("llm.start", model=request.model, max_tokens=request.max_tokens, temperature=request.temperature)
        try:
            result = await original(request)
        except Exception as error:
            emit("llm.failed", exception_chain=exceptions(error))
            raise
        emit("llm.complete", response_model=result.metadata.get("response_model"),
             input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens)
        return result
    gateway.client._client.event_hooks["request"].append(on_request)
    gateway.client._client.event_hooks["response"].append(on_response)
    gateway.complete = complete


def load_recovery(output, tid, partial):
    directory = output / "recovery" / tid
    source = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    original = directory / "original.partial.json"
    if (source["task_id"] != tid or source["original_command_exit_code"] != 124 or
            original.read_bytes() != partial.read_bytes()):
        raise ValueError("恢复来源与原partial不一致")
    if (directory / "progress.partial.json").exists():
        raise ValueError("已有恢复尝试；新中断须暂停，不能自动再次恢复")
    state = json.loads(original.read_text(encoding="utf-8"))
    if (source["historical_cost"]["embedding_requests"] != 2 or
            source["historical_cost"]["rerank_http_requests"] != 2 or
            len(state["trajectory"]["searches"]) != 2):
        raise ValueError("本次恢复须保留已核实的两次真实检索及请求记录")
    return state, source, directory


async def evaluate(root, config, output, tasks, standards, clauses):
    check_inputs(root, config, tasks, standards, "evaluate")
    if config.get("adapter_selection"):
        selection = json.loads((root / config["adapter_selection"]).read_text(encoding="utf-8"))
        if selection["phase"] != "sft" or selection["baseline"] != config["baseline"]:
            raise ValueError("一次检索对照须读取同一正式SFT选模记录")
        config = {**config, "adapter_path": selection["selected_adapter"]}
        write_json(output / "resolved_model.json", {"selection": config["adapter_selection"], "adapter_path": config["adapter_path"]})
    ledger = RunBudget(root, config)
    record_path = output / "first_records.jsonl"
    records = ([json.loads(line) for line in record_path.read_text(encoding="utf-8").splitlines()]
               if record_path.exists() else [])
    resolved = processed_infrastructure_faults(root, output, records)
    if any(r["infrastructure_failure"] and r["task_id"] not in resolved for r in records):
        raise RuntimeError("首次表含未处理基础设施故障，暂停")
    finished = {r["task_id"] for r in records}
    gateway = rerank_gateway(config["retrieval"])
    http_requests = [0]
    request_times = []
    async def on_request(request):
        http_requests[0] += 1
        request_times.append(time.time())
    gateway.client._client.event_hooks["request"].append(on_request)
    if config["run_kind"] == "diagnostic":
        install_http_diagnostics(gateway, output / "http-trace.jsonl")
    retrieval = build_retrieval(root, config["retrieval"], gateway)
    def metrics():
        result = summarize(config["task_ids"], records, tasks, standards)
        result.update(table_kind="infrastructure_diagnostic" if config["run_kind"] == "diagnostic" else "first_table",
                      diagnostic_of=config.get("diagnostic_of"),
                      processed_infrastructure_tasks=sorted(resolved),
                      unresolved_infrastructure_failures=sum(r["infrastructure_failure"] and r["task_id"] not in resolved for r in records))
        return result
    generation = None
    try:
        if retrieval.knowledge.stats()["pending_main"] or not retrieval.knowledge._main.count():
            raise RuntimeError("须复用完整 Qwen 实验索引")
        generation = QwenGeneration(root / config["model_dir"], config["generation"],
            root / config["adapter_path"] if config.get("adapter_path") else None)
        generation.torch.manual_seed(config["seed"])
        write_json(output / "model_load.json", generation.load_info)
        for tid in config["task_ids"]:
            if tid in finished:
                continue
            path = output / f"{tid}.json"
            partial = output / f"{tid}.partial.json"
            start = time.monotonic()
            state, source, recovery_dir = None, None, None
            if partial.exists() and config.get("resume_partial_task") == tid:
                state, source, recovery_dir = load_recovery(output, tid, partial)
            if partial.exists() and state is None:
                row = json.loads(partial.read_text(encoding="utf-8"))
                row.update(runtime_error={"kind": "interrupted", "detail": "首次中断保留，不重采替换"},
                           infrastructure_failure=True, elapsed_s=None, cost={})
            else:
                if state is None:
                    ledger.reserve("student_trajectories", 1, f"evaluate-{output.name}-{tid}")
                before_embed = retrieval.knowledge.embedding.usage["requests"]
                before_rerank = http_requests[0]
                env = PolicyEnvironment(retrieval)
                env.max_searches = config["search_limit"]
                progress = recovery_dir / "progress.partial.json" if state is not None else partial
                row = await run_task(tasks[tid], generation, env, budget=config["budget"],
                    on_progress=lambda value: write_json(progress, value), resume_state=state)
                row["elapsed_s"] = time.monotonic()-start
                row["search_requests"] = sum(len(g["assistant_message"].get("tool_calls", [])) for g in row["generations"])
                row["cost"] = {"embedding_requests": retrieval.knowledge.embedding.usage["requests"]-before_embed,
                    "rerank_http_requests": http_requests[0]-before_rerank,
                    "rerank_request_started_unix_s": request_times[before_rerank:],
                    "usage": [{"embedding": d.get("embedding_usage"), "rerank": d.get("diagnostics", {}).get("rerank_usage")} for d in row["tool_diagnostics"]],
                    "gpu_generation_s": sum(g.get("latency_ms", 0)/1000 for g in row["generations"])}
                if state is not None:
                    new_cost = deepcopy(row["cost"])
                    new_cost["usage"] = new_cost["usage"][len(state["tool_diagnostics"]):]
                    new_cost["gpu_generation_s"] = sum(g.get("latency_ms", 0)/1000
                        for g in row["generations"][len(state["generations"]):])
                    for key in ("embedding_requests", "rerank_http_requests"):
                        row["cost"][key] += source["historical_cost"][key]
                    row["recovery"] = {"source": str((recovery_dir / "source.json").relative_to(output)),
                        "boundary": "完整工具观察后的消息边界，重新计算未落盘生成轮",
                        "original_command_exit_code": source["original_command_exit_code"],
                        "historical_cost": source["historical_cost"], "new_cost": new_cost,
                        "resumed_elapsed_s": row["elapsed_s"],
                        "interrupted_generation_output_tokens": None,
                        "interrupted_generation_elapsed_s": None,
                        "original_trajectory_elapsed_s": None}
                    # 原partial未保存完整耗时；保留已知生成成本，未知中断开销不记成0。
                    row["elapsed_s"] = None
            row["grade"] = grade_result(standards[tid], row["raw_result"],
                VisibleTrajectory.model_validate(row["trajectory"]), clauses).to_dict()
            write_json(path, row)
            with record_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False)+"\n")
            records.append(row)
            write_json(output / "metrics.json", metrics())
            print(f"{tid} reward={row['grade']['reward']} infrastructure={row['infrastructure_failure']} elapsed={row['elapsed_s']}", flush=True)
            if row["infrastructure_failure"]:
                # 首次记录保留，其余未启动题目由指标显式列出，暂停诊断。
                raise RuntimeError("首次评测基础设施故障，暂停；不得替换该结果")
    finally:
        write_json(output / "metrics.json", metrics())
        retrieval.knowledge.close()
        retrieval.knowledge.embedding.close()
        await gateway.client.close()
