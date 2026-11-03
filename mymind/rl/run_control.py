"""正式训练划分、有限运行额度与 checkpoint 交接。"""
import json
from pathlib import Path

from rl.runtime import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_diagnostic_configuration(original, diagnostic):
    """诊断只改变题目子集、记录位置和命令时限，模型及策略设置保持。"""
    changed = {"run_kind", "task_ids", "output_dir", "timeout_s", "diagnostic_of", "diagnostic_trigger"}
    if ({k: v for k, v in original.items() if k not in changed} !=
            {k: v for k, v in diagnostic.items() if k not in changed} or
            diagnostic.get("run_kind") != "diagnostic" or
            diagnostic.get("diagnostic_of") != original["output_dir"]):
        raise ValueError("故障诊断须沿用首次模型及共同配置/额度")


def processed_infrastructure_faults(root, output, records):
    """核对单列健康诊断，只标记已处理，不修改首次故障或成绩。"""
    receipt = output / "infrastructure_resolution.json"
    if not receipt.exists():
        return set()
    original = read(output / "configuration.json")
    faults = {r["task_id"] for r in records if r.get("infrastructure_failure")}
    resolved = set()
    for tid, directory in read(receipt)["diagnostics"].items():
        if tid not in faults:
            raise ValueError("处理记录必须关联原首次基础设施故障")
        diagnostic = read(root / directory / "configuration.json")
        check_diagnostic_configuration(original, diagnostic)
        if diagnostic["task_ids"] != [tid] or diagnostic["output_dir"] != directory:
            raise ValueError("诊断题目或记录位置与首次故障不对应")
        rows = [json.loads(line) for line in (root / directory / "first_records.jsonl").read_text(encoding="utf-8").splitlines()]
        metrics = read(root / directory / "metrics.json")
        execution = read(root / directory / "execution.json")
        if (len(rows) != 1 or rows[0]["task_id"] != tid or rows[0].get("infrastructure_failure") or
                (rows[0].get("runtime_error") or {}).get("kind") in {"generation_failure", "tool_failure", "interrupted"} or
                not metrics["all_tasks_finished"] or metrics["infrastructure_failures"] or
                metrics["table_kind"] != "infrastructure_diagnostic" or
                metrics["diagnostic_of"] != original["output_dir"] or execution["exit_code"] != 0):
            raise ValueError("故障诊断未完整退出或仍有未处理基础设施故障")
        resolved.add(tid)
    return resolved


def check_inputs(root, config, tasks, standards, purpose):
    if config.get("run_purpose") != purpose:
        raise ValueError("运行用途与入口不一致")
    ids = config["task_ids"]
    if not ids or len(ids) != len(set(ids)) or any(t not in tasks or t not in standards for t in ids):
        raise ValueError("任务清单为空、重复或缺少公开/私有输入")
    baseline = read(root / config["baseline"])
    kind = config.get("run_kind")
    if kind not in {"formal", "preflight", "diagnostic"} or (purpose != "evaluate" and kind == "diagnostic"):
        raise ValueError("训练用途只能是formal或preflight；diagnostic仅用于基础设施故障单列评测")
    if kind in {"formal", "preflight"} and config["limits_file"] != baseline[f"{kind}_limits"]:
        raise ValueError("不能另换额度文件重置本轮预算")
    if kind == "preflight" and ids != baseline["preflight_task_ids"]:
        raise ValueError("预检必须使用采样前固定的任务子集")
    if config["standards"] != baseline["standards"] or config["grading_revision"] != baseline["grading_revision"]:
        raise ValueError("必须使用当前运行基线的 v5 标准及 v13 评分")
    for key in ("model_id", "model_revision", "model_dir", "retrieval", "context_tokens", "turn_tokens"):
        if config[key] != baseline[key]:
            raise ValueError(f"共同运行配置变化: {key}")
    if config.get("budget") != baseline["budget"]:
        raise ValueError("共同上下文及生成预算变化")
    if config["tasks"] != baseline["public_tasks"] or config["policy_support"] != baseline["policy_support"]:
        raise ValueError("公开输入或政策支持映射变化")
    if purpose in {"sft", "grpo"}:
        pool = read(root / baseline["common_pool"])["effective_sft_rl_pool"]
        if any(standards[t].split != "train" for t in ids):
            raise ValueError("验证/测试不能进入训练采样池")
        if config["run_kind"] == "formal" and ids != pool:
            raise ValueError("正式 SFT/RL 必须使用同一固定158池")
        if config["run_kind"] == "preflight" and (not 2 <= len(ids) <= 4 or not set(ids) <= set(pool)):
            raise ValueError("预检只允许共同池内2—4题")
        if purpose == "sft" and config.get("sft_adapter"):
            raise ValueError("SFT 必须从原始 Qwen 新建 LoRA")
        if kind == "preflight" and config["training"]["max_steps"] > (2 if purpose == "sft" else 4):
            raise ValueError("预检更新步数超过授权上限")
        if purpose == "grpo":
            if not config.get("sft_adapter"):
                raise ValueError("RL 必须显式加载 SFT adapter")
            if config["run_kind"] == "formal":
                selection = read(root / config["sft_selection"])
                if selection["phase"] != "sft" or selection["baseline"] != config["baseline"] or selection["selected_adapter"] != config["sft_adapter"]:
                    raise ValueError("RL 起点与正式 SFT 选模交接不一致")
                origin = read(root / selection["training_output"] / "configuration.json")
                if origin["run_kind"] != "formal" or origin["run_purpose"] != "sft" or origin["task_ids"] != pool:
                    raise ValueError("开发/预检 adapter 不能作为正式起点")
            else:
                origin = read((root / config["sft_adapter"]).parent / "configuration.json")
                if origin["run_kind"] != "preflight" or origin["run_purpose"] != "sft" or origin["task_ids"] != ids:
                    raise ValueError("预检 RL 必须接本轮预检 SFT")
    else:
        split = config["evaluation_split"]
        if any(standards[t].split != split for t in ids):
            raise ValueError("评测任务与声明划分不一致")
        if config["run_kind"] == "formal" and set(ids) != {t for t,s in standards.items() if s.split == split}:
            raise ValueError("正式首次评测必须包含全部预定任务")
        if config["generation"]["temperature"] != 0 or config.get("samples_per_task", 1) != 1:
            raise ValueError("正式/预检评测使用每题一次贪心轨迹")
        if config["generation"] != baseline["evaluation_generation"]:
            raise ValueError("共同评测解码变化")
        if config["search_limit"] not in {1, 3}:
            raise ValueError("仅支持共同三搜或固定一次检索")
        if kind == "diagnostic":
            source = root / config["diagnostic_of"]
            original = read(source / "configuration.json")
            check_diagnostic_configuration(original, config)
            trigger = root / config.get("diagnostic_trigger", config["diagnostic_of"])
            triggering_config = read(trigger / "configuration.json")
            for key in ("baseline", "tasks", "standards", "grading_revision", "policy_support", "generation", "retrieval", "budget", "search_limit", "limits_file"):
                if triggering_config[key] != original[key]:
                    raise ValueError("同题对照诊断的共同设置与触发故障不一致")
            first = {r["task_id"]:r for r in [json.loads(line) for line in (trigger / "first_records.jsonl").read_text(encoding="utf-8").splitlines()]}
            if root / config["output_dir"] == source or any(not first.get(t, {}).get("infrastructure_failure") for t in ids):
                raise ValueError("仅将首次基础设施故障任务诊断到单独目录；正常低分不重采")


class RunBudget:
    """单作业按顺序记账；分布式训练只由 rank0 在同步点扣全组额度。"""
    def __init__(self, root, config):
        self.path = root / config["limits_file"]
        self.authorization = (read(root / config["stage_authorization"])
                              if config.get("stage_authorization") else None)

    def reserve(self, key, amount, reason):
        state = read(self.path)
        if self.authorization:
            allowance = self.authorization["additional_limits"]
            before = self.authorization["consumed_before"]
            requested = {key: amount}
            if key in {"student_trajectories", "infrastructure_trajectories"}:
                requested["api_request_allowance"] = amount * 12
            for resource, increment in requested.items():
                if state["consumed"].get(resource, 0) - before.get(resource, 0) + increment > allowance[resource]:
                    raise RuntimeError(f"阶段新增授权额度不足: {resource}")
        consumed = state["consumed"].get(key, 0)
        if consumed + amount > state["limits"][key]:
            raise RuntimeError(f"有限运行额度不足: {key} {consumed}+{amount}>{state['limits'][key]}")
        state["consumed"][key] = consumed + amount
        if key in {"student_trajectories", "infrastructure_trajectories"}:
            # 固定接口每搜一次 Qwen 查询编码，DeepSeek 重排至多两次HTTP尝试。
            # 另留一次编码余量；每轨迹三搜的保守请求上限为12。
            api = state["consumed"].get("api_request_allowance", 0) + amount * 12
            if api > state["limits"]["api_request_allowance"]:
                raise RuntimeError("API 请求预留额度不足")
            state["consumed"]["api_request_allowance"] = api
        state["events"].append({"kind": key, "amount": amount, "reason": reason})
        write_json(self.path, state)




def reserve_distributed(root, config, key, amount, reason):
    import torch.distributed as distributed
    status = [None]
    if distributed.get_rank() == 0:
        try:
            RunBudget(root, config).reserve(key, amount, reason)
        except RuntimeError as error:
            status[0] = str(error)
    distributed.broadcast_object_list(status, src=0)
    if status[0]:
        raise RuntimeError(status[0])


def select_checkpoint(root, plan_path):
    plan = read(plan_path)
    config = read(root / plan["training_output"] / "configuration.json")
    if config["run_kind"] != "formal" or config["run_purpose"] != plan["phase"]:
        raise ValueError("仅选择正式训练已更新的候选")
    updates = [json.loads(line) for line in (root / plan["training_output"] / "updates-rank0.jsonl").read_text().splitlines()]
    candidates = []
    for node in plan["nodes"]:
        metrics = read(root / node["evaluation"] / "metrics.json")
        eval_config = read(root / node["evaluation"] / "configuration.json")
        adapter = f"{plan['training_output']}/checkpoint-{node['step']}"
        resolved = set()
        if metrics["infrastructure_failures"]:
            records = [json.loads(line) for line in (root / node["evaluation"] / "first_records.jsonl").read_text(encoding="utf-8").splitlines()]
            resolved = processed_infrastructure_faults(root, root / node["evaluation"], records)
            if any(r["infrastructure_failure"] and r["task_id"] not in resolved for r in records):
                raise ValueError("选模节点仍有未处理基础设施故障")
        if (eval_config["adapter_path"] != adapter or eval_config["baseline"] != config["baseline"] or
                eval_config["evaluation_split"] != "validation" or eval_config["search_limit"] != 3 or
                not metrics["all_tasks_finished"]):
            raise ValueError("选模节点必须为完整、同配置的验证评测；故障须单列诊断")
        if not any(0 < r["step"] <= node["step"] and r["parameter_delta_l1"] > 0 for r in updates):
            raise ValueError("候选没有实际参数更新")
        if read(root / adapter / "trainer_state.json")["global_step"] != node["step"]:
            raise ValueError("checkpoint 步数不对应")
        candidates.append({**node, "adapter": adapter,
            "success": metrics["combination_equal_success_rate"],
            "severe": metrics["combination_equal_severe_error_rate"],
            "original_infrastructure_failures": metrics["infrastructure_failures"],
            "processed_infrastructure_tasks": sorted(resolved)})
    selected = min(candidates, key=lambda r: (-r["success"], r["severe"], r["step"]))
    result = {"phase": plan["phase"], "baseline": config["baseline"],
        "training_output": plan["training_output"], "selected_adapter": selected["adapter"],
        "selected_step": selected["step"], "candidates": candidates,
        "rule": "组合等权完整成功率最高；完全同分严重错误率更低；再同分步数更早"}
    write_json(root / plan["selection_output"], result)
    if plan.get("rl_config"):
        target = root / plan["rl_config"]
        rl = read(target)
        rl["sft_adapter"] = selected["adapter"]
        rl["sft_selection"] = plan["selection_output"]
        write_json(target, rl)
    return result


def main():
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", required=True)
    args = parser.parse_args()
    print(json.dumps(select_checkpoint(Path(os.environ["MYMIND_WORK_ROOT"]), args.select), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
