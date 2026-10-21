"""生成验证、最终测试和公开基本回归分开的结果交付。"""
import csv
import json
from pathlib import Path

from stage8_results import add_breakdown_counts, describe, read, records, usage

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "agentic_runs"
OUT = RUNS / "stage8-20261016"
GROUPS = ("base", "sft", "rl")
LABELS = {"base": "Base三搜", "sft": "SFT120三搜", "rl": "RL40三搜"}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def differences(evaluations):
    result = {}
    for left, right in (("sft", "base"), ("rl", "base"), ("rl", "sft")):
        a, b = evaluations[left], evaluations[right]
        result[f"{left}_minus_{right}"] = {
            "combination_equal_success_pp": 100*(a["combination_equal_success_rate"]-b["combination_equal_success_rate"]),
            "combination_equal_severe_pp": 100*(a["combination_equal_severe_error_rate"]-b["combination_equal_severe_error_rate"]),
            "instance_success_pp": 100*(a["instance_success_rate"]-b["instance_success_rate"]),
            **{key: a[key]-b[key] for key in ("complete_success", "severe_errors", "protocol_failures")}}
    return result


def table(evaluations):
    lines = ["| 配置 | 完整成功 | 实例成功 | 组合等权成功 | 严重错误 | 等权严重错误 | 协议失败 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in GROUPS:
        r = evaluations[name]
        lines.append(f"| {LABELS[name]} | {r['complete_success']}/{r['planned_tasks']} | {r['instance_success_rate']:.2%} | {r['combination_equal_success_rate']:.2%} | {r['severe_errors']} | {r['combination_equal_severe_error_rate']:.2%} | {r['protocol_failures']} |")
    return "\n".join(lines)


def main():
    original = read(OUT / "RESULT.json")
    final = {k: describe(OUT / "test" / k) for k in GROUPS}
    regression = {k: describe(OUT / "regression" / k) for k in GROUPS}
    validation = {k: read(RUNS / "stage7c-20261016/RESULT.json")["evaluations"][k] for k in GROUPS}
    public = {r["task_id"]: r for r in read(ROOT / "data/agentic_rl/formal/public_tasks.json")}
    standards = {r["task_id"]: r for r in read(RUNS / "pre-stage7-fixes-20261014/standards/formal/private_standards.v5.json")}
    validation_paths = {"base": "base", "sft": "sft-validation-120", "rl": "grpo-validation-40"}
    for group, name in validation_paths.items():
        metrics = read(RUNS / "stage7-preparation-20261014/formal" / name / "metrics.json")
        validation[group]["breakdown"] = add_breakdown_counts(metrics, public, standards)
    commands = {p.stem: read(p) for p in (OUT / "commands").glob("*.json") if not p.stem.endswith("-checked")}
    for scope, evaluations, count in (("test", final, 120), ("regression", regression, 13)):
        for group, result in evaluations.items():
            assert result["planned_tasks"] == result["finished_tasks"] == count
            assert result["all_tasks_finished"] and not result["unresolved_infrastructure_failures"]
            assert commands[f"{scope}-{group}"]["exit_code"] == 0
    ledger = read(OUT / "limits-after.json")
    authorization = read(OUT / "authorization.json")
    added = {k: ledger["consumed"].get(k, 0)-authorization["consumed_before"].get(k, 0) for k in ledger["limits"]}
    assert all(added[k] <= authorization["additional_limits"][k] for k in added)
    rows = [r for scope in ("test", "regression") for group in GROUPS
            for r in records(OUT / scope / group / "first_records.jsonl")]
    demos = []
    for group in ("sft", "rl"):
        path = OUT / "demo" / group / "trajectory.json"
        if path.exists():
            demos.append(read(path))
    result = {"date": "2026-10-16", "baseline": "stage7-baseline-20261014-v13-v5-158",
              "grading_revision": "pre-stage7-grading-v13-20261014", "standard_version": "v5",
              "model_revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
              "policy_execution_revision": "stage7-http-trace-20261015-v5",
              "comparison_configurations": "comparison-configurations.json",
              "model_selections": {phase: read(RUNS / f"stage7-preparation-20261014/formal/{phase}-selection.json") for phase in ("sft", "grpo")},
              "validation": validation, "final_test": final, "basic_regression": regression,
              "validation_differences": differences(validation), "final_test_differences": differences(final),
              "commands": commands, "stage8_consumed": added,
              "stage8_authorized_remaining": {k: v-added[k] for k, v in authorization["additional_limits"].items()},
              "cumulative_ledger_remaining": {k: v-ledger["consumed"].get(k, 0) for k, v in ledger["limits"].items()},
              "stage8_usage": original["stage8_usage"],
              "retained_trajectory_usage": usage(rows+demos),
              "retained_trajectories": len(rows)+len(demos),
              "historical_one_search": original.get("historical_one_search") or {
                  scope: original[scope]["one_search"]
                  for scope in ("validation", "final_test", "basic_regression")},
              "new_demo_trajectories": len(demos),
              "new_diagnostic_trajectories": 0, "no_new_training_updates": added["sft_updates"] == added["grpo_updates"] == 0,
              "historical_failed_http_without_usage": 12, "new_independent_review": False}
    write(OUT / "RESULT.json", result)
    with (OUT / "main-results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        keys = ["split", "configuration", "planned_tasks", "complete_success", "instance_success_rate", "combination_equal_success_rate", "severe_errors", "combination_equal_severe_error_rate", "protocol_failures", "mean_reward"]
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for scope, mapping in (("validation", validation), ("final_test", final), ("basic_regression", regression)):
            for group in GROUPS:
                writer.writerow({"split": scope, "configuration": group, **{k: mapping[group][k] for k in keys[2:]}})
    with (OUT / "breakdown.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "configuration", "dimension", "item", "tasks", "complete_success", "success_rate", "severe_errors", "severe_error_rate", "protocol_failures", "infrastructure_failures"])
        for scope, mapping in (("validation", validation), ("final_test", final), ("basic_regression", regression)):
            for group in GROUPS:
                for dimension, values in mapping[group]["breakdown"].items():
                    for item, metrics in values.items():
                        writer.writerow([scope, group, dimension, item, metrics["tasks"], metrics.get("complete_success", round(metrics["tasks"]*metrics["success_rate"])), metrics["success_rate"], metrics.get("severe_errors", round(metrics["tasks"]*metrics["severe_error_rate"])), metrics["severe_error_rate"], metrics.get("protocol_failures"), metrics["infrastructure_failures"]])
    diff = result["final_test_differences"]
    lines = ["# MyMind验证与最终测试结果", "", "验证集用于预定节点选模，最终测试在原配置冻结后执行。当前保留Base、SFT120和RL40三组，每题一次贪心首次轨迹；原完整分母和成绩不变。严重错误与协议失败可能重叠，不相加成互斥失败数。", "", "## 最终测试120题", "", table(final), "",
             "最终测试8个组合各15题，因此组合等权率和实例率相等。领域、主题、组合及标准预定处置分项见breakdown.csv及RESULT.json。", "",
             "| 差值 | 组合等权成功百分点 | 等权严重错误百分点 | 成功题净变化 |", "| --- | ---: | ---: | ---: |"]
    for key in ("sft_minus_base", "rl_minus_base", "rl_minus_sft"):
        d = diff[key]
        lines.append(f"| {key} | {d['combination_equal_success_pp']:+.2f} | {d['combination_equal_severe_pp']:+.2f} | {d['complete_success']:+d} |")
    lines += ["", "## 验证集80题", "", table(validation), "", "验证组合大小不同，等权率与实例率不混用。SFT选择120步、RL选择40步均沿用正式selection；未用最终测试重新选模。", "",
              "## S01至S13基本回归", "", table(regression), "", "公开开发任务及原对应标准单列，不混入120题最终测试，不参与选模。", "",
              "## 奖励与检索", "", "| 配置 | −1 | 0 | 0.3 | 1 | 平均奖励 | 请求搜索 | 实际搜索 | 观察后补检题数 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in GROUPS:
        r = final[group]
        counts = {float(k): v for k, v in r["reward_distribution"].items()}
        b = r["retrieval"]
        lines.append(f"| {LABELS[group]} | {counts.get(-1,0)} | {counts.get(0,0)} | {counts.get(.3,0)} | {counts.get(1,0)} | {r['mean_reward']:.5f} | {b['requested_searches']} | {b['actual_searches']} | {len(b['post_observation_search_tasks'])} |")
    analysis = read(RUNS / "analysis-20261016-intent-rag/metrics.json")["evaluations"]
    lines += ["", "## 主要处置与证据可见性", "",
              "主要处置为answer/request_info/escalate三类，不是独立的19类意图识别。全部预定题目参与准确率和Macro-F1；无输出记错，其他字段协议失败不抹掉可读处置字段的单项成绩。以下读取已有首次轨迹的离线补算，不新增模型调用。", "",
              "| 模型 | 验证处置正确 | 验证Macro-F1 | 最终处置正确 | 最终Macro-F1 |", "| --- | ---: | ---: | ---: | ---: |"]
    for group in GROUPS:
        v = analysis["validation"][group]["summary"]
        t = analysis["final_test"][group]["summary"]
        lines.append(f"| {LABELS[group]} | {v['disposition_correct']}/80（{v['disposition_accuracy']:.2%}） | {v['disposition_macro_f1']:.4f} | {t['disposition_correct']}/120（{t['disposition_accuracy']:.2%}） | {t['disposition_macro_f1']:.4f} |")
    lines += ["", "最终测试三组首搜Top5相关来源命中、标准涉及政策条款的正文覆盖均100%，全部条款可见均120/120；验证均80/80。相关条款含标准允许的备选和可选输出，只在同来源实际可见正文中完整匹配。来源命中和正文可见不等同于正确引用或业务成功；仅两份政策来源的结果不外推至大知识库。动态查询没有独立相关性标注，不补造Precision@K/nDCG。", "",
              "## 收益、退化与真实案例", "",
              "SFT与RL共同成功40题，RL新增3成功也失去3成功，集合见[paired-changes.json](paired-changes.json)。最终主要处置正确数99→106（+5.83个百分点），配对新增9题正确、退化2题；完整任务成功仍同为43/120。RL主要减少把补信息错判为升级，不能概括为每个类别均改善。", "",
              "| 真实首次轨迹 | 说明 |", "| --- | --- |",
              "| [RL40 R07 001](../stage7-preparation-20261014/formal/grpo-validation-40/M6-R07-001.json) | 验证成功，必要elapsed_workdays=unknown完整；原SFT为0.3。 |",
              "| [RL40 D07 001](../stage7-preparation-20261014/formal/grpo-validation-40/M6-D07-001.json) | 验证多对象失败；两笔退款时效、未知状态和对象绑定错误。 |",
              "| [RL40 F03 005](../stage7-preparation-20261014/formal/grpo-validation-40/M6-F03-005.json) | 验证严重错误，确认了用户未提供的http_status和occurrence。 |",
              "| [RL40 R07 003](../stage7-preparation-20261014/formal/grpo-validation-40/M6-R07-003.json) | 验证0.3，仍漏必要的elapsed_workdays=unknown。 |", "",
              "最终测试P11 006新增成功，D03 006由成功退化为严重错误，D08 002因条件绑定缺项退化；F04 001消除严重断言后仍不完整。完整成对原始案例见[CASES.json](CASES.json)，正文核对见[case-evidence-coverage.json](case-evidence-coverage.json)。这些案例标准涉及的政策正文均已实际展示，失败不能仅解释为没搜到政策。验证案例不混入最终测试。", "",
              "最终D09/F04仍0/15成功；验证D07仍0/15、F03仍8/10严重错误。用户未知事实、多个对象绑定、必要unknown和条件完整性仍是当前边界。"]
    lines += ["", "观察进入上下文不等于有效使用全部证据；没有实际观察后搜索的轨迹，不宣称自适应补检。固定政策下的未见组合、单种子与本轮RL路径增量分别解释，不推广为未知政策泛化、稳定提升或优于同预算继续SFT。", "", "原一次检索消融仅保留historical_one_search汇总和原累计费用，不再提供该分支的逐题轨迹。原累计534条轨迹与预算消耗保持；当前原始记录保留三组测试、三组公开回归和两条演示，共401条。", "", "原始首次表和逐题轨迹在[test](test/)、[regression](regression/)，逐命令退出、硬时限、usage与余额在[commands](commands/)；成本见[COSTS](COSTS.md)，训练机制见[AGENTIC_RL_GUIDE](AGENTIC_RL_GUIDE.md)，成对案例见[CASES.json](CASES.json)。"]
    (OUT / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"final_test": {k: {f: v[f] for f in ("complete_success", "severe_errors", "protocol_failures")} for k, v in final.items()}, "stage8_consumed": added}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
