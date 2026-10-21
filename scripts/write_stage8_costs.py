"""由本轮原始命令及用量生成费用说明，未知费用保持未知。"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from stage8_results import read

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "agentic_runs/stage8-20261016"
LABELS = {"base": "Base三搜", "sft": "SFT120三搜", "rl": "RL40三搜", "one_search": "同SFT120一搜"}


def amount(account):
    response = account.get("response", {})
    values = [b["total_balance"] for b in response.get("balance_infos", []) if b["currency"] == "CNY"]
    return "/".join(values) if values else "未知"


def local_time(account):
    value = account.get("time_utc")
    return datetime.fromisoformat(value).astimezone(timezone(timedelta(hours=8))).strftime("%m-%d %H:%M:%S") if value else "未知"


def main():
    result = read(OUT / "RESULT.json")
    authorization = read(OUT / "authorization.json")
    before = read(OUT / "limits-before.json")
    after = read(OUT / "limits-after.json")
    lines = ["# MyMind阶段8预算与成本", "", "阶段8沿用唯一正式累计账本`runs/stage7-preparation-20261014/formal/limits.json`。完整授权前态在limits-before.json，本次追加上限和来源在authorization.json，结束态在limits-after.json。原consumed及全部events保留，失败不退还，旧余量不构成本次额外授权。", "",
             "| 资源 | 原已耗 | 本次新增授权 | 本次已耗 | 本次未用 | 累计已耗 | 累计余量 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, limit in authorization["additional_limits"].items():
        added = result["stage8_consumed"][key]
        lines.append(f"| {key} | {before['consumed'].get(key,0)} | {limit} | {added} | {limit-added} | {after['consumed'].get(key,0)} | {after['limits'][key]-after['consumed'].get(key,0)} |")
    lines += ["", "GPU按一张物理卡乘命令硬时限预留，不按实际运行较短退还。API预留按每条轨迹12次保守额度，与实际HTTP计数分列。SFT和GRPO本轮均没有参数更新。", "",
              "| 命令 | 硬上限秒 | 实际命令秒 | 退出 | Embedding HTTP | DeepSeek HTTP | 已知DeepSeek usage价元 | 前余额元 | 后余额元 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    wall = 0
    ordered = [f"test-{g}" for g in LABELS] + [f"regression-{g}" for g in LABELS] + [f"demo-{g}" for g in ("sft", "rl")]
    for name in ordered:
        if name not in result["commands"]:
            continue
        command = result["commands"][name]
        if name.startswith("demo-"):
            usage = command.get("usage", {})
        else:
            scope, group = name.split("-", 1)
            mapping = result["final_test" if scope == "test" else "basic_regression"]
            usage = mapping[group]["all_task_cost"]["usage"]
        wall += command["elapsed_s"]
        price = usage.get("known_deepseek_usage_price_cny")
        price_text = f"{price:.8f}" if price is not None else "未知"
        lines.append(f"| {name} | {command['hard_timeout_s']} | {command['elapsed_s']:.3f} | {command['exit_code']} | {usage.get('embedding_requests',0)} | {usage.get('rerank_http_requests',0)} | {price_text} | {amount(command['balance_before'])} | {amount(command['balance_after'])} |")
    total = result["stage8_usage"]
    price = total["known_deepseek_usage_price_cny"]
    lines += ["", f"本轮实际分配命令墙钟合计{wall:.3f}秒；它包含加载和退出，不是GPU利用率积分。实际Embedding {total.get('embedding_requests',0)}次、DeepSeek HTTP {total.get('rerank_http_requests',0)}次，返回Embedding用量事件{total.get('embedding_usage_events',0)}、DeepSeek用量事件{total.get('rerank_usage_events',0)}。新缺usage及其他不明项见RESULT.stage8_usage.unknown。", "",
              f"按保存的官方人民币单价及各HTTP开始时间区分峰谷，本轮已知DeepSeek usage价{price:.8f}元；Embedding输入{total.get('embedding_input_tokens',0)} token，按0.125元/百万输入token估算{total['embedding_usage_estimate_cny']:.9f}元。它们是用量计价参考，精确归属账单、Embedding实际账单/余额和GPU货币费用仍未知，不记0。", "",
              "官方价来源：[DeepSeek模型与价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。本轮浏览器核对Flash空闲时段每百万缓存命中输入0.02元、未命中输入1元、输出4元；北京时间工作日9—12及14—18时价格双倍。Embedding估算沿用阶段7保存的官方价格。允许峰时，10元非硬限。", "",
              f"跨日记账保留阶段7末已知DeepSeek参考1.50577936元，加本轮后为{1.50577936+price:.8f}元。原历史12次失败HTTP的usage/费用未知继续保留；不把这个参考值当完整实际账单，不与账户累计消费重复相加。账户页面可得观察和延迟说明见billing-observations.json，逐命令余额原响应见commands/。", "",
              "## 最终测试全部任务与成功子集", "", "| 配置与子集 | 任务数 | 请求搜索 | Embedding HTTP | DeepSeek HTTP | 任务耗时秒 | 生成秒 | 已知DeepSeek usage价元 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group, label in LABELS.items():
        for key, subset in (("all_task_cost", "全部"), ("successful_task_cost", "成功子集")):
            cost = result["final_test"][group][key]
            u = cost["usage"]
            lines.append(f"| {label}/{subset} | {cost['tasks']} | {cost['search_requests']} | {u.get('embedding_requests',0)} | {u.get('rerank_http_requests',0)} | {cost['elapsed_s']:.3f} | {cost['gpu_generation_s']:.3f} | {u['known_deepseek_usage_price_cny']:.8f} |")
    lines += ["", "成功子集成本只累加成功任务本身开销；所有尝试开销除成功数另见RESULT各组all_cost_per_success，无成功时未定义。任务耗时不含命令加载，生成秒来自保存记录。基本回归和新演示计入本轮总账，未混入120题成绩或其成本子表。", "",
              "共享GPU负载、峰谷与缓存命中均会影响耗时或货币价，不能把组间金额差全部归因于策略。一次检索若因超额提前协议失败，低开销不构成效率或可靠性改善的证据。所有原故障、首次低分及未明费用保留。", "",
              "## 逐命令余额时间", "", "| 命令 | 前查询北京时间 | 后查询北京时间 |", "| --- | --- | --- |"]
    for name in ordered:
        if name in result["commands"]:
            c = result["commands"][name]
            lines.append(f"| {name} | {local_time(c['balance_before'])} | {local_time(c['balance_after'])} |")
    (OUT / "COSTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
