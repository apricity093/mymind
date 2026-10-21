"""对照历史 A 与本次 B 的首次开发验证，保留双向变化与成本限制。"""
from collections import Counter
import ast
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import statistics
import textwrap

SOURCE=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get("MYMIND_WORK_ROOT",SOURCE))
RUNS=ROOT/("runs" if os.environ.get("MYMIND_WORK_ROOT") else "agentic_runs")
RUN=RUNS/"grpo-standard-20261021"
A=RUNS/"stage7-preparation-20261014/formal"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()]


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def disposition(row):
    try:
        if isinstance(row["raw_result"],dict):
            return row["raw_result"].get("disposition")
        text=row["raw_result"].strip()
        if text.startswith("```"):
            text="\n".join(text.splitlines()[1:-1])
        return json.loads(text).get("disposition")
    except (ValueError,KeyError,AttributeError):
        return None


def evaluation(directory):
    metrics=read(directory/"metrics.json")
    records=rows(directory/"first_records.jsonl")
    standards={r["task_id"]:r for r in read(directory/"inputs/private_standards.v5.json")}
    result={key:metrics[key] for key in ("combination_equal_success_rate","instance_success_rate",
        "combination_equal_severe_error_rate","protocol_failures","infrastructure_failures","all_tasks_finished")}
    result.update(complete_success=sum(r["grade"]["complete_success"] for r in records),
        severe_errors=sum(r["grade"]["severe_error"] for r in records),
        primary_disposition_correct=sum(disposition(r) in standards[r["task_id"]]["dispositions"] for r in records),
        tasks=len(records),requested_searches=sum(r.get("search_requests",0) for r in records),
        actual_searches=sum(len(r["trajectory"]["searches"]) for r in records))
    return result,{r["task_id"]:r for r in records}


def paired(left,right):
    if set(left)!=set(right):
        raise ValueError("对照任务集合不一致")
    success=lambda data:{tid for tid,r in data.items() if r["grade"]["complete_success"]}
    a,b=success(left),success(right)
    return dict(gained=sorted(b-a),lost=sorted(a-b),shared_success=sorted(a&b),
        details=[dict(task_id=tid,a_grade=left[tid]["grade"],b_grade=right[tid]["grade"],
            a_disposition=disposition(left[tid]),b_disposition=disposition(right[tid])) for tid in sorted(a^b)])


def learning(directory):
    groups=[read(p) for p in directory.glob("group-step*-rank0.json")]
    training=[g for g in groups if g["policy_step"]<80]
    accepted=[g for g in training if not g["infrastructure_failure"]]
    updates=rows(directory/"updates-rank0.jsonl")
    all_updates=[r for p in directory.glob("updates-rank*.jsonl") for r in rows(p)]
    losses=rows(directory/"loss-rank0.jsonl")
    rewards=[r for g in accepted for r in g["rewards"]]
    state=read(directory/"trainer_state.json")
    logs=[r for r in state["log_history"] if "entropy" in r]
    avg=lambda values:statistics.mean(values) if values else None
    result=dict(optimizer_steps=len(updates),training_groups=len(accepted),training_trajectories=len(rewards),
        mean_reward=avg(rewards),reward_distribution=dict(Counter(map(str,rewards))),
        group_reward_std_mean=avg([g.get("reward_std",statistics.stdev(g["rewards"])) for g in accepted]),
        zero_advantage_group_ratio=avg([not g["nonzero_advantage"] for g in accepted]),
        zero_advantage_trajectory_ratio=avg([abs(a)<1e-8 for g in accepted for a in g["advantages"]]),
        nonzero_gradient_steps=sum(r["gradient_norm"]>0 for r in updates),
        parameter_changed_steps=sum(r["parameter_delta_l1"]>0 for r in updates),
        update_elapsed_s=sum(r["elapsed_step_s"] for r in updates),
        peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in all_updates),
        peak_reserved_bytes=max(r["peak_reserved_bytes"] for r in all_updates),
        model_generated_tokens=sum(g["effective_model_tokens"] for g in accepted),
        mean_model_generated_trajectory_tokens=avg([g["effective_model_tokens"]/4 for g in accepted]),
        actual_searches=sum(r["actual_search_count"] for g in accepted for r in g["rows"]),
        requested_searches=sum(r["search_requests"] for g in accepted for r in g["rows"]),
        metrics_step_means={key:avg([r[key] for r in logs if key in r]) for key in
            ("kl","entropy","clip_ratio/low_mean","clip_ratio/high_mean","clip_ratio/region_mean","completions/mean_length")},
        loss_components_step_means={key:avg([r[key] for r in losses if key in r]) for key in
            ("reward_loss","kl_loss","masked_kl","reward_logp_gradient_l1","kl_logp_gradient_l1")})
    reference=[r for p in directory.glob("reference-forward-rank*.jsonl") for r in rows(p)]
    result["reference_forward_rank_seconds"]=sum(r["reference_forward_s"] for r in reference)
    result["training_reference_forward_rank_seconds"]=sum(r["reference_forward_s"] for r in reference if r["policy_step"]<80)
    result["post_update_reference_forward_rank_seconds"]=sum(r["reference_forward_s"] for r in reference if r["policy_step"]==80)
    result["reference_forward_measurement_available"]=bool(list(directory.glob("reference-forward-rank*.jsonl")))
    result["zero_advantage_steps_with_kl_gradient"]=sum(r.get("reward_logp_gradient_l1") == 0 and r.get("kl_logp_gradient_l1",0)>0 for r in losses)
    result["api"]=api_usage([r for g in groups for r in g["rows"]],directory)
    return result


def api_usage(records,directory):
    counter=Counter()
    models=Counter()
    dated_price=0.
    dated_events=0
    for row in records:
        diagnostics=row.get("tool_diagnostics",[])
        request_times=row.get("cost",{}).get("rerank_request_started_unix_s",[])
        periods=[]
        for stamp in request_times:
            instant=datetime.fromtimestamp(stamp,timezone(timedelta(hours=8)))
            periods.append(2 if instant.weekday()<5 and (9<=instant.hour<12 or 14<=instant.hour<18) else 1)
        for index,diagnostic in enumerate(diagnostics):
            for embedding in diagnostic.get("embedding_usage") or []:
                counter["embedding_usage_tokens"]+=embedding.get("reported_tokens") or 0
            usage=diagnostic.get("diagnostics",{}).get("rerank_usage")
            model=diagnostic.get("diagnostics",{}).get("rerank_response_model")
            if model:
                models[model]+=1
            if not usage:
                continue
            counter["rerank_usage_events"]+=1
            for source,target in (("input_tokens","rerank_input_tokens"),("output_tokens","rerank_output_tokens"),
                                  ("cache_read_tokens","cache_hit_tokens"),("cache_miss_tokens","cache_miss_tokens")):
                counter[target]+=usage.get(source) or 0
            multiplier=(periods[index] if len(periods)==len(diagnostics) else
                        periods[0] if periods and len(set(periods))==1 else None)
            if multiplier is not None and usage.get("cache_read_tokens") is not None and usage.get("cache_miss_tokens") is not None:
                dated_price+=multiplier*(usage["cache_read_tokens"]*.02+usage["cache_miss_tokens"]+usage["output_tokens"]*4)/1e6
                dated_events+=1
        for key in ("embedding_requests","rerank_http_requests"):
            counter[key]+=row.get("cost",{}).get(key,0)
    # 训练rank汇总含首次失败及更新后采样，也含诊断未返回usage的请求。
    costs=[read(p) for p in directory.glob("api-cost-rank*.json")]
    if costs:
        counter["embedding_requests"]=sum(r["embedding_requests"] for r in costs)
        counter["rerank_http_requests"]=sum(r["rerank_http_requests"] for r in costs)
        counter["embedding_usage_tokens"]=sum(r["embedding_reported_tokens"] for r in costs)
    lower=(counter["cache_hit_tokens"]*.02+counter["cache_miss_tokens"]+counter["rerank_output_tokens"]*4)/1e6
    result={**dict(counter),"response_model_counts":dict(models),
        "deepseek_usage_estimate_cny_bounds":[lower,2*lower],
        "embedding_list_price_estimate_cny":counter["embedding_usage_tokens"]*.125/1e6,
        "http_without_usage":counter["rerank_http_requests"]-counter["rerank_usage_events"],
        "usage_events_with_request_time":dated_events,
        "deepseek_usage_estimate_cny":dated_price if dated_events and dated_events==counter["rerank_usage_events"] else None,
        "actual_attributable_bill_cny":None}
    execution=directory/"execution.json"
    if execution.exists():
        command=read(execution)
        if command.get("started_utc") and command.get("finished_utc"):
            cst=timezone(timedelta(hours=8))
            start=datetime.fromisoformat(command["started_utc"]).astimezone(cst)
            finish=datetime.fromisoformat(command["finished_utc"]).astimezone(cst)
            fixed=None
            if start.date()==finish.date():
                begin=start.hour+start.minute/60+start.second/3600
                end=finish.hour+finish.minute/60+finish.second/3600
                if start.weekday()>=5:
                    fixed=1
                else:
                    for lo,hi,multiplier in ((0,9,1),(9,12,2),(12,14,1),(14,18,2),(18,24,1)):
                        if lo<=begin<=end<hi:
                            fixed=multiplier
            result["single_tariff_interval_multiplier"]=fixed
            if result["deepseek_usage_estimate_cny"] is None and fixed is not None:
                result["deepseek_usage_estimate_cny"]=lower*fixed
            result["observed_account_balance_decrease_cny"]=(balance_amount(command,"balance_before")-balance_amount(command,"balance_after")
                if balance_amount(command,"balance_before") is not None and balance_amount(command,"balance_after") is not None else None)
    return result


def balance_amount(record,key):
    balances=record.get(key,{}).get("response",{}).get("balance_infos",[])
    values=[float(r["total_balance"]) for r in balances if r.get("currency")=="CNY"]
    return sum(values) if values else None


def render_report(result):
    nodes=result["nodes"]
    sa,sb=result["selected_a"],result["selected_b"]
    a,b=nodes[str(sa)]["a"],nodes[str(sb)]["b"]
    pair=result["selected_pair"]
    lines=["# GRPO式配置对照结果", "", "2026-10-21；状态：已完成。", "",
        f"B在20/40/80步的组合等权完整成功率均低于历史A的同一步数。按原规则分别选中A{sa}与B{sb}：完整成功{a['complete_success']}/80与{b['complete_success']}/80，B新增{len(pair['gained'])}题、失去{len(pair['lost'])}题，净{b['complete_success']-a['complete_success']:+d}题。本次没有支持B更好的结果。原正式SFT120→RL40与展示指针保持，不追加训练或120题测试。", "",
        "A使用保留的正式历史结果；B从原正式SFT120重新加载并建立新优化器，参考策略为固定SFT120。固定基座revision、158共同池、划分、v13/v5、公开输入、索引、重排配置、工具与token预算。四项实际目标差异为loss_type、scale_rewards、epsilon_high和beta，不能将整体差异归于单项。", "",
        "**实质环境差异**：DeepSeek于10月17日将同名`deepseek-flash`服务升级为V4.1-Flash-1017，晚于历史A运行。配置名相同不能保证底层重排器相同，检索观察与顺序可能受影响。本次是含该差异的历史配置比较，不是仅改变训练配置的严格受控实验；没有为消除差异重训A。[官方更新记录](https://api-docs.deepseek.com/updates/)。", "",
        "## 逐节点开发验证", "",
        "每个节点80题、每题一次贪心首次轨迹；基础设施故障均为0。组合等权率为七个组合成功率的平均，实例率以80题为分母。严重错误与协议失败可重叠；主要处置正确不等于完整成功。", "",
        "| 步数 | A组合等权完整 | B组合等权完整 | A实例完整 | B实例完整 | A/B严重 | A/B协议 | A/B主要处置 | B新增/失去成功 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for step,node in nodes.items():
        left,right=node["a"],node["b"]
        lines.append(f"| {step} | {left['combination_equal_success_rate']:.2%} | {right['combination_equal_success_rate']:.2%} | {left['complete_success']}/80（{left['instance_success_rate']:.2%}） | {right['complete_success']}/80（{right['instance_success_rate']:.2%}） | {left['severe_errors']}/{right['severe_errors']} | {left['protocol_failures']}/{right['protocol_failures']} | {left['primary_disposition_correct']}/{right['primary_disposition_correct']} | {len(node['paired']['gained'])}/{len(node['paired']['lost'])} |")
    lines += ["", "选模规则：组合等权完整成功率最高；完全同分时组合等权严重错误率更低；再同分取更早步数。B只评测20/40/80节点，Trainer例行保存的60步未参与选模。", "",
        f"各自选模比较：A{sa}组合等权{a['combination_equal_success_rate']:.2%}、实例{a['instance_success_rate']:.2%}；B{sb}为{b['combination_equal_success_rate']:.2%}、{b['instance_success_rate']:.2%}，差值分别{100*(b['combination_equal_success_rate']-a['combination_equal_success_rate']):+.2f}与{100*(b['instance_success_rate']-a['instance_success_rate']):+.2f}个百分点。严重错误{a['severe_errors']}→{b['severe_errors']}，协议失败{a['protocol_failures']}→{b['protocol_failures']}，主要处置正确{a['primary_disposition_correct']}→{b['primary_disposition_correct']}。共同成功{len(pair['shared_success'])}题；所有同题变化与错误明细见[RESULT.json](RESULT.json)。", "",
        "## 代表性双向变化", "", f"以下比较各自选中的A{sa}/B{sb}；完整原输出、观察和grade见[CASES.json](CASES.json)。", "",
        "| 题目 | A→B奖励 | 原始输出支持的变化 |", "| --- | ---: | --- |",
        "| M6-D04-008 | −1→1 | 在线支付已过4工作日、上限3日；A仍写within_window=true并答复等待，B改为超时核验和升级。 |",
        "| M6-F03-004 | −1→1 | A把未提供的occurrence写为continuous，B不再将其写成已确认事实。 |",
        "| M6-D04-011 | 1→−1 | 已过10工作日、上限3日；B错误写within_window=true，整体升级处置正确仍无法通过完整任务。 |",
        "| M6-R07-001 | 1→0.3 | 两者都补信息，B遗漏必要的elapsed_workdays=unknown事实。 |",
        "| M6-P07-006 | 1→−1 | B的outbound_ip_allowed事实与用户材料/适用规则矛盾。 |",
        "| M6-R07-008 | 1→−1 | B将未提供的payment_affected写成已确认。 |", "",
        "这些案例分别说明改善与退化，不能据此确认某个参数、奖励归一化或KL是原因。", "",
        "## 训练诊断与资源", "", "两臂均为80个optimizer步骤、40组、160条在线训练轨迹；下表训练统计不混入固定8条后采样。", "",
        "| 项目 | A历史配置 | B本次配置 |", "| --- | ---: | ---: |"]
    left,right=result["training"]["a"],result["training"]["b"]
    diagnostics=[("平均奖励",f"{left['mean_reward']:.4f}",f"{right['mean_reward']:.4f}"),
        ("组内样本标准差均值",f"{left['group_reward_std_mean']:.6f}",f"{right['group_reward_std_mean']:.6f}"),
        ("零优势组比例",f"{left['zero_advantage_group_ratio']:.2%}",f"{right['zero_advantage_group_ratio']:.2%}"),
        ("零优势轨迹比例",f"{left['zero_advantage_trajectory_ratio']:.2%}",f"{right['zero_advantage_trajectory_ratio']:.2%}"),
        ("KL有效token均值的步骤平均","关闭，未测距",f"{right['metrics_step_means']['kl']:.8f}"),
        ("entropy步骤均值",f"{left['metrics_step_means']['entropy']:.8f}",f"{right['metrics_step_means']['entropy']:.8f}"),
        ("clip region步骤均值",f"{left['metrics_step_means']['clip_ratio/region_mean']:.5%}",f"{right['metrics_step_means']['clip_ratio/region_mean']:.5%}"),
        ("平均有效模型输出token/轨迹",f"{left['mean_model_generated_trajectory_tokens']:.2f}",f"{right['mean_model_generated_trajectory_tokens']:.2f}"),
        ("有效模型输出token合计",str(left['model_generated_tokens']),str(right['model_generated_tokens'])),
        ("实际搜索调用",str(left['actual_searches']),str(right['actual_searches'])),
        ("非零有限总梯度步骤",f"{left['nonzero_gradient_steps']}/80",f"{right['nonzero_gradient_steps']}/80"),
        ("参数实际变化步骤",f"{left['parameter_changed_steps']}/80",f"{right['parameter_changed_steps']}/80"),
        ("训练命令耗时，含装载/后采样",f"{left['command_elapsed_s']:.2f}秒",f"{right['command_elapsed_s']:.2f}秒"),
        ("每卡峰值CUDA分配，跨rank取最大",f"{left['peak_allocated_bytes']/2**30:.2f} GiB",f"{right['peak_allocated_bytes']/2**30:.2f} GiB"),
        ("每卡峰值CUDA预留，跨rank取最大",f"{left['peak_reserved_bytes']/2**30:.2f} GiB",f"{right['peak_reserved_bytes']/2**30:.2f} GiB"),
        ("Embedding/重排HTTP，含后采样",f"{left['api']['embedding_requests']}/{left['api']['rerank_http_requests']}",f"{right['api']['embedding_requests']}/{right['api']['rerank_http_requests']}")]
    lines += [f"| {name} | {x} | {y} |" for name,x,y in diagnostics]
    lines += ["",f"B新增固定参考策略前向：训练{right['training_reference_forward_rank_seconds']:.2f}卡秒、后采样{right['post_update_reference_forward_rank_seconds']:.2f}卡秒，总计{right['reference_forward_rank_seconds']:.2f}卡秒。双rank并行，卡秒求和不等于额外墙钟；KL反向及其他新增检查开销未单独分离。两臂运行时点和服务/GPU负载未受控，不能把命令耗时差异归因于某个参数，也不以不同目标的训练loss大小判断质量。", "",
        f"B的rank0有{right['zero_advantage_steps_with_kl_gradient']}个微批次奖励项logp导数为0、KL项导数非0。A的零总梯度步骤仍可有Adam动量更新。非零梯度、参数变化与新的奖励区分信号分开解释；逐rank奖励项/KL项/归一化分母及排除token导数见train/loss-rank*.jsonl，总梯度和参数变化见updates-rank*.jsonl。", "",
        "## 有限预算与API费用", "", "正常轨迹8预检+160训练+240验证+8后采样=416；基础设施整组补采0/16。自然预检4步、正式80步，另单列1个受控KL回放optimizer诊断步骤，不新增轨迹/API。", "",
        "| 阶段 | 轨迹 | 墙钟秒 | Embedding请求 | 重排HTTP | 已返回usage的DeepSeek刊例估算/元 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    phases=[("预检","preflight",8,result["preflight"]["api"]),("训练及后采样","train",168,right["api"])]
    phases += [(f"验证{s}",f"validation-{s}",80,nodes[str(s)]["b"]["api"]) for s in (20,40,80)]
    for title,name,count,cost in phases:
        money=cost.get("deepseek_usage_estimate_cny")
        price=f"{money:.8f}" if money is not None else "未知"
        lines.append(f"| {title} | {count} | {result['commands'][name]['elapsed_s']:.2f} | {cost['embedding_requests']} | {cost['rerank_http_requests']} | {price} |")
    cost=result["b_total_api"]
    lines += ["",f"实际Embedding {cost['embedding_requests']}次、重排HTTP {cost['rerank_http_requests']}次，另余额查询{cost['control_api_requests']}/12次；检索API共{cost['embedding_requests']+cost['rerank_http_requests']}次，小于预留上限5,184次。独立账本正常轨迹预留4,992次API，未用基础设施余量不构成后续调用授权。", "",
        f"已返回usage的DeepSeek刊例估算{cost['deepseek_usage_estimate_cny']:.8f}元；Embedding输入{cost['embedding_usage_tokens']} token，刊例估算{cost['embedding_list_price_estimate_cny']:.9f}元。有{cost['http_without_usage']}次重排HTTP尝试未返回usage，其费用未知，不记0；它在原重排重试规则内成功，未触发轨迹补采。", "",
        f"收尾余额快照显示DeepSeek账户由{result['initial_account_balance_cny']:.2f}降至{result['final_account_balance_cny']:.2f}元，观察到减少{result['settled_account_balance_decrease_cny']:.2f}元。逐命令即时余额差合计{cost['observed_account_balance_decrease_cny']:.2f}元，存在记账时滞，不能作为精确归属账单。Embedding实际账单/免费额度、精确归属账单、GPU货币费用仍未知。价格与来源见[pricing.json](pricing.json)，原余额和usage随命令与轨迹保留。", "",
        f"GPU预留54,000卡秒（15卡时），按实际命令墙钟×卡数计算{result['execution_card_seconds']:.2f}卡秒（{result['execution_card_seconds']/3600:.3f}卡时，含受控回放），不是GPU利用率积分。正式训练、三个验证及受控回放均退出0。自然预检模型子进程退出0，首次验收因没有实际参数变化退出；单列KL回放后最终验收通过，原首次日志保留。超时、argv、物理GPU、用量和余额见commands/，每10秒GPU采样见对应CSV。", "",
        "## 实现与验收", "",
        "实际依赖为torch 2.7.1+cu118、TRL 1.10.0、Transformers 5.2.0、PEFT 0.18.0、Accelerate 1.15.0。保存的历史TRL损失函数与当前函数AST一致；其他源码差异和固定输入逐字节核验见environment.json，实际运行源码随各阶段source/保留。", "",
        "group优势按跨两rank完整同题四轨迹计算，样本标准差分母n−1、epsilon=1e−4。GRPO每轨迹按completion_mask×tool_mask的有效模型token求均值，再平均样本；DAPO保留原全生成批次有效token分母。prompt、padding、工具观察不参与直接策略/KL损失或长度分母。", "",
        "KL为exp(logp_ref−logp_policy)−(logp_ref−logp_policy)−1，use_bias_correction_kl=true再乘token概率比。importance_sampling_level=token、sync_ref_model=false、entropy_coef=0、multi_objective_aggregation=sum_then_normalize、delta=None、top_entropy_quantile=1；logp和entropy使用temperature=.8缩放后的logits，top_p=.95用于采样。参考为原生PEFT ref adapter，504张量初始等于真实SFT120，训练后逐张量不变，不进入优化器、无参考梯度，不新增第二套基座。", "",
        "本地57项相关回归及追加5项检查通过，两个正式保留验证器通过；服务器安装版TRL的CPU损失核验涵盖GRPO/DAPO、工具mask和零优势KL导数。八条GPU自然预检全部奖励1、优势和KL均0，因此四步无自然参数变化，未重采。另在原预检时间/GPU预留内对已保存轨迹做受控KL回放：单张LoRA B加.01、奖励与优势不变，新轨迹/API均0；KL梯度范数.00591546，optimizer参数变化L1为87.28513，参考仍冻结。该结果是更新能力验证，不能称自然奖励学习。正式B重新加载原SFT120、重置seed42与优化器，不继承扰动或预检状态。", "",
        "预检与原始首次验收退出记录完整保留；最终验收见preflight-acceptance.json、kernel-verification.json、kl-replay/verification.json及train/reference-*.json。权重、optimizer与RNG保留服务器；本地保存配置、实际源码、命令、逐题结果、训练日志和选模。", "",
        "## 结论边界", "", *["- "+text for text in result["limitations"]], "",
        "这次未观察到B优于历史A；不能推出原始GRPO普遍更差，也不能推出beta、标准差缩放、clip或归一化中任何单项造成了差值。"]
    return "\n".join(lines)+"\n"


def main():
    nodes={}
    records={}
    for step in (20,40,80):
        left,l=evaluation(A/f"grpo-validation-{step}")
        right,r=evaluation(RUN/f"validation-{step}")
        nodes[str(step)]=dict(a=left,b=right,paired=paired(l,r))
        nodes[str(step)]["a"]["api"]=api_usage(list(l.values()),A/f"grpo-validation-{step}")
        nodes[str(step)]["b"]["api"]=api_usage(list(r.values()),RUN/f"validation-{step}")
        records[("a",step)]=l
        records[("b",step)]=r
    selected_a=read(A/"grpo-selection.json")["selected_step"]
    selected_b=read(RUN/"selection.json")["selected_step"]
    for arm,selected in (("a",selected_a),("b",selected_b)):
        expected=min((20,40,80),key=lambda step:(-nodes[str(step)][arm]["combination_equal_success_rate"],
            nodes[str(step)][arm]["combination_equal_severe_error_rate"],step))
        if selected!=expected:
            raise ValueError("选模记录与原预定规则不一致")
    commands={p.stem:read(p) for p in (RUN/"commands").glob("*.json")}
    for command in commands.values():
        before,after=balance_amount(command,"balance_before"),balance_amount(command,"balance_after")
        command["observed_account_balance_decrease_cny"]=(before-after if before is not None and after is not None else None)
    result=dict(name="GRPO式配置对照",nodes=nodes,selected_a=selected_a,selected_b=selected_b,
        selected_pair=paired(records[("a",selected_a)],records[("b",selected_b)]),
        training=dict(a=learning(A/"grpo"),b=learning(RUN/"train")),
        commands=commands,limits=read(RUN/"limits.json"),environment=read(RUN/"environment.json"),
        service_comparison=read(RUN/"service-comparison.json"),pricing=read(RUN/"pricing.json"),
        limitations=["单seed=42", "使用既有80题开发集和历史A参照，不是全新盲测", "多项配置同时变化，不推断单参数因果",
            "同轨迹/optimizer步数不等于同计算成本；B新增固定参考策略前向", "训练loss不跨目标比较质量",
            "未新增120题测试，未替换原正式SFT120→RL40展示指针",
            "DeepSeek同名重排服务于10月17日更新；历史A与当前B底层重排器版本存在实质差异"])
    result["preflight"]=learning(RUN/"preflight")
    result["effective_probability_semantics"]=dict(temperature=.8,top_p=.95,
        logp="installed TRL divides logits by sampling temperature before log-softmax",
        entropy="same temperature-scaled logits",top_p_scope="generation sampling")
    result["controlled_kl_replay"]=read(RUN/"kl-replay/verification.json")
    costs=[result["preflight"]["api"],result["training"]["b"]["api"]]+[n["b"]["api"] for n in nodes.values()]
    result["b_total_api"]={key:sum(c.get(key,0) for c in costs) for key in
        ("embedding_requests","rerank_http_requests","embedding_usage_tokens","rerank_input_tokens","rerank_output_tokens",
         "embedding_list_price_estimate_cny","http_without_usage")}
    result["b_total_api"].update(deepseek_usage_estimate_cny_bounds=[sum(c["deepseek_usage_estimate_cny_bounds"][i] for c in costs) for i in (0,1)],
        deepseek_usage_estimate_cny=sum(c["deepseek_usage_estimate_cny"] for c in costs) if all(c.get("deepseek_usage_estimate_cny") is not None for c in costs) else None,
        observed_account_balance_decrease_cny=sum(c["observed_account_balance_decrease_cny"] for c in commands.values()
            if c["observed_account_balance_decrease_cny"] is not None),actual_attributable_bill_cny=None,
        control_api_requests=len(read(RUN/"control-api.json")))
    control=read(RUN/"control-api.json")
    result["initial_account_balance_cny"]=balance_amount({"balance":control[0]},"balance")
    result["final_balance_snapshot"]=read(RUN/"final-balance.json")
    result["final_account_balance_cny"]=balance_amount({"balance":result["final_balance_snapshot"]},"balance")
    result["settled_account_balance_decrease_cny"]=result["initial_account_balance_cny"]-result["final_account_balance_cny"]
    result["execution_card_seconds"]=sum(c["elapsed_s"]*len(c["physical_gpus"]) for c in commands.values())
    post=[read(p) for p in (RUN/"train").glob("group-step80-*-rank0.json")]
    result["actual_trajectories"]=dict(preflight=result["preflight"]["training_trajectories"],
        training=result["training"]["b"]["training_trajectories"],validation=sum(n["b"]["tasks"] for n in nodes.values()),
        post_update=sum(len(g["rows"]) for g in post if not g["infrastructure_failure"]),
        infrastructure_resamples=result["limits"]["consumed"].get("infrastructure_trajectories",0),
        controlled_kl_replay=0)
    assert result["training"]["b"]["optimizer_steps"]==80
    assert result["training"]["b"]["training_trajectories"]==160
    assert all(n["b"]["tasks"]==80 and n["b"]["all_tasks_finished"] for n in nodes.values())
    assert result["actual_trajectories"]["preflight"]==8 and result["actual_trajectories"]["post_update"]==8
    assert result["limits"]["consumed"]["student_trajectories"]==416
    assert result["b_total_api"]["embedding_requests"]+result["b_total_api"]["rerank_http_requests"]<=result["limits"]["limits"]["api_request_allowance"]
    result["training"]["a"]["command_elapsed_s"]=int((A.parent/"formal-grpo-elapsed-seconds.txt").read_text().strip())
    result["training"]["b"]["command_elapsed_s"]=commands["train"]["elapsed_s"]
    historical=(RUNS/"stage8-20261016/installed-trl-compute-loss.txt").read_text(encoding="utf-8")
    installed=(RUN/"implementation/installed_grpo_trainer.py").read_text(encoding="utf-8")
    current=next(n for n in ast.walk(ast.parse(installed)) if isinstance(n,ast.FunctionDef) and n.name=="_compute_loss")
    saved=ast.parse(textwrap.dedent(historical)).body[0]
    result["historical_installed_trl_loss_ast_equal"]=ast.dump(current)==ast.dump(saved)
    selected_pair=result["selected_pair"]
    cases=[]
    # 取不同任务类型的真实双向变化，避免用两条相似D04题代表全部改善。
    representatives=[("gained","M6-D04-008"),("gained","M6-F03-004"),
        ("lost","M6-D04-011"),("lost","M6-R07-001"),("lost","M6-P07-006"),("lost","M6-R07-008")]
    for kind,tid in representatives:
        if tid not in selected_pair[kind]:
            raise ValueError("代表案例与实际配对变化不一致")
        left=records[("a",selected_a)][tid]
        right=records[("b",selected_b)][tid]
        cases.append(dict(kind=kind,task_id=tid,a_step=selected_a,b_step=selected_b,
            a_grade=left["grade"],b_grade=right["grade"],a_result=left["raw_result"],b_result=right["raw_result"],
            a_trajectory=left["trajectory"],b_trajectory=right["trajectory"]))
    write(RUN/"CASES.json",cases)
    write(RUN/"RESULT.json",result)
    (RUN/"RESULTS.md").write_text(render_report(result),encoding="utf-8")
    print(json.dumps({"nodes":nodes,"selected_a":selected_a,"selected_b":selected_b},ensure_ascii=False))


if __name__=="__main__":
    main()
