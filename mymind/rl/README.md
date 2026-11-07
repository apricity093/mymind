# Agentic RL 训练与评测

本模块实现给定账单或技术角色后的共享客服策略训练。应用层主辅 Agent 编排与策略训练分别运行。模块职责、独立服务启动和公开资源边界见[策略架构与运行](../docs/agentic_rl_architecture.md)，正式指标与选模见下文；本地结果页的生成方式见[离线分析说明](../benchmarks/README.md)。

## 实现入口

| 入口 | 用途 |
| --- | --- |
| `prepare.py`、`download_model.py` | 政策索引准备与固定基座下载 |
| `teacher.py`、`sft_data.py` | 教师候选采集、筛选与 assistant-only token 导出 |
| `sft.py` | LoRA SFT，工具观察 labels 为 -100 |
| `grpo.py`、`training.py` | 在线工具轨迹、同题四轨迹分组、奖励与优化 |
| `grpo_diagnostics.py` | 优势、有效 token 分母、参考策略与更新诊断 |
| `grading.py`、`tasks.py` | 私有标准隔离、可见证据支持和业务评分 |
| `formal_evaluate.py`、`metrics.py`、`run_control.py` | 冻结评测、实例/组合等权指标、额度、选模与故障恢复 |

教师采集、索引准备、训练及评测入口通过 `--config` 读取配置；辅助脚本的参数以各自 `--help` 为准。训练与评测使用 `python -m rl.sft`、`python -m rl.grpo` 和 `python -m rl.evaluate`，其中 `formal_evaluate.py` 由评测入口调用。资源与输出相对 `MYMIND_WORK_ROOT`。开发配置模板位于 `configs/`，开发任务、标准和政策支持位于 `resources/`；使用模板前，将其中三个 JSON 文件复制到工作根目录的 `data/agentic_rl/dev/`。正式训练还需要原始冻结配置、正式数据集、v5 标准、示范及 token 导出；正式和校准数据、各版本评分标准及来源记录均不随仓库公开。

## 奖励与训练

评分区分完整成功、严重错误、协议错误和缺项，奖励为 1/0.3/0/-1。事实与处置依据必须由实际可见且被引用的正文支持；同来源未展示内容不提供支持。用户、system、工具观察和 padding 不参与直接 loss。

正式 SFT 使用 158 条教师示范，训练 120 步；在 40/80/120 步各做 80 题开发验证后选择 SFT120。RL 从所选 SFT 加载 LoRA、新建优化器，完成 80 步，在 20/40/80 步各做开发验证后选择 RL40。选模规则为组合等权完整成功最高、严重错误更低、步数更早，未用最终测试换 checkpoint。

正式 RL 使用 DAPO 有效模型 token 归一化、未按组标准差缩放的优势及 β=0。训练器也支持 GRPO 逐轨迹归一化、跨 rank 完整四轨迹的组标准差和固定 SFT120 参考 adapter；两条目标路径分别记录。

## 结果与限制

正式 120 题 Base/SFT/RL 完整成功为 0/43/43，严重错误为 38/38/27，协议失败为 76/24/19。RL 新增 3 道成功也失去 3 道，D09/F04 各 0/15。策略已实现工具观察后的反馈决策；正式测试未观察到补检，不能据此确认该能力的效果。

原始轨迹、教师来源、监督 mask、实际梯度与 LoRA 更新、选模节点、首次故障及独立推理证据单独保存。公开仓库提供实现与结果摘要；它不包含独立核验全部历史记录或直接重跑正式训练所需的全部数据和模型资源。CPU 回归及可选本地证据复核的运行方式见[验证说明](../docs/agentic_rl_architecture.md#仓库验证与资源边界)。

## 当前三分支先导

原正式基线保持。新先导使用独立来源成功池中的23题，A01/B01/R01分别7/7/9题；整族11/6/6划分。11条train含3,309个监督token位置，SFT3 epoch/9更新。RL共11组、44轨迹、22更新，仅3组有奖励差异；全体奖励为40条0和4条-1。validation选SFT9、RL8，最终test完整成功0/0/0、严重1/1/1、协议2/2/3。

当前维护的采集模块为teacher_collection、teacher_budget、teacher_mainline及teacher_source_expansion；three_branch_pilot保存固定先导的选模与执行逻辑。业务评分使用v17，私有标准v5。历史运行配置不自动适用于当前源码。

训练监督与实际学生生成不同：loss只覆盖assistant动作及回复，学生评测需要自行生成工具调用。teacher reasoning不进入学生监督。当前应先核实学生在train上的生成能力；没有此证据不能直接判断泛化失败。真实诊断和增训须另定预算。
