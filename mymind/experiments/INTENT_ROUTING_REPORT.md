# 意图识别与 Agent 路由评测

日期：2026-10-10。代码 HEAD：`a435a86778f8c163595483d99dee2a5d99b5d1da`，保留未提交工作区。环境为 `learn_claude`。生产代码、真实 `.env`、冻结题目、分类输出和开发选择均未修改。

## 计算方式与运行范围

主 Agent 路由准确率 = 实际主 Agent 与标准主 Agent 相同的题数 / 带路由标准的题数。辅助 Agent 准确率要求实际集合与标准集合完全相同；升级准确率比较是否升级，不与主 Agent 准确率混合。

当前配置没有启用独立升级 Agent：紧急或转人工请求由 General 接待并标记 `escalated=true`。General、Technical、Billing 均为健康实例，监控降权为0。

在同一批152道正式意图测试题上配对比较：

- 路由模块：输入标准意图，生产代码从问题提取实体及紧急度，再执行路由。
- 意图接路由：重放原真实 LLM 和 Qwen 分类输出，经实际 `IntentRecognizer.recognize()` 的融合、实体和紧急度计算，再以未指定意图的 Request 执行 `AgentOrchestrator.run()`。
- 两路对照复用同一道题的 LLM 输出。全部152题的两路/三路预测及置信度均与原正式记录一致；关键词结果也核对一致。

决策后的回答生成、工具执行和 Composer 使用空结果捕获，不参与准确率计分。重放没有生成假向量，没有使用新模型预测，也没有重新请求提供商。因此这是对已有真实分类结果的生产路由链路重放，不是新跑152次在线API或回答质量实验。

## 数据与标准

[原意图数据](../data/eval/intent_optimization_v2.json)的152题、19类、76个问题家族保持原测试划分。新增[路由标准映射](../data/eval/intent_routing_v2.json)按既有业务角色职责映射标准意图，并补充账户凭据泄露且明确要求紧急处理的升级例外。标注由执行代理依据产品策略补充，非独立人工路由标注。

另对[240条迁移数据](../data/eval/agent_migration_dataset.json)复测路由模块。该数据已有主辅Agent、紧急度及升级标准；本次使用当前默认配置，将独立升级角色按既定关闭行为计为General，保留升级标准。按原单条 `question` 输入，不执行 `turns` 多轮，也不将其作为新的独立分类测试集。

## 当前结果

| 数据及输入 | 主 Agent 正确数 | 路由准确率 | 意图正确数 |
|---|---:|---:|---:|
| 正式152题，标准意图 | 152/152 | 100% | 标准输入，不评分类 |
| 同152题，真实三路70/20/10输出重放 | 152/152 | 100% | 144/152（94.74%） |
| 同152题，真实两路85/15输出重放 | 152/152 | 100% | 143/152（94.08%） |
| 240题路由回归，标准意图 | 240/240 | 100% | 标准输入，不评分类 |

152题按标准主Agent分组：General 81/81、Technical 24/24、Billing 47/47。240题辅助Agent集合正确235/240（97.92%），升级标记正确240/240（100%）。两种数据分别统计，不合并成392题独立效果成绩。

三路8道意图误判全部保留正确业务Agent：request→logistics、order_status→query/logistics、invoice→billing、account_security→account、technical_crash→technical、other→query。这说明19类细粒度标签错误可以不改变三个业务Agent之间的分流；不能用路由满分代替分类准确率。

### 标注修订记录

首版附加标准只按意图归业务角色，漏掉已经规定的紧急优先策略，将 `intent-v2-account_security-4-1`“凭据落到别人手里怎样紧急处理”预期为Billing。首轮标准意图及三路链路均151/152（99.34%）；生产路由识别CRITICAL后转General并标记升级，符合当前配置。

按已明确产品策略补充该题升级标准后得到152/152。首轮[原始结果](../artifacts/qwen-optimization-v2/20261010-qwen-01/routing/initial_label_only.json)保留。修订发生在本次附加标注阶段，不能将最终结果称为独立盲测；没有据此改变模型、分类标准或生产路由。

## 原始结果与验证

- [最终逐题及分组统计](../artifacts/qwen-optimization-v2/20261010-qwen-01/routing/current_test.json)：主辅Agent、意图预测、紧急度、实体、升级标记、理由、混淆及标准来源。
- [原真实分类输出](../artifacts/qwen-optimization-v2/20261010-qwen-01/intent_raw_test.json)及[原正式分类统计](../artifacts/qwen-optimization-v2/20261010-qwen-01/intent_test.json)。
- [评测脚本](intent_routing.py)和[7项新契约测试](../tests/test_intent_routing_evaluation.py)：跨领域误判不得读gold纠正、细粒度误判仍可正确分流、澄清优先、升级节点配置、真实预测复用一致性和紧急优先标注。
- 当前全量 **167 passed / 24.21s**；[最终JUnit](../artifacts/qwen-optimization-v2/20261010-qwen-01/routing/tests-routing-full-final.xml)。初版166项测试及统计保留。

新增LLM、embedding HTTP及编码文本均为0；原累计3703 HTTP / 3815文本 / 592网关尝试不变。

从Python后端目录运行，复用已有真实模型结果，不新增模型调用：

```powershell
Set-Location 'D:\Pythonprojects\yan_1\agent\mymind\mymind'
& 'D:\anaconda3\envs\learn_claude\python.exe' -m experiments.intent_routing
```

脚本读取 `.env` 中当前Agent功能配置但不输出凭证、不修改配置。要复现上述准确率，应保持当前独立升级节点关闭的配置及原分类运行目录。

## 限制

数据为合成题目，路由标准来自业务策略映射；19类到业务Agent的映射比细粒度分类容易。152题只覆盖单域单轮；240题主要为模板改写回归。未统计真实客服分布、线上模型变化、多轮路由、监控降权、Agent执行失败后的回退与独立人工审核。100%仅是这两批数据在当前条件下的路由结果。
