# Qwen RAG 与意图识别第二轮优化报告

日期：2026-10-10。环境：Windows、Python 3.10.20、learn_claude、Chroma 0.5.23。HEAD：`a435a86778f8c163595483d99dee2a5d99b5d1da`。运行目录：[20261010-qwen-01](../artifacts/qwen-optimization-v2/20261010-qwen-01/manifest.json)。起点已有未提交实现保留，未 reset、clean、commit、push 或部署；运行创建时的工作区状态保存在 manifest。

## 交付结果

四项工程修复及原口径交付全量 **160项测试通过，20.56s**；随后路由评测新增7项，当前全量 **167项通过，24.21s**，见[路由报告](INTENT_ROUTING_REPORT.md)。前次补齐新增6项，原口径回归新增4项；相对原129项起点累计新增38项。独立Qwen运行的13个阶段全部 measured，已完成开发选择、正式检索、完整意图配对、三组固定30题回答、实际生产组件和完整应用验收。所有模型质量结果使用真实配置模型，不使用假向量或stub回答。

主embedding为 `qwen3.7-text-embedding-flash / 768`，实际可用基地址为 `https://maas.qianwenaiapi.com/compatible-mode/v1`；原给定 `/apps/anthropic` 的两条embedding探测路由返回404，记录保留。聊天独立使用 `deepseek-flash`、DeepSeek Anthropic Messages接口。可用性及此前35次HTTP接入记录见 [迁移报告](QWEN_EMBEDDING_MIGRATION_REPORT.md)。当前阶段没有再修改真实 `.env`。

最终主参数为结构递归 **512服务token、overlap 0、向量权重0.5**，备用MiniLM仍 **240 token、overlap 0、向量权重0.5**；意图保持 **70/20/10**，embedding失败或未配置时 **85/15**，阈值0.5、原54条模板不变。参数数值保持；Qwen主分块的计量单位由保守字节改为真实服务token，已有字节索引需显式补齐。

## 工程正确性

| 审查问题 | 当前实现与本轮验证 |
|---|---|
| 主模型和工具同时30秒，备用无法交回 | 主查询共享8秒总预算，包含计数、等待、编码与解析/网络任务；工具仍30秒，主预算最多20秒，导入与模板另计。完整MCP/Agent测试包含立即失败、计数+编码共用预算、慢HTTP及解析阻塞、双路失败与恢复。真实MiniLM延迟故障在281.38ms返回备用正文。 |
| 同步统计阻塞事件循环、锁跨远程调用 | 异步统计端点卸载同步工作，远程准备不持索引状态锁，短提交更新主备/BM25与缓存。受控导入/查询/repair、并发改写、更新/删除及旧在途缓存回填回归通过。实际应用导入暂停时健康/监控/统计三请求在116.94ms内返回，早于解除远程暂停。 |
| 并发导入共享last_import串批 | add/upload使用 `import_documents()` 的本请求结果。测试覆盖交错主成功/主失败；真实应用并发add 1篇/upload 2篇的来源ID、逐文档结果和片段数分别匹配。 |
| BM25单路仍编码文档 | BM25单路跳过文档和查询embedding，复用保存的相同分块。所有embedding接口一调用即失败的测试仍完成BM25；新增Qwen入口契约同样通过。Qwen重新分块的计数本身是实际embedding请求，单列TOKEN_COUNT，不声称无分块缓存时必定零HTTP。 |

生产实现：[embedding.py](../core/embedding.py)、[document_chunking.py](../core/document_chunking.py)、[indexed_knowledge_base.py](../mcp/indexed_knowledge_base.py)、[api/main.py](../api/main.py)、[tool_manager.py](../mcp/tool_manager.py)、[intent_recognizer.py](../core/intent_recognizer.py)。相关测试：[四项与状态回归](../tests/test_rag_optimization_v2.py)、[Qwen接入契约](../tests/test_qwen_embedding.py)、[本次缓存/预算/BM25/长文/工具协议测试](../tests/test_qwen_experiment_budget.py)。[当前JUnit](../artifacts/qwen-optimization-v2/tests-original-metrics.xml)；编译及 `git diff --check` 通过。Windows受限工具执行环境失败时使用正常审批后的环境，不把历史129项通过当作本轮证据。

新增实际错误也已处理：支持的长章节若超过计数接口单次输入上限，递归拆分后再精确校验；鉴权与网络错误不吞掉。实验 `MeasuredGateway` 转发原生 `tool_protocol`，使实际Anthropic工具续轮保持正确消息格式。原生产gateway本身使用正确协议；问题发生在计量包装器，前两次应用实验的8次422保留，第三次17项验收全部通过且无新增提供商失败。

## 精确计量与缓存

计量方式为 `qwen_single_input_usage_tokens`。单文本输入“标题/章节 + 正文”发送真实Qwen embedding请求，读取其 `usage.total_tokens`，另计16余量。不存在已验证的本地同模型tokenizer时不借用其他模型；缺失用量字段明确失败。官方说明提供20条批次及128000总token输入约束，usage字段返回模型计量用量。[模型说明](https://platform.qianwenai.com/docs/developer-guides/embeddings/embedding)、[同步embedding用量字段](https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api)。批次保护使用保守UTF-8字节上界；分块预算使用真实服务token，两者不混称。

计数请求返回的归一化向量复用于完全相同的文档输入。线上计数缓存上限4096；实验SQLite按实际模型、endpoint、维度、任务、标题与文本复用计数/向量。重启缓存契约通过。本次最终缓存含2491条计数记录、1926条任务向量记录（文档1498、查询180、分类248）；记录数与API文本数不同，计数产生的向量并非额外免费生成的向量。

主索引签名包含计量方式；旧字节块不会被当作精确token块使用，待补齐期间完整备用持续工作。主块调整不重建固定MiniLM分块，主备各自片段、向量和BM25始终分离。相同排序用于不同RRF权重，不做完整参数笛卡尔积。

## 冻结数据与选择

[RAG数据](../data/eval/rag_optimization_v2.json)：60篇公开合成政策、180题、90家族；开发60题（50可答/10无答案），测试120题（100可答/20无答案）。问题家族和近义改写不跨集，逐事实原文引用/偏移、条件、例外、干扰与无答案主题校验通过。16篇长章节用真实Qwen计量为1513–1542 token，均超过768，部分超过1536。测试长章节40题，条件/多事实100题，例外20题，错误码80题，标签可重叠。旧题保持独立回归用途。

[意图数据](../data/eval/intent_optimization_v2.json)：当前19类、228题、114家族，开发76/测试152，每类4/8题；另20条歧义诊断单列。复用完全相同题目和实际 `deepseek-flash` 的248条历史LLM分类输出，两路/三路使用同一道题同一LLM输出和关键词结果。Qwen分类向量为新真实编码。

2026-10-10按用户后续要求恢复原项目检索指标；原冻结选择时间及参数保持，开发集复核结果一致。数据未根据测试结果重写。[校验](../artifacts/qwen-optimization-v2/20261010-qwen-01/data_validation.json) 与 [selection.json](../artifacts/qwen-optimization-v2/20261010-qwen-01/selection.json) 可追溯；选择于 **12:37:34.576615** 冻结，先于正式测试。保留512与等权的规则是开发主指标无清楚收益、不到预设2个百分点参考量级时保持起点；模板仅保留原54条，开发阈值0.4/0.5/0.6结果相同，保持0.5。

这是共享知识库的问题泛化评测，同文档可出现于两集，不是未见文档泛化。政策和题目有模板化内容，精确实体/错误码有利于词法检索；Qwen主路多个开发组饱和。冻结数据沿用已有正式结果，未在看过测试后补题，因此本轮不具备充分区分普通/结构切分和主融合权重的能力。

## 开发集结果

固定768维、单路候选20、RRF k=60，关闭改写/重排/回答。分块对照固定overlap0；以下60题总体口径包含50可答和10无答案。

检索直接使用原项目 `evaluation/retrieval_metrics.py::query_metrics`，未修改该函数。Recall@5/10为前K条是否命中任一相关source；Precision@3为前3条首次出现的相关source数量除3；MRR@10为首个相关source的倒数排名，nDCG沿用原来源分级及重复来源零增益。所有查询参与总体均值，包括无答案题：相关来源为空时Recall=1、Precision/MRR/nDCG=0。无答案非空率按原定义单列，不能等同回答编造率。

本次仅对已保存真实排名重计分，未重新编码、检索或生成回答，新增embedding/LLM调用均为0。检索逐题保留前10条，实际生产工具返回Top5；相同生产题的离线比较也截断到Top5。原逐事实引用/条件标注仍供数据校验和回答审核使用。

| 配置 | 块数 | Recall@5 | Recall@10 | Precision@3 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| 普通递归256 | 324 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 普通递归512 | 108 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 普通递归768 | 92 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 结构递归256 | 384 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 结构递归512 | 168 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 结构递归768 | 152 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |

普通/结构及三个块长的原Recall与Precision均相同，不能据此宣称结构或256更优。开发集未达到预设2个百分点改参条件，保留结构512。

| 结构512 overlap | Recall@5 | Precision@3 | 平均Top5上下文字符 | 近似重复比例 |
|---|---:|---:|---:|---:|
| 0 | 1.0000 | 0.2778 | 2750.97 | 0.6291 |
| 32 | 1.0000 | 0.2778 | 2795.40 | 0.6331 |
| 64 | 1.0000 | 0.2778 | 2855.53 | 0.6495 |

overlap 0/32/64的Recall、Precision相同，正overlap没有可确认收益，并增加上下文与近似重复；保留0。所有块预算/章节边界校验通过，正overlap未进入正式测试候选。

同章节48对相邻块的真实重叠：0组均0，32组48对均32 token，64组为44（32对）/57（4对）/58（8对）/59（4对），均值48.67。输入含16余量最大值分别512/467/478，均未越512。保存的真实计数缓存可离线重建 [overlap_observations.json](../artifacts/qwen-optimization-v2/20261010-qwen-01/overlap_observations.json)。

主融合0.25/0.5/0.75的Recall@5/10均1.0000、Precision@3均0.2778；备用0.1/0.25/0.5同样如此。备用开发MRR分别0.8250/0.8083/0.8056，但没有达到原Recall主指标改参条件，保持主备0.5。按开发数据复核得到512/0/0.5/0.5，与原冻结参数一致；没有根据测试重新选择。

## 正式检索结果

120题总体口径：100可答、20无答案。以下是相同真实排名按原算法重新计算的结果。

| 配置 | 块数 | Recall@5 | Recall@10 | Precision@3 | MRR@10 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen结构512/0混合0.5（当前=候选） | 168 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 同块纯向量 | 168 | 1.0000 | 1.0000 | 0.2778 | 0.8292 | 0.8303 |
| 同块纯BM25 | 168 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 普通递归512/0纯向量 | 108 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| MiniLM240/0混合0.5（当前=候选） | 466 | 1.0000 | 1.0000 | 0.2778 | 0.7917 | 0.8024 |
| MiniLM纯向量 | 466 | 0.9083 | 0.9417 | 0.2083 | 0.5532 | 0.6074 |
| MiniLM纯BM25 | 466 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 原Qwen字节512/0混合0.5 | 740 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 原字节块纯向量 | 740 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |
| 原字节块纯BM25 | 740 | 1.0000 | 1.0000 | 0.2778 | 0.8333 | 0.8333 |

每道可答题仅标注一个相关来源，因此单题Precision@3最高1/3；总体上限100/(120×3)=27.78%。主备均达到该上限，不能解释为72.22%的片段语义无关。若仅看100道可答题，主备Recall@5/10均100%、Precision@3均33.33%；主MRR=1.0000，备用MRR=0.9500。可答题分组与逐题原分数在各retrieval文件的 `answerable_summary` 和 `rows` 中。

20道无答案题各组均返回非空候选，按原算法的 `no_answer_false_positive` 为20/20=100%；这里没有使用旧实验的KnowledgePolicy门控来代替实际检索结果。它只反映非空检索，与回答阶段是否编造分别计分。

MiniLM真实编码/排名匹配复用并保留 `reused_from`；Qwen为真实编码。原字节512是固定对照，未参与开发选择。恢复原口径后，真实token与字节混合的Recall、Precision和来源MRR均相同，不能再宣称检索召回提升；块数740→168、上下文变化和回答审核差异仍为实测事实。

按全部120题、60家族做配对bootstrap（1000次、seed20261009），备用BM25相对等权混合的Recall与Precision差为0；MRR差+0.0417，95%CI[0.0125,0.0750]，nDCG差+0.0310，CI[0.0092,0.0558]。这支持本合成集上BM25首次相关来源排名更靠前，不支持召回率更高。既定混合生产策略保持，正式测试没有用于重新调权重。

## 意图与模板成本

| 数据 | 两路85/15 macro-F1 / accuracy | 三路70/20/10 macro-F1 / accuracy |
|---|---|---|
| 开发76题，阈值0.5 | 0.986633 / 75/76 | 0.958563 / 73/76 |
| 测试152题，阈值0.5 | 0.940719 / 143/152 | 0.947192 / 144/152 |

测试仅1个分歧：`intent-v2-other-4-0`“计算三角形面积公式”，两路query、三路OTHER，预期OTHER。真实LLM输出query/0.6，embedding输出billing/0.40293；三路是融合后低于阈值而拒到OTHER，并非embedding本身正确分类。无改错题，OTHER比例两路6/152、三路7/152；开发三路更差，20条诊断允许标签匹配两路19/20、三路16/20。不能据测试多对1题宣称稳定收益，固定三路与降级方式保持。

模板冷准备54条实际编码2391.92ms，命中重启55.27ms、0模板编码。另测实际recognize：冷首请求及准备就绪约2825.22ms（54模板+1消息、4次QwenHTTP、1次LLM）；缓存命中启动59.86ms、0模板编码；3次热请求为687/610/1297ms，各1条消息编码+1次LLM。计入实验限速，样本小，不作为生产P95承诺。复用248条旧LLM输出的离线融合耗时不冒充新并行LLM耗时；真实热请求与模板准备分开记录，冷成本没有消失。

后续按用户要求补齐意图接路由统计：同152道正式题标准意图路由与真实三路分类接生产路由重放均152/152，两路对照也152/152；三路8道细粒度误判均未改变正确业务Agent。240条标准意图路由回归240/240，辅助235/240。新增路由标准按产品业务策略派生，首版漏掉紧急升级例外的151/152结果保留，补正附加标注后重新统计；非独立人工标注或盲测。决策后的回答/工具不计分，新API为0。完整计算与逐题原始结果见[路由报告](INTENT_ROUTING_REPORT.md)。

## 固定30题回答与复核

名单在检索前冻结，各组相同24可答/6无答案，模型deepseek-flash、温度0、Top-K5、重排输入12000字符、最终上下文6000字符、max_tokens4096；沿用网关已有不完整响应8192续试并计入返回用量，续试标记与其缺失边界另列。回答实验关闭改写，保存真实候选、重排、送入模型的上下文、回答和JSON审核。

| 组别 | 已完成审核 | 事实有支持 | 条件完整（可答24题） | 条件完整（全部30题） | 无依据主张 | 引用错误回答 | 无答案编造 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen真实token512/0/0.5（当前=候选） | 30/30 | 30/30（100%） | 24/24（100%） | 30/30（100%） | 0/30 | 0/30 | 0/6 |
| 原Qwen字节512/0/0.5 | 30/30 | 30/30（100%） | 23/24（95.83%） | 29/30（96.67%） | 0/30 | 1/30（2处） | 0/6 |
| MiniLM240/0/0.5 | 30/30 | 30/30（100%） | 20/24（83.33%） | 26/30（86.67%） | 0/30 | 0/30 | 0/6 |

审核直接统计已有 `facts_supported`、`conditions_complete`、`unsupported_claims`、`citation_errors`、`unanswerable_fabricated` 字段。原审核没有生成relevance/accuracy/completeness/helpfulness四项0–1分数，不补造这类分数。事实有支持在24道可答题中三组也均为24/24；6道无答案题单列编造率。

字节对照 `rag-v2-004-1` 有两处引用指向错误，模型仍判断核心事实有支持；事实支持与引用正确分别统计。原审核内容保持，逐题失败及引用错误见summary的 `answer_failures`。
备用29题与原文/模型/上下文契约匹配后复用，旧审核失败 `rag-v2-002-0` 补跑真实重排、回答和审核3次，原失败保留。备用条件遗漏为001-0/005-0/007-0/011-0，均遗漏“原订单申请人提供订单号和完整操作记录”；字节组005-0也遗漏该条件。主抽查001-0和长题014-1包含该条件及例外，014-1含12步骤。

审核是同模型辅助审核，**不是人工审核**。代理复核确认上述遗漏与各组无答案目标未编造；另指出备用absent-04-0附带摘要遗漏两处“经确认”的故障限定。该遗漏未改变无答案目标判定，但模型审核的事实支持30/30不能等同零瑕疵，代理争议记录单列而不悄悄改写原始审计。详见 [代理复核](../artifacts/qwen-optimization-v2/20261010-qwen-01/answer_agent_review.md)。

## 真实生产组件与应用

20道固定可答题经 `MCPToolManager → KnowledgeBase → Persistent Chroma`，实际Top5的Recall@5=1.0000、Precision@3=0.3333、首次相关来源MRR=1.0000，与同题离线Top5计分一致，指标变化ID为空。工具只返回5条，不能把其中的Recall@10当作独立Top10实验。暖查询P50 **260.29ms**、P95 **291.53ms**。11项状态检查全部通过：查询故障真实备用、恢复主路、降级新增/更新、pending继续备用、更新可见/旧文消失、重启pending保留、显式repair、恢复新主正文、主备删除和281.38ms延迟备用返回。故障为受控注入，不声称真实提供商发生同样故障。2题实际LLM改写及重排路径单列，共4次新网关调用。

Docker Redis6379和Chroma8001可用。实际FastAPI lifespan、原生Agent、HTTP Chroma、Redis验收17项全通过；主/备用工具返回正文保存在application的tool_payloads。并发导入、删除、远程暂停期间监控、主Chat与注入慢查询Chat、业务库恢复、隔离缓存清理及旧collection数量保持均通过。第三次无新增422，不将前两次工具协议错误掩盖为成功。

确认旧knowledge_base六篇与仓库公开内置文档逐条相同后，完成当前业务主备：`knowledge_qwen3_7_text_embedding_flash_768_v1=6`、`knowledge_minilm_v1=6`，main_complete=true。原knowledge_base=6、episodic=0、user_profile=0保持。临时业务文档、实验memory collection及Redis DB15随机前缀清理；真实用户内容未用于外部模型实验。Gemini生产collection当前不存在；同前缀旧维度清理及受控HTTP清理已验证，不声称删除实际不存在的对象。

此前18项真实HTTP/Redis smoke另外证明跨worker缓存命中与更新失效；degraded结果不缓存以允许恢复。轻量20次健康观测P50 0.71/P95 1.16ms、10ms心跳最大间隔11.13ms，范围为真实知识库/端点配固定编排器统计替身，不作为全应用压测；完整实际应用的三端点观测和暂停验收另在application.json。实际应用经ASGI生命周期执行，未启动外部监听服务或部署。

## API用量与失败

| 计账项目 | 本次Qwen正式运行 | 历史扣除 | 累计 / 上限 |
|---|---:|---:|---:|
| embedding HTTP尝试 | 3018 | 685 | 3703 / 20000 |
| 编码输入文本（含计数） | 3551 | 264 | 3815 / 10000 |
| LLM网关尝试 | 249 | 343 | 592 / 600 |

历史HTTP685由Google650与Qwen接入35组成；文本264由原Google128尝试输入和Qwen接入136确认编码组成。原404没有返回向量，失败请求另在探测记录。Qwen本次3018个HTTP均成功，实际3551输入为TOKEN_COUNT2491、RETRIEVAL_DOCUMENT400、RETRIEVAL_QUERY243、CLASSIFICATION417；服务报告 **781048 embedding token**，另有此前Qwen接入2864 token。计数是付费编码，不声称其HTTP或成本消失。

本次249次网关尝试中，240份成功返回用量合计 **input424187、output572397、cache_read69120 token**；cache_read为input的一部分，不能再加成总输入。8次422与1次中断未返回可用token，尝试已计账，不虚构其token为0。底层不完整响应自动续试聚合到网关返回用量；36份返回明确标记续试，早期20份记录缺少续试标记，不能据此声称底层LLM HTTP次数恰等于249或全部续试次数已知。

含历史342份LLM成功返回，累计582份返回报告input647627/output951467/cache_read125310 token；历史Google没有返回编码用量，不将其账本0解释为免费。总网关预算口径是complete顶层尝试，底层自动续试合并token，提供商账单仍以其实际计费为准。

实际限速等待 **21358ms**，与编码计时/缓存复用分别保存。计数缓存命中7754次、查询向量命中660次、文档向量缓存命中1643次；这些是复用事件，不是等量可承诺的账单节约。额度余量为HTTP16297、计账文本6185、LLM8次，不自行扩容或重复付费实验。

必要失败均保留：原Google每日额度429及历史blocked；原Qwen基地址404；测试临时目录setup失败；1次回答运行中断；旧备用审核空JSON；首次实际应用并发断言误判；包装器协议8次422。第二次应用旧检查曾写measured但仍有422，不接受为最终原生协议证据，见 [尝试复审](../artifacts/qwen-optimization-v2/20261010-qwen-01/attempt_reviews.json)。最终manifest只采用第三次修正验收与全部完成阶段；没有把失败从预算或报告中删除。

## 原始交付与复现

- [运行清单](../artifacts/qwen-optimization-v2/20261010-qwen-01/manifest.json)、[冻结选择](../artifacts/qwen-optimization-v2/20261010-qwen-01/selection.json)、[原口径完整汇总与逐题排名移动](../artifacts/qwen-optimization-v2/20261010-qwen-01/summary.json)、[最终验证与用量](../artifacts/qwen-optimization-v2/20261010-qwen-01/delivery_validation.json)。
- 逐题检索/排名/片段位于运行目录 `retrieval/`、`rankings/`、`chunks/`；90份回答上下文/审核位于 `answers/`。如 [主长文回答](../artifacts/qwen-optimization-v2/20261010-qwen-01/answers/qwen-structure-512-0-test-0.5-rag-v2-014-1.json)。
- [意图开发](../artifacts/qwen-optimization-v2/20261010-qwen-01/intent_dev.json)、[意图正式与每类结果](../artifacts/qwen-optimization-v2/20261010-qwen-01/intent_test.json)、[真实LLM/embedding逐题原输出](../artifacts/qwen-optimization-v2/20261010-qwen-01/intent_raw_test.json)、[模板及热请求](../artifacts/qwen-optimization-v2/20261010-qwen-01/intent_latency.json)。
- [生产20题及状态验证](../artifacts/qwen-optimization-v2/20261010-qwen-01/production.json)、[实际应用与Agent工具正文](../artifacts/qwen-optimization-v2/20261010-qwen-01/application.json)、[费用](../artifacts/qwen-optimization-v2/20261010-qwen-01/api_usage.json)、[LLM返回用量](../artifacts/qwen-optimization-v2/20261010-qwen-01/llm_usage.json)、[422失败](../artifacts/qwen-optimization-v2/20261010-qwen-01/llm_failures.json)、`attempts/`及`failed_application/`。
- [可执行Qwen入口](qwen_optimization.py)、[实际应用验收](qwen_application_acceptance.py)、[汇总入口](summarize_optimization_v2.py)、[使用说明](../docs/rag.md)、[冻结方案与历史边界](../docs/rag_optimization_experiment_plan.md)。实验产物/运行数据库按仓库规则忽略Git，但保留在工作区，本次不提交或推送。

以下复核当前完成运行，不新增模型调用：

```powershell
Set-Location 'D:\Pythonprojects\yan_1\agent\mymind'
$taskPython='D:\anaconda3\envs\learn_claude\python.exe'
& $taskPython -m pytest -q -p no:cacheprovider
Set-Location 'D:\Pythonprojects\yan_1\agent\mymind\mymind'
$taskOutput='artifacts/qwen-optimization-v2/20261010-qwen-01'
& $taskPython -m experiments.qwen_optimization --stage all --resume --output $taskOutput
& $taskPython -m experiments.qwen_optimization --stage byte_baseline --resume --output $taskOutput
& $taskPython -m experiments.summarize_optimization_v2 --output $taskOutput
```

新目录完整复跑命令见实验README；当前只有8次LLM余量，不足新跑30题三组。新目录不是新的授权预算，不自动再跑。单阶段 `--stage` 支持validate/smoke/chunking/overlap/fusion/intent_dev/select/test/intent_test/answers/production/application/byte_baseline，依赖未完成会明确blocked。保持同模型、同endpoint、冻结数据及历史运行路径；模型或题目不匹配不能复用历史输出。

## 结论与未覆盖边界

恢复原指标后，当前主混合和原字节混合的来源召回、Precision与MRR相同；真实token计量减少片段数量，回答条件与引用审核更好。正overlap没有来源指标收益且增加重复/上下文，关闭。主备等权与意图阈值保持；备用BM25的来源排序更好、三路意图开发退化均如实报告。

尚未覆盖：真实客服分布、未见文档泛化、人工审核、外部部署负载、多后端进程写入、提供商真实长时间中断；最大10MB文件接口上限通过本地长章节契约验证，未外发10MB文本做成本实验。主路开发指标饱和、相同模型回答和审核存在偏差，代理抽查不是完整人工确认。Chroma遥测依赖有 `capture()` 参数告警，实际读写通过，未引入无关依赖升级。原Gemini质量实验仍保持历史blocked，不需要用新Qwen密钥补跑旧模型。

可用于求职说明的真实结论：定位并修复工具超时、事件循环锁阻塞、并发导入归属及单路评测成本问题；实现独立主备索引、明确降级/补齐/恢复与简单模板缓存；在累计预算内完成冻结数据真实模型对照，以当前160项回归及实际Docker/Agent验收验证工程正确性。报告合成集token计量与回答审核结果、关闭无清楚收益的overlap，也如实报告备用混合和意图embedding未证明优势；不宣称真实生产准确率100%。
