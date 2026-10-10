# RAG 与意图识别第二轮实验报告

本报告保留 Gemini 第二轮的历史实测和原额度断点。2026-10-10 已按后续授权切换 Qwen，并完成替代模型的第二轮质量实验与实际应用验收；完整当前结果、累计用量与复现命令见 [Qwen 第二轮报告](QWEN_RAG_OPTIMIZATION_REPORT.md)，配置见 [RAG 使用说明](../docs/rag.md)。下文 Gemini 的 failed/blocked 状态仍是历史事实，不改写为 Qwen 结果。

## 2026-10-10 当前交付

四项工程修复全部通过当前 `learn_claude` 全量 **160项 / 20.56s**，其中原第二轮新增13项、Qwen接入新增8项、前次补齐新增6项，本次原口径新增4项。冻结语料为60篇/180题RAG与19类/228题意图，沿用家族60/120、76/152划分；原248条真实LLM分类输出、固定MiniLM结果匹配复用并标注。Qwen开发选择在正式测试前冻结，主512 token/overlap0、主备等权0.5、意图阈值0.5保持；计量改为单输入真实usage，原字节主块需显式补齐。

按用户后续要求恢复原来源算法。120题总体主/备用Recall@5/10均100%、Precision@3均27.78%，来源MRR为0.8333/0.7917；无答案题按原规则参与均值。产品混合策略未更改，正overlap原指标无收益且增加上下文/重复，保持0。两路/三路意图macro-F1为0.940719/0.947192，仅多改对1题，开发集三路更差，未证明稳定质量增益；固定融合方式保留。

预选30题主回答均完成真实模型生成与审核；备用30题中29题匹配复用、原审核失败1题补跑成功，4项条件遗漏仍保留。审核为模型审核和代理复核，无人工审核。20题实际KnowledgeBase/Persistent Chroma/工具的来源Recall、Precision及首次相关来源排名与离线一致，11项降级、更新、重启、补齐和删除检查通过；实际FastAPI lifespan、HTTP Chroma、Redis及Agent验收17项通过。原工具协议计量包装器造成的8次422已修复并保留失败，最终应用无新增提供商错误。

本机旧6篇业务原文与仓库内置公开文档逐条确认后完成Qwen/MiniLM双索引，未发送私人语料；Gemini生产索引当前不存在，同前缀清理逻辑与受控测试已验证。完整应用通过ASGI实际生命周期验收，未部署外部服务。原字节与真实token512固定混合对照的来源Recall/Precision相同，片段数及回答条件/引用审核有差异；该对照不用于调参，其30题回答及最终用量见当前报告。

日期：2026-10-09。运行目录：[`20261009-v2-01`](../artifacts/rag-optimization-v2/20261009-v2-01)。

## 交付状态

四项工程问题已修复，本轮新增 13 项测试及全量 142 项测试通过。数据已校验冻结，备用检索开发与正式测试、真实两路意图测试、30 题备用回答及 20 题真实 Chroma/工具验证已执行；其中一次回答审核失败，阶段仍保留 failed。Google 返回每日额度耗尽，主路参数实验、三路意图质量与真实模板启动测量尚未完成。保留断点、累计用量和失败记录，不以历史测试、假向量或备用结果代替这些实验。

生产参数保持：Gemini `gemini-embedding-001 / 768`、结构递归 `512 / overlap 0`、主向量权重 `0.5`；MiniLM `all-MiniLM-L6-v2 / 240 / overlap 0`、备用向量权重 `0.5`。正常意图融合仍为 `70/20/10`，embedding 不可用时为 `85/15`，阈值 `0.5`。未改真实 `.env`，未重建用户生产索引，未修改 Java、前端或部署，未 commit、push。

起点 HEAD 为 `a435a86778f8c163595483d99dee2a5d99b5d1da`。第一轮的未提交改动保留；本轮开始时的实际状态见 [`baseline.json`](../artifacts/rag-optimization-v2/20261009-v2-01/baseline.json)。实验入口另保存自己的运行清单，不能将它的较晚状态当作原始工作区。

## 工程修复与验证

| 问题 | 实现 | 本轮证据 |
|---|---|---|
| HTTP 与工具同时 30 秒，备用来不及返回 | Google 查询共用 8 秒墙钟预算，覆盖能力探测、token 计数、等待、编码和解析/网络任务；工具仍 30 秒，Google 子预算最多 20 秒。导入与模板准备独立计时 | 真实本地慢 HTTP 经 `MCPToolManager.call → knowledge_search → KnowledgeBase.search` 和 Agent 路径返回实际备用正文；计数与编码不重置预算。受控 DNS 解析阻塞时，在释放事件前返回备用正文 |
| 统计同步阻塞事件循环，锁跨 Google 请求 | `/monitor`、`/knowledge/stats` 卸载同步工作；远程准备移到状态锁外，写操作串行，状态/Chroma/BM25/缓存仅在短提交阶段更新 | import/query/repair 受控暂停时，健康、统计和心跳先于远程释放返回；并发查询、更新、删除、补齐和旧在途缓存失效测试通过 |
| add/upload 从共享 last_import 读到其他批次 | 两端点保留 `import_documents` 的当前请求返回字典，使用其中 ID、状态、片段数 | 两端点均受控交错 A/B 响应；A 主成功、B 主失败而备用成功，响应归属分别正确 |
| BM25 单路无条件编码 | 单路 BM25 跳过文档和查询 embedding；已有分块直接复用；MiniLM 纯 BM25 仅准备 tokenizer | embedding 一调用即抛错仍完成 BM25；所有 Google 接口不可用、已有分块仍完成。重新 Gemini 分块可能需要 countTokens，未声称必定零 HTTP |

相关实现：[`embedding.py`](../core/embedding.py)、[`indexed_knowledge_base.py`](../mcp/indexed_knowledge_base.py)、[`api/main.py`](../api/main.py)、[`gemini_rag.py`](gemini_rag.py)。

意图模板缓存按模型、维度、CLASSIFICATION 任务、模板内容与顺序直接匹配。并发请求共享一个准备任务，在线最多等待 3 秒，准备总预算独立为 180 秒；关闭生命周期等待准备结束。学习或配置变化使旧缓存不再匹配，损坏缓存重建，准备失败走 85/15 并允许后续恢复。冷启动、重启命中与热请求的真实 API 测量保留在独立阶段，未把启动准备声称为消失的成本。

验证环境：Python `3.10.20`，`learn_claude`，Chroma `0.5.23`。交付前全量结果 **142 passed / 29.73s**，包含新增 **13 项**。测试源为 [`test_rag_optimization_v2.py`](../tests/test_rag_optimization_v2.py)，JUnit 为 [`tests-engineering-delivery.xml`](../artifacts/rag-optimization-v2/tests-engineering-delivery.xml)，先前本轮运行记录也保留。旧测试替身同步到请求返回值契约。Windows 沙箱阻塞本地 socketpair/临时目录时，使用工具审批后的正常执行环境。另已完成 Python 编译检查与 `git diff --check`。

## 数据与评测口径

[`rag_optimization_v2.json`](../data/eval/rag_optimization_v2.json) 含 60 篇明确标记的合成政策、180 题、90 个两表达家族：开发 60（50 可答、10 无答案），测试 120（100 可答、20 无答案）。每题保存正确来源、逐事实原文及偏移、适用条件、禁止断言和类型标签。16 篇长文的自然章节经真实 Google 计数为 1704–1736 token，超过最大候选 768 的两倍；测试长章节标签 40 题，条件/多事实 100 题，例外 20 题，精确错误码 80 题，标签可重叠。

[`intent_optimization_v2.json`](../data/eval/intent_optimization_v2.json) 对齐当前 19 类，228 题，开发 76、测试 152；每类 4/8 题，按场景家族切分。另有 20 道多意图诊断，不加入单标签 macro-F1 分母。旧 `gemini_rag_eval.json` 和边界数据保留为旧回归资料，没有冒充新增独立题。

原文引用、家族不跨集、无答案主题全库缺失及真实长章节计数均通过，记录为 [`data_validation.json`](../artifacts/rag-optimization-v2/20261009-v2-01/data_validation.json)。语义复核由执行代理完成，未标为人工审核。

这是共享知识库上的问题泛化评测，同一文档可以出现在两集。政策和题目有模板化特征，错误码与产品词有利于词法检索；结果不能推广为真实客服用户表现或未见文档泛化。主块 BM25 在开发集饱和，但备用向量/混合尚未饱和；Google 主向量失败前没有完整主基线，不依据已看到的正式测试重新写题或补难题。

检索直接使用原项目 `evaluation/retrieval_metrics.py::query_metrics`，未修改该函数。Recall@5/10为前K条是否命中任一相关source；Precision@3为前3条首次出现的相关source数量除3；MRR@10为首个相关source的倒数排名，nDCG沿用原来源分级及重复来源零增益。所有查询参与总体均值，包括无答案题：相关来源为空时Recall=1、Precision/MRR/nDCG=0。无答案非空率按原定义单列，不能等同回答编造率。

本次仅对已保存真实排名重计分，未重新编码、检索或生成回答，新增embedding/LLM调用均为0。检索逐题保留前10条，实际生产工具返回Top5；相同生产题的离线比较也截断到Top5。原逐事实引用/条件标注仍供数据校验和回答审核使用。

## 开发选择与正式检索结果

历史已测MiniLM开发权重0.1/0.25/0.5的原Recall@5/10均1.0000、Precision@3均0.2778，保留0.5。Google主路开发质量实验当时被每日额度阻塞，512/0不宣称为当时已验证最优。

| 历史正式组（120题总体） | Recall@5 | Recall@10 | Precision@3 | MRR@10 |
|---|---:|---:|---:|---:|
| gemini-structure-512-0-test-bm25 | 1.0000 | 1.0000 | 0.2778 | 0.8333 |
| minilm-structure-240-0-test-0.5 | 1.0000 | 1.0000 | 0.2778 | 0.7917 |
| minilm-structure-240-0-test-bm25 | 1.0000 | 1.0000 | 0.2778 | 0.8333 |
| minilm-structure-240-0-test-vector | 0.9083 | 0.9417 | 0.2083 | 0.5532 |

以上是历史真实排名的离线重计分；未改写旧API输出、旧失败状态或旧运行选择。备用混合和BM25的来源Recall/Precision相同，BM25的首次相关来源排名更靠前。旧运行JSON保留作为历史记录，当前统一指标以Qwen正式报告及其summary为准。

## 意图实验

本轮真实可测路径为 LLM 85% + 关键词 15%。19 类测试 macro-F1 **0.940719**，accuracy **0.940789（143/152）**，OTHER 输出比例 **6/152=0.039474**；开发 macro-F1 0.986633、accuracy 75/76。原始分类、关键词、融合、逐标签 precision/recall、混淆矩阵及 20 道歧义诊断见 [`intent_llm_raw_test.json`](../artifacts/rag-optimization-v2/20261009-v2-01/intent_llm_raw_test.json) 与 [`intent_llm_test.json`](../artifacts/rag-optimization-v2/20261009-v2-01/intent_llm_test.json)。

错误共 9 题：request→logistics 1；order_status→query/logistics 各 1；invoice→billing 1；account_security→account 2；technical_crash→technical 1；other→query 2。每类测试仅 8 题，细分类错分说明层级近义边界仍有困难，不足以推断真实线上错误率。

真实 LLM 分支测试耗时 n=152，P50 **1027.93ms**、P95 **1731.94ms**，含网关完整响应重试，关键词开销很小。这不包括 Gemini 模板准备，也不是三路实际并行延迟。三路质量、改对/改错、真实模板冷启动/缓存命中/热请求均 blocked；模板缓存的契约与失败恢复已测试，真实 API 节省尚未测量。后续三路将复用同一道题已保存的真实 LLM 输出，保持 70/20/10 与 85/15，不引入提前结束分类或生产 n-gram。

## 回答与生产组件

回答名单在检索前固定为 30 题（24 可答、6 无答案），备用当前与候选相同，实际只计算一组；主配置与主候选未运行。固定 deepseek-flash、温度 0、候选 10、最终 Top-5、重排输入 12000 字符、实际上下文最多 6000 字符，本回答组不改写。原始检索、重排状态、实际上下文、回答与模型审核在 [`answers/`](../artifacts/rag-optimization-v2/20261009-v2-01/answers)。

30 题均取得真实回答。重排成功 29/30，`rag-v2-008-1` 为实际失败回退，未称为重排成功。29 题取得有效模型审核，均被模型判断事实有支持，其中条件齐备 25/29；`001-0、005-0、007-0、011-0` 遗漏原订单申请人、订单号和完整操作记录。代理复核确认这四题正确来源的条件根本没有进入候选 Top-10，不能将只回答正确核心结论等同完整答复。

`rag-v2-002-0` 的真实审核返回空文本，解析为 JSONDecodeError，回答和 audit_raw 均保留，`backup_answers` 阶段仍为 failed。代理复核其真实回答：E611 操作、申请人/材料、齐全后 3 工作日、11 自然日和故障例外均有上下文支持；该结论不补写成模型审核成功。

6 道无答案题的模型审核和代理复核均未发现目标事实编造，实际回答明确外币汇率、运输丢件赔付倍数和海外仓城市未规定。所有题仍有检索候选，没有把非空检索算作误答。代理另发现 `absent-04-0` 附带流程摘要将“经确认的系统故障”简写为“系统故障”，遗漏确认限定；模型审核未识别这一措辞问题。争议与无答案复核共 11 题，见 [`answer_agent_review.md`](../artifacts/rag-optimization-v2/20261009-v2-01/answer_agent_review.md)，明确为代理复核而非人工。这些材料不支持“全部回答条件完整”或“主备质量相同”的结论。

实际组件验证使用隔离 Persistent Chroma 与真实 MiniLM，固定测试前 20 题经过 KnowledgeBase/工具链。主库只写入已取得的真实 Gemini 文档向量，未取得的文档保留 pending；整个知识库走完整备用路径，未混合向量空间。最终 60 文档、主片段 111、备用片段 466、pending_main=29、pending_backup=0。

历史20题实际备用Top5来源Recall@5=1.0000、Precision@3=0.3333、首次相关来源MRR=1.0000；仅来源计分，不以返回相关来源保证回答条件完整。实际备用查询 n=20，P50 187.66ms、P95 213.74ms；该路径因 pending 直接选备用，不含 Google 网络查询等待，不是主路延迟。20 次 `/health` 预热观测 P50 0.972ms、P95 1.285ms，最大心跳间隔 12.44ms。端点使用真实知识库和 Chroma，编排器统计为固定就绪替身，这只是本机轻量 ASGI 观测；同步阻塞正确性另由事件控制测试证明。

Google 额度停止状态下新增“七天受理”、更新为“九天受理，须订单号”：新内容进入备用、旧内容消失、pending 保持备用；重启 pending 不丢失；删除后主存储和备用检索都不含该来源。九项实际生产检查通过，结果与失败/降级日志见 [`backup_production.json`](../artifacts/rag-optimization-v2/20261009-v2-01/backup_production.json)。另在单文档完整主库注入同步 Google 延迟，以本轮缓存的真实 Gemini 文档向量和真实 MiniLM 执行兜底，252.13ms 返回含 E610 的正文，标为故障注入而非真实 Google 超时。两题实际查询改写及重排工具路径成功，不混入上述离线主指标。

20 题工具缓存均为 miss；追加真实重复查询和显式失效验证中，降级结果始终 `cached=false`，符合生产不缓存 degraded 结果、允许后续恢复的策略。首次追加实验错误地预期降级命中，失败快照保留，修正的是实验预期。主路径的真实缓存命中、Google 恢复、显式补齐后主路径检索新内容仍未运行；完整故障/恢复序列的契约测试通过，不能把它写成已完成真实 API 恢复。参见 [`production_cache.json`](../artifacts/rag-optimization-v2/20261009-v2-01/production_cache.json)。

## API 用量、失败与剩余边界

初始估计 Google 编码文本约 2500–4500、HTTP 约 6000–16000、LLM 顶层调用至多约 522；硬上限分别 10000/20000/600，持久账本跨恢复累计。

Google 已尝试 **650 HTTP**：640 countTokens、2 次能力信息、8 次 embedding 批请求。尝试编码文本 **128**，实际成功保存 **112 个真实 768 维文档向量**，失败批次 16 文本；未成功取得主查询或 CLASSIFICATION 向量。Google 未返回编码 token 用量，账本 0 表示未报告，不能解读成免费或零 token。限速等待累计 **613.635 秒**，服务端 retry 等待为 0（每日限制后直接停止）；保存的计数、向量与排名直接按内容/配置复用，没有新增 checksum。

LLM 顶层网关调用 **343/600**：228 道正式意图 +20 道歧义诊断 +1 次错误模型名失败，90 次备用重排/回答/审核，4 次生产改写/重排。342 次返回用量合计 input_tokens **223440**、output_tokens **379070**、cache_read_tokens **56190**，无报告的 cache_write 用量；这些按返回字段列出，不擅自按未知定价换算金额。早期分类和回答调用的 SDK 重试标志未全部采集，保留日志和合并 token，不估造底层 HTTP 次数。Google 仍有 9872 文本/19350 HTTP 授权余额，LLM 有 257 次余额；障碍是外部每日配额。

后续至少还需 180 道主查询编码、约 282 条模板/分类文本以及剩余分块文档编码，具体取决于开发选择；真实模板延迟测量、未重复的主回答组和生产改写预计还需约 100–190 次顶层 LLM（主当前与候选相同时复用），已存分类输出全部复用。必要需求仍在原预算内。新增复用计数从采集启用后累计观测到文档向量 112 次、计数 16 次，早期未采集的复用不补造；完整持久缓存和每次阶段用量均保留。

外部错误为 `HTTP_429 quota=EmbedContentRequestsPerDayPerProjectPerModel-FreeTier retry=37021s`，并非达到本轮授权调用上限。故障与依赖状态分别保存在 [`api_failures.json`](../artifacts/rag-optimization-v2/20261009-v2-01/api_failures.json)、[`attempts/`](../artifacts/rag-optimization-v2/20261009-v2-01/attempts) 与 [`manifest.json`](../artifacts/rag-optimization-v2/20261009-v2-01/manifest.json)。默认恢复不会反复请求已保存的每日额度故障；只有显式 `--retry-google` 才解除停止状态，仍受原累计上限约束。

原聊天配置名 `deepseek-v4.1-flash` 被实际网关以 HTTP 400 拒绝，返回支持 `deepseek-flash / deepseek-v4-pro`。本轮使用第一轮已验证的 `deepseek-flash` 进程配置覆盖，真实 `.env` 保持原样，旧失败保存于 [`llm_failures.json`](../artifacts/rag-optimization-v2/20261009-v2-01/llm_failures.json) 和 `failed_intent/`。顶层调用与返回 token 在 `api_usage.json`、`llm_usage.json` 保存；SDK 内部未暴露的重试请求数量未知，不能伪称完整供应商账单。返回 token 按网关契约已合并完整响应重试。

离线表的 `vector_build_wall_ms` 是该排名批次共同准备文档矩阵的墙钟时间，会出现在同批 BM25 汇总中；不是 BM25 额外编码成本，也不能跨组累加。独立 BM25 路径与保存块路径均由零 embedding 测试验证。排名复用与实际编码字段仅对采集到的调用报告，不回填未测历史延迟。

尚未覆盖：Google 主路正式质量/分块/overlap/融合；三路意图实际对照与真实模板缓存启动成本；主回答质量；真实 Google 配额恢复后的补齐/主路径新内容检索；HTTP Chroma、Redis、全部署负载；真实用户语料与人工审核。控制事件和 stub 仅用于并发、契约与故障注入，不计入模型质量结果。Google 恢复后继续已有断点即可，无需扩展费用授权。

## 复现与交付文件

在仓库根目录运行工程验证：

```powershell
& D:\anaconda3\envs\learn_claude\python.exe -m pytest -q -p no:cacheprovider
```

在 Python 项目目录运行实验：

```powershell
Set-Location D:\Pythonprojects\yan_1\agent\mymind\mymind
$env:ANTHROPIC_MODEL='deepseek-flash'
$taskRun='artifacts/rag-optimization-v2/20261009-v2-01'
& D:\anaconda3\envs\learn_claude\python.exe -m experiments.optimization_v2 --stage local --output $taskRun --resume
# Google 每日额度恢复后，继续完整方案；仍复用相同分块、向量与真实 LLM 输出
& D:\anaconda3\envs\learn_claude\python.exe -m experiments.optimization_v2 --stage all --output $taskRun --resume --retry-google
& D:\anaconda3\envs\learn_claude\python.exe -m experiments.summarize_optimization_v2 --output $taskRun
```

失败/blocked 阶段返回非零退出码；不应将它改成成功状态。`--stage` 支持方案固定阶段和独立本地续作阶段，包括 `production_cache`；依赖、预算与测试前选择在 [`optimization_v2.py`](optimization_v2.py) 中执行。`--stage local --resume` 会复用已完成组，并重试失败的回答审核；如果仅查看已有结果，直接运行汇总命令即可。数据生成器为 [`build_optimization_v2.py`](../data/eval/build_optimization_v2.py)，已经冻结的运行不应重新生成后继续旧结果。当前使用说明见 [`rag.md`](../docs/rag.md)。

核心交付：实现和测试；两份新数据；`selection.json`；逐题 `retrieval/`、`rankings/`、`intent_llm_raw_*.json`、`answers/`；计数/向量复用缓存；`baseline.json`、`manifest.json`、`attempts/`、API 失败/用量；隔离 Chroma 集成产物；汇总脚本和本报告。每批改动记录在根目录 `report.md`。

## 可用于求职讲述的结论

“我修复了 RAG 的嵌套超时问题，把 Google 查询所有阶段纳入统一子预算，通过完整工具与 Agent 路径验证慢计数、慢编码和解析阻塞后仍返回备用知识；同时把远程索引准备移出状态锁，验证统计端点、并发导入结果归属与更新缓存一致性。”这一结论有本轮 142 项测试和受控故障证据。

“我建立了按问题家族冻结的合成评测和持久调用账本，发现备用等权混合在测试 Coverage@5=0.87，而 BM25=0.95，能追溯正确条件证据如何被弱向量结果挤出。按预先选择规则保留原参数，诚实报告 Google 每日额度导致的主路未完成部分。”可以说明评测设计与负结果分析；不能声称召回提升、三路意图增益、最佳 overlap 或真实用户覆盖。
