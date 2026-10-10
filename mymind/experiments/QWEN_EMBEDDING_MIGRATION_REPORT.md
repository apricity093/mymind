# Qwen embedding 接入与迁移验证

日期：2026-10-09。模型：`qwen3.7-text-embedding-flash`，768 维。

模型和正确的 embedding 接口已经真实验证，Python 后端与本地 `.env` 已切换。全量 `learn_claude` 回归 **150 passed / 19.40s**，包含本次新增 8 项测试；隔离生产组件真实 smoke 的 13 项检查全部通过。本次验证只发送公开合成文本，未上传用户现有知识库，未调用聊天 LLM，没有把连通性验证当作质量提升。

## 接口与配置

用户提供的 `https://maas.qianwenaiapi.com/apps/anthropic` 下，`/embeddings` 和 `/v1/embeddings` 均返回 HTTP 404。该基地址用于 Anthropic Messages；官方 embedding 文档指定 OpenAI 兼容基地址 `https://maas.qianwenaiapi.com/compatible-mode/v1`。在同一域名使用现有凭证调用 `/embeddings` 实际返回 HTTP 200，响应模型为指定的 Qwen Flash，两条输入均返回有效 768 维向量，报告 48 token，耗时 189.9ms。[官方 embedding 接口](https://platform.qianwenai.com/docs/api-reference/text-embedding/openai-embedding)、[Anthropic 聊天接口](https://platform.qianwenai.com/docs/api-reference/chat/anthropic)。

本地 `.env` 的 `GEMINI_API_KEY` 改名为 `EMBEDDING_API_KEY`，密钥值逐值验证保持不变，没有写入报告或终端输出。配置如下：

```dotenv
EMBEDDING_MODEL=qwen3.7-text-embedding-flash
EMBEDDING_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
EMBEDDING_DIMENSIONS=768
RAG_EMBEDDING_QUERY_BUDGET_S=8
```

预算变量从 `RAG_GOOGLE_QUERY_BUDGET_S` 改为通用名称。聊天配置保持独立，其他本地配置值没有改变。`.env` 仍不纳入 Git。

## 实现与索引行为

- `core/embedding.py` 提供 Qwen OpenAI 协议接入、Bearer 鉴权、按响应 index 对齐、维度/有限值校验和单位归一化。批次最多 20 条，并按保守总输入预算拆批，实际 token 用量读取服务响应。旧 Gemini 接入保留，Gemini 实验会拒绝误用当前 Qwen 配置。[官方模型与批次说明](https://platform.qianwenai.com/docs/developer-guides/embeddings/embedding)。
- `api/main.py`、`core/intent_recognizer.py` 与 `mcp/indexed_knowledge_base.py` 使用同一模型工厂。Qwen 主库为 `knowledge_qwen3_7_text_embedding_flash_768_v1`，片段模型元数据与实际模型一致；同维度的 Gemini 向量不会复用到 Qwen。根据 2026-10-10 后续授权，Qwen 启动时删除同知识库命名范围内全部维度的 Gemini v1 collection；原文和 `knowledge_minilm_v1` 保留，其他命名范围的历史实验 collection 不变。
- 已有原文在模型签名变化后标为主库待补齐。主库完整之前继续使用完整 MiniLM + BM25，显式补齐完成后切回 Qwen + BM25。失败、降级、恢复记录标明 Qwen 模型；更新、删除与补齐使缓存失效，降级结果仍不跨请求缓存。
- 主 embedding 查询计量、等待和编码共享 8 秒总预算，保留现有工具 30 秒预算及备用时间。模板缓存绑定模型、维度、endpoint、任务与模板内容，旧 Gemini 模板不会误命中。
- 意图权重仍为 LLM/embedding/关键词 70/20/10，失败时 85/15；主备向量融合权重均保持 0.5，备用分块固定 240 token / overlap 0。主分块预算数值 512 / overlap 0 没有调参。

Qwen OpenAI 接口统一发送文本，文档标题与章节路径拼入正文，查询和分类直接编码原文，不发送 Google taskType。本次没有匹配该具体模型的精确 tokenizer，主分块明确使用 `utf8_byte_upper_bound`。512 是保守字节预算，不是精确的 512 个 Qwen token，中文片段可能比原 Gemini 片段更短；`chunk_config.main_counting` 和真实用量分别记录计量方式与服务 token。这一分块方式尚未做独立质量对比。

## 验证结果

新增 `tests/test_qwen_embedding.py` 覆盖配置与旧实验隔离、真实协议字段与标题、响应顺序、批次/输入预算、无效向量、同维度模型切换独立索引、备用保持与显式补齐，以及完整工具链超时返回备用正文。stub 只用于契约、状态与故障注入。首次聚焦收集因既有 fixture 未显式导入而报错，修正后新增 8 项通过；随后全量 150 项通过。

真实 smoke 使用公开退款、配送和发票政策，独立 Persistent Chroma、真实 Qwen 文档/查询向量及真实本地 MiniLM。通过的 13 项检查：

1. Qwen 文档导入成功。
2. 退款查询经过工具返回 Qwen 主库证据。
3. 配送查询经过工具返回 Qwen 主库证据。
4. 发票查询经过工具返回 Qwen 主库证据。
5. 主路径工具缓存命中。
6. 注入供应商失败后返回真实 MiniLM 的七天退款正文。
7. 解除故障后恢复 Qwen 主路径。
8. 主导入失败的更新仍从备用返回新九天规则。
9. 重启保留待补齐状态。
10. 显式补齐后 Qwen 主库返回新九天正文。
11. 删除后当前主备都移除该来源。
12. 真实意图模板重启命中本地缓存，零新增 HTTP。
13. 三条意图消息的真实 embedding 分支均正常完成。

最后一项仅验证 embedding 分支，未调用 LLM，不代表完整三路分类质量。三条公开消息得到 order_status、refund、technical_login；没有据此计算正式准确率。

| 意图 embedding 测量 | 实际结果 |
| --- | --- |
| 冷模板准备 | 950.17ms，54 条模板，3 次 HTTP |
| 重启缓存读取 | 61.47ms，0 次 HTTP |
| 热查询 | 3 条，84.54 / 86.46 / 94.86ms，P50 86.46ms |

冷模板成本计入总用量，没有因移到准备阶段而扣除。延迟来自本机本次样本，不作为稳定 SLA 或与 Gemini 的性能对比。

## 实际 API 用量

| 阶段 | HTTP | 确认成功编码文本 | 服务报告 token |
| --- | ---: | ---: | ---: |
| 给定 Anthropic 路径探测（两次 404） | 2 | 0 | 未返回 |
| 正确 embedding 路径探测 | 1 | 2 | 48 |
| 真实生产组件及意图缓存 smoke | 16 | 67 | 1408 |
| 合计 | **19** | **69** | **1456** |

17 次 HTTP 成功。smoke 文本分为文档 4、查询 6、分类 57（54 模板 + 3 消息）。两个 404 请求各提交两条探测文本，但没有成功编码证据，不加入 69 条。聊天 LLM、Google HTTP 均为 0。本次接入账目独立于先前 Gemini 优化实验，旧失败与用量没有被覆盖。返回 token 是服务响应值，404 是否产生其他账单费用未获服务确认。

## 使用与复现

重启 Python 后端后配置生效。已有知识使用备用，需在本机明确运行下面的补齐操作，才会发送保存原文到 Qwen 并恢复主路；本次只验证了公开隔离索引，未对用户现有知识库自动执行补齐。

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/knowledge/repair
Invoke-RestMethod http://localhost:8000/knowledge/stats
```

测试从仓库根目录执行：

```powershell
New-Item -ItemType Directory -Path mymind/artifacts/qwen-embedding-migration/reproduce -Force | Out-Null
& D:\anaconda3\envs\learn_claude\python.exe -m pytest -q -p no:cacheprovider --basetemp mymind/artifacts/qwen-embedding-migration/reproduce/pytest --junitxml mymind/artifacts/qwen-embedding-migration/reproduce/tests.xml
```

以下命令从 `mymind` 目录执行，会使用本地凭证产生实际 embedding 调用。冷测量使用新输出目录；重复使用已有模板缓存时测到的是缓存读取。

```powershell
$taskPython='D:\anaconda3\envs\learn_claude\python.exe'
& $taskPython -m experiments.probe_qwen_embedding --output artifacts/qwen-embedding-migration/reproduce/availability.json
& $taskPython -m experiments.qwen_embedding_smoke --output artifacts/qwen-embedding-migration/reproduce/real-smoke
```

复现原 404 路径须显式传 `--base-url https://maas.qianwenaiapi.com/apps/anthropic`，探测退出码为 1。使用说明见 [RAG 配置与补齐](../docs/rag.md)。

## 原始记录与边界

- [给定路径两次 404](../artifacts/qwen-embedding-migration/availability.json)
- [正确路径真实向量可用性](../artifacts/qwen-embedding-migration/availability-compatible.json)
- [真实 smoke 逐查询正文、恢复、缓存与用量](../artifacts/qwen-embedding-migration/real-smoke/smoke.json)
- [全量测试 JUnit](../artifacts/qwen-embedding-migration/tests.xml)
- [本次新增测试](../tests/test_qwen_embedding.py)

本次证明指定模型/768维/正确接口可用，以及当前后端的索引、工具、备用与模板缓存接线正常。尚未完成 Qwen 独立检索质量、正式两路/三路意图、回答质量或分块与融合选择实验。先前 [Gemini 第二轮报告](RAG_OPTIMIZATION_V2_REPORT.md) 的 Google 配额阻塞和模型质量边界继续成立；旧评测数据、选择记录、实验向量与逐题结果保持原样，不可作为 Qwen 的质量结论。

## 2026-10-10 Gemini 生产索引清理

按后续授权不再保留 Gemini 生产向量索引。清理限定当前 `collection_prefix` 的 `<前缀>_gemini_001_<维度>_v1`，在 Qwen 启动并准备备用后执行，覆盖旧维度并刷新缓存，记录被移除的 collection 名称。再次启动没有旧索引时不重复删除。

首次清理检查时，本机配置为 `localhost:8001`，本地回退目录为 `mymind/data/chroma`。本地实际只有 `knowledge_base`、`episodic`、`user_profile`，没有 Gemini collection；当时 8001 连接失败，没有声称已删除服务内索引。该次未更改 `.env`、上传原文或调用模型 API。[实际存储检查](../artifacts/qwen-embedding-migration/gemini-cleanup/storage-inspection.json)保存当时本地 collection 和服务不可用状态。后续 Docker 恢复后的实际验证见下节。

本次清理修改后 `learn_claude` 全量 **150 passed / 25.53s**。模型切换测试使用真实 Chroma API，验证同命名范围两个旧维度均移除、原文与备用保持、无关历史 collection 保留、显式补齐与再次启动正常。首轮因临时目录父目录未创建产生 setup 错误，未进入测试；补建目录后复跑通过。保留 [首轮 setup 记录](../artifacts/qwen-embedding-migration/gemini-cleanup/tests-setup-failed.xml) 与 [本次全量通过记录](../artifacts/qwen-embedding-migration/gemini-cleanup/tests.xml)。未重复运行真实模型 smoke，本次模型 API 用量为 0。

## 2026-10-10 HTTP Chroma 与 Redis 验证

用户开启 Docker 后，Redis 正常，既有 `mymind-chromadb` 仍停止。本次启动已有 `chromadb/chroma:0.5.23` 容器，使用原持久卷和 8001 端口，没有构建或部署应用。HTTP heartbeat 返回 200。服务中实际只有 `knowledge_base`、`episodic`、`user_profile`，没有 Gemini 生产 collection，无需删除实际生产目标。

扩展 `experiments.qwen_embedding_smoke` 支持 HTTP Chroma 和非生产 Redis 库。在独立 collection 前缀、独立原文 SQLite 与 Redis DB15 随机前缀内，用公开三篇政策及真实 Qwen/MiniLM 执行。18 项检查全部通过：原 13 项接入/更新/删除/补齐/缓存检查，另加 HTTP 删除两种维度的受控空 Gemini collection、Redis 跨实例命中、更新使另一实例失效、原服务 collection 保持、实验 Redis key 清理。受控旧 collection 不包含假向量，不作为质量证据。

结束时删除本次自身 HTTP collection 和 Redis 前缀。用户已有 collection 名称与片段数前后完全相同：`knowledge_base=6`、`episodic=0`、`user_profile=0`。逐查询正文、缓存结果、故障恢复、服务数量及用量在 [HTTP smoke 原始结果](../artifacts/qwen-embedding-migration/http-smoke-20261010/smoke.json)。本次实现路径为客户端 `KnowledgeBase → HTTP Chroma` 与 `MCPToolManager → RedisCacheStore`，未启动完整聊天应用或审核回答。

实际新增 **16 次成功 Qwen HTTP、67 条编码文本、1408 服务报告 token、0 次聊天 LLM**。冷模板准备 54 条、1173.54ms；缓存重启 55.98ms、0 HTTP；3 热消息 embedding P50 89.53ms。含前次迁移探测与本地 smoke，Qwen 接入验证累计 **35 HTTP（33 成功，2 个原路径404）、136 确认编码文本、2864 token**。历史 Gemini/LLM 实验账目未覆盖或改写。

复现命令从 `mymind` 目录执行，选择新输出目录：

```powershell
& D:\anaconda3\envs\learn_claude\python.exe -m experiments.qwen_embedding_smoke --chroma-host localhost --chroma-port 8001 --redis-db 15 --output artifacts/qwen-embedding-migration/reproduce/http-smoke
```

当前代码最近的全量回归为 150 项通过；本次脚本用真实组件运行验证，编译与文档本地链接检查通过，没有为文档删除重复全量测试。启动日志有 `ClientStartEvent` 遥测 `capture()` 参数告警，本次读写与 18 项检查均通过；日志保留，尚未调整遥测依赖。

## 接入阶段结束时的剩余工作

1. **Qwen 计量与检索选择**：512 仍为明确标记的保守 UTF-8 字节预算，未接入已验证的精确 tokenizer；尚未完成当前模型的 256/512/768、两种递归、overlap 0/32/64、主融合权重开发选择，以及 180 题正式检索。旧 Gemini 的块和向量不能作 Qwen 对照或写回同一旧运行。
2. **完整意图对照**：228 题开发/测试的 Qwen 两路/三路配对、模板/阈值开发选择和改对改错尚未完成。应复用同题已保存真实 LLM 输出，保持 70/20/10 与 85/15；54 模板准备和 3 热消息测量不能代替分类质量或整条并行请求延迟。
3. **回答质量**：仍需预选 30 题在 Qwen 当前/候选与备用上下文下，使用同一聊天模型做回答及事实条件审核。历史备用组有 1 次审核失败、4 次条件遗漏，仍保留失败；它们不能成为 Qwen 回答的通过证据。当前本机聊天模型读到 `deepseek-flash`，旧模型名配置错误已不适用于当前文件。
4. **真实业务库与应用验收**：本次只测试公开隔离文档。实际服务仍仅有旧 `knowledge_base` 6 个片段，没有真实业务 Qwen 主库与新 MiniLM 双索引，也没有完整应用启动、实际 `/knowledge/...` 与聊天 Agent 端到端验收。需要在允许提交的业务原文范围内完成导入/补齐；不能从旧零碎片段假定已恢复完整原文。完整应用负载与人工审核也未覆盖。

HTTP Chroma 与 Redis 连接边界已得到本次证据，四项工程修复的当前测试仍通过；以上效果与业务验收项目继续保留未完成状态。当前说明集中在 [RAG 使用说明](../docs/rag.md)，过期首轮设计、Gemini 命名使用说明与重复执行提示已移除，实测报告和冻结方案保留。

## 2026-10-10 第二轮补齐

上述列表为接入smoke结束时的状态。后续已完成独立Qwen开发选择、正式检索、完整意图配对、主备回答与实际应用验收，详见 [Qwen第二轮正式报告](QWEN_RAG_OPTIMIZATION_REPORT.md) 及其原始运行清单；历史接入测量不覆盖或替代正式实验。

当前分块使用 `qwen_single_input_usage_tokens`：单文本真实embedding响应的 `usage.total_tokens` 精确计量标题/章节与正文，计数本身有编码与HTTP成本，同文本返回向量复用于文档。原512字节配置仅作固定对照，生产数值512保持且单位改为服务token；索引签名含计量方式，旧字节块待显式补齐期间保持备用。MiniLM240/0及固定融合方式不变。

最终learn_claude全量156项通过（24.17s）。在确认旧knowledge_base的6篇原文与仓库内置公开文档完全相同后，本机业务Qwen主库和MiniLM备用均已完整；原knowledge_base、episodic及user_profile数量保持，未发送私人内容。实际应用的主路与延迟故障备用Agent工具、并发导入与监控已通过。Gemini生产collection当前不存在，无需声称删除未存在的生产对象。
