# Qwen 与 MiniLM 双索引 RAG

生产入口为 `api/main.py` 与 `mcp/indexed_knowledge_base.py`。当前 embedding 为 `qwen3.7-text-embedding-flash`，768 维；聊天 LLM 独立配置。接入记录见 [迁移报告](../experiments/QWEN_EMBEDDING_MIGRATION_REPORT.md)，正式效果与验收见 [Qwen 第二轮报告](../experiments/QWEN_RAG_OPTIMIZATION_REPORT.md)，命令见 [实验说明](../experiments/README.md)。

## 配置与启动

在 Python 后端 `.env` 配置，密钥只保存在本机：

```dotenv
EMBEDDING_API_KEY=在本机填写
EMBEDDING_MODEL=qwen3.7-text-embedding-flash
EMBEDDING_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1
EMBEDDING_DIMENSIONS=768
RAG_CHUNK_TOKENS=512
RAG_OVERLAP_TOKENS=0
RAG_EMBEDDING_QUERY_BUDGET_S=8
RAG_MAIN_VECTOR_WEIGHT=0.5
RAG_BACKUP_VECTOR_WEIGHT=0.5
INTENT_TEMPLATE_CACHE_PATH=./data/intent_templates.json
RAG_RERANK_MAX_CHARS=24000
RAG_RERANK_MAX_TOKENS=4096
RAG_ONNX_PATH=./data/onnx_models
```

Qwen 向量接口为 `/compatible-mode/v1/embeddings`；`/apps/anthropic` 用于聊天 Messages。聊天模型仍由 `LLM_PROVIDER/LLM_MODEL/LLM_API_KEY/LLM_BASE_URL` 或原有 `ANTHROPIC_*` 配置决定。[官方 embedding 接口](https://platform.qianwenai.com/docs/api-reference/text-embedding/openai-embedding)。

从 `mymind` 目录运行已有依赖及 Python 后端：

```powershell
docker compose up -d redis chromadb
& D:\anaconda3\envs\learn_claude\python.exe -m api.main
```

本机通常使用 Chroma `localhost:8001`、Redis `localhost:6379`；容器内地址由 Compose 配置。MiniLM 在 Python 客户端加载与预热，HTTP Chroma 仍需要本地模型文件。模型项为空时使用 MiniLM + BM25，意图使用 85/15；指定 embedding 模型但缺密钥时启动报错。

## 索引与检索

主库为 `knowledge_qwen3_7_text_embedding_flash_768_v1`，备用为 `knowledge_minilm_v1`；原文和就绪状态保存在 `CHROMA_PERSIST_DIRECTORY/knowledge_sources.sqlite3`，与 Chroma 数据一并保存。Qwen 启动会删除同知识库前缀下所有维度的旧 Gemini v1 collection，记录 `cleanup:removed`；其他命名范围的实验库、原 `knowledge_base`、情景记忆与用户画像不变。

正常采用 Qwen + 对应 BM25，Qwen 查询失败采用独立 MiniLM + 对应 BM25。主备各自使用自己的片段、文档向量与查询向量，不混合空间。主库未覆盖全部最新文档时继续完整备用路径，显式补齐成功后恢复主路。查询暂时失败且主库完整时，下一次成功查询即可恢复。

主分块为结构递归、512 个服务计量 token、overlap 0。计数方式为 `qwen_single_input_usage_tokens`：对标题/章节与正文发送单文本真实 embedding 请求，读取该输入的 `usage.total_tokens`，另留 16 token 余量；相同文本的计数和返回向量缓存复用，不重复编码已计数的文档块。计数本身有 API 和编码成本，缺少用量字段时明确失败。超出接口单次输入限制的长章节先递归拆分，再校验预算。查询与分类仍编码原文，响应按 index 对齐并归一化。批次最多 20 条；128000 总输入限制用保守 UTF-8 字节上界拆批，批次保护不替代分块的真实 token 计量。[模型说明](https://platform.qianwenai.com/docs/developer-guides/embeddings/embedding)、[用量字段](https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api)。

计量方式属于主索引签名。原 Qwen 字节块在新代码中转为待补齐，保留完整备用路径；调用 `/knowledge/repair` 后以真实 token 重新分块并恢复主路。数值 512 保持，但含义已经由字节改为服务 token，已有业务库需要显式补齐。

MiniLM 固定 240 token、overlap 0，用自己的完整 tokenizer 计入标题及特殊 token；主分块调整不会重建备用。当前主备向量权重均为 0.5，RRF k=60。开发集比较 overlap 0/32/64 后保留 0：原Recall与Precision相同，正 overlap 增加重复和上下文；正式测试只运行冻结配置。

## 导入、更新、删除与补齐

支持 `.md`、`.txt` 和 title/content JSON 数组。Markdown 保留章节结构，TXT 不识别 Markdown 标题。同批相同标题按出现顺序区分来源；后续按返回的 `source_id` 更新可避免重命名产生另一来源。

```powershell
$body=@{documents=@(@{title='公开测试政策';source_id='public-example';content='# 退款' + "`n" + '七天内可申请退款。';format='md'})} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:8000/knowledge/add -Method Post -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
curl.exe -F 'file=@政策.md' http://localhost:8000/knowledge/upload
Invoke-RestMethod http://localhost:8000/knowledge/stats
Invoke-RestMethod -Method Delete http://localhost:8000/knowledge/documents/public-example
Invoke-RestMethod -Method Post http://localhost:8000/knowledge/repair
Invoke-RestMethod -Method Post 'http://localhost:8000/search?query=退款期限&top_k=5'
```

导入返回本请求的 `success/degraded/partial/failed`、逐来源结果和 `pending_main`。Qwen 导入失败但备用成功，新正文立即可从备用检索；主库待补齐期间不会返回过时主正文。备用也失败时保留原文，准确报告失败或批次部分完成。补齐会把保存的原文发送给当前 embedding 供应商。本次验收在逐条确认服务内旧6篇文档与仓库内置公开文档完全相同后完成其主备索引；私人内容未用于外部实验。

索引更新、删除和补齐会失效工具缓存；在途旧查询不会回填新缓存，降级结果不跨请求缓存。远程编码不持索引状态锁，统计端点在线程执行同步工作。`/knowledge/stats` 与 `/monitor` 可查看待补齐、失败和恢复状态。

## 意图、超时与重排

意图常规融合固定 LLM/embedding/关键词 70/20/10；embedding 未配置或失败时 85/15，没有 n-gram 或 embedding 提前终止。模板缓存绑定模型、维度、endpoint、任务及内容；准备共享受管理任务，在线最多等待 3 秒，独立准备最多 180 秒。重启命中不重复编码，准备的冷成本仍计入用量。

主 embedding 在线查询共享 8 秒总预算，覆盖计量、等待与编码；工具总预算 30 秒，主预算最多 20 秒以留备用时间。导入与模板准备独立执行。重排提供完整候选编号、标题、章节和正文，超出总字符预算时整条省略；失败记录后回退原排序，缺失编号由原排序补足。

## 验证与正式实验

从仓库根目录运行 `learn_claude` 回归：

```powershell
& D:\anaconda3\envs\learn_claude\python.exe -m pytest -q -p no:cacheprovider
```

从 `mymind` 目录验证真实模型接入，以下会产生 Qwen API 调用，使用新输出目录测冷成本：

```powershell
$taskPython='D:\anaconda3\envs\learn_claude\python.exe'
& $taskPython -m experiments.probe_qwen_embedding --output artifacts/qwen-check/availability.json
& $taskPython -m experiments.qwen_embedding_smoke --output artifacts/qwen-check/isolated
& $taskPython -m experiments.qwen_embedding_smoke --chroma-host localhost --chroma-port 8001 --redis-db 15 --output artifacts/qwen-check/http
```

smoke 验证真实向量、Chroma/工具、更新删除、备用和恢复，以及模板冷准备/重启缓存，未调用聊天 LLM，不代表质量对比。HTTP 模式使用独立 collection 与本地原文目录；Redis 使用隔离库和前缀，不操作用户知识。

四项工程修复及当前全量160项回归通过。独立 Qwen 运行已完成冻结数据校验、开发选择、120题正式检索、152题两路/三路意图、预选30题主备回答，以及实际知识库/工具与完整应用验收。检索恢复原来源算法，120题总体主/备用Recall@5/10均100%、Precision@3均27.78%；来源MRR为0.8333/0.7917。20道无答案题也按原规则参与平均，非空候选率100%只反映检索，不能解释为回答编造。三路意图仅比两路多改对1题，开发集更差，不能据此称稳定提升。主备等权及固定意图融合保持。回答为真实模型生成与模型审核，另有代理复核，无人工审核；主/备用30题模型事实支持均30/30，可答题条件完整24/24与20/24，无答案编造各0/6、引用错误各0/30；备用4次条件遗漏保留，原1次审核失败补跑成功且原失败保存。

从 `mymind` 目录顺序执行，以下使用真实模型并产生调用费用；`application` 要求既有 HTTP Chroma 与 Redis 可用，且当前业务库只能是已确认的内置公开6篇文档：

```powershell
$taskPython='D:\anaconda3\envs\learn_claude\python.exe'
$taskOutput='artifacts/qwen-optimization-v2/reproduce'
& $taskPython -m experiments.qwen_optimization --stage all --output $taskOutput
& $taskPython -m experiments.qwen_optimization --stage byte_baseline --output $taskOutput
& $taskPython -m experiments.summarize_optimization_v2 --output $taskOutput
```

`all` 依次完成 validate、smoke、chunking、overlap、fusion、intent_dev、select、test、intent_test、answers、production、application；`--stage` 可选单阶段，`--resume` 跳过已经完成的阶段。`byte_baseline` 是原字节512的固定对照，不参与选择。新目录会重新运行实际 API；入口累计扣除历史685次embedding HTTP、264条尝试编码和343次LLM网关调用，受20000/10000/600上限约束。它复用匹配的历史真实意图输出和MiniLM结果；预算不足保留blocked与断点，不能把新目录当作额度重置。

旧 `experiments.optimization_v2` 是 Gemini 实验入口，不能直接把 Qwen 向量写入旧运行或用当前密钥续跑 Google 阶段。独立运行清单、费用、逐题上下文和已知局限见 [第二轮报告](../experiments/QWEN_RAG_OPTIMIZATION_REPORT.md)。冻结数据为合成政策，主路部分指标饱和，不能推广为真实客服泛化或生产负载保证。
