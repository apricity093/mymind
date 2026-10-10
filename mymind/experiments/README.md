# Cache, Memory and RAG Experiments

Run commands from the `mymind/` directory with the project on `PYTHONPATH`.

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer offline
```

## RAG retrieval experiment (check.md R0-R4)

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer rag
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer rag --variants r1 r4 --top-k 10
```

The RAG layer rebuilds every variant from `data/eval/rag_corpus.json` into an
independent, versioned collection, evaluates `data/eval/rag_dataset.json`
(128 queries: 8 no-answer calibration + 120 test) with the result cache
disabled, and reports recall/ranking/fact-coverage/no-answer/duplicate metrics
plus paired bootstrap CIs, cold/hot latency percentiles and call counters.

真实 Chroma 默认 all-MiniLM-L6-v2 层（rewrite/rerank 仍为确定性代理）：

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.rag_chroma --variants r1 r4 --top-k 10
```

Docker integration uses an isolated Redis database and temporary Chroma collections:

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer integration --redis-url redis://:mymind123@localhost:6379/15 --chroma-port 8001
```

The real-model layer is opt-in because it incurs API cost:

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer real --confirm-cost --provider deepseek --repeat 5 --cache-scenario stable-prefix
```

Supported real-model providers are `deepseek`, `openai`, and `anthropic`. The
cache scenario can be `identical`, `stable-prefix`, or `invalidation`. Existing
`ANTHROPIC_*` variables remain supported; provider-neutral deployments may use
`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, and `LLM_BASE_URL`.

Every layer writes timestamped JSON and Markdown reports under `artifacts/experiments/`.

## Agent tool migration experiment (E0–E5)

The deterministic migration experiment executes the same orchestrator code for all six feature variants against `data/eval/agent_migration_dataset.json`:

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer agent-migration
```

The dataset contains 240 cases (40 each for general, technical, billing, composite, escalation, and unauthorized/schema safety). Gold fields cover primary/supporting routes, required/allowed/forbidden tools, escalation, knowledge access, required facts, and forbidden claims. The report includes route/support accuracy, tool precision/recall/F1, unauthorized executions, forbidden claims, and latency percentiles.

The paid DeepSeek E0/E5 paired experiment is deliberately opt-in. It samples 10 cases from each category, repeats each variant three times (360 top-level requests), judges the first repeat with one paired E0/E5 call per case, and calculates a paired bootstrap 95% confidence interval. Agent calls, tool continuations, Composer calls, and 60 paired Judge calls share a hard 600-call budget; incomplete-response auto-retry is disabled for this experiment.

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer agent-migration-real --confirm-cost --provider deepseek
```

Keep the model, dataset order, temperatures, and judge prompt fixed when comparing reports. A deterministic pass proves contracts and policy wiring; it is not a substitute for the paid quality/latency/cost gate.

## Runtime RAG and evaluation semantics

The HTTP chat runtime enables tool use and `KNOWLEDGE_TOOL_MODE=tool_only` by default. Agents decide whether to call `search_knowledge_base`; the API does not pre-retrieve from intent labels. `disabled` removes the knowledge tool. Profile, escalation, Composer, internal Trace and Trace API remain disabled by default; this runtime configuration is not the full E5 variant.

R0–R4 retrieval experiments retain `KnowledgePolicy` as a deterministic benchmark gate and use their own retrieval variants. E0–E5 explicitly select their feature configurations. Runtime defaults do not silently enable RAG inside those Agent ablations, whose `knowledge_tool_mode` stays `disabled`.

The end-to-end evaluator keeps the `knowledge_gate_accuracy` field name but compares actual `search_knowledge_base` calls in `tools_used` with `expect_knowledge_search`. It measures tool selection, independently of retrieval quality or answer grounding. Interpret no-answer and quality metrics together with the dataset, embedding, rewrite/rerank implementation and executed variant.

The current runtime uses Qwen Flash/MiniLM. Configuration and index repair are documented in [RAG usage](../docs/rag.md). The historical Gemini/MiniLM experiments use `python -m experiments.gemini_rag` and require an explicit Gemini model and matching Google credential; the current Qwen configuration cannot reproduce those results.

Measured results, retained failed attempts and interpretation limits are in [Gemini RAG experiment report](GEMINI_RAG_REPORT.md).

## Qwen embedding migration

The verified OpenAI-compatible embedding base is `https://maas.qianwenaiapi.com/compatible-mode/v1`. The supplied `/apps/anthropic` base returned 404 for embedding routes. The migration keeps 768 dimensions and creates a separate main collection. Qwen startup removes Gemini collections in the same knowledge namespace, retaining source documents, MiniLM and historical experiment collections in other namespaces.

```powershell
$taskPython='D:\anaconda3\envs\learn_claude\python.exe'
& $taskPython -m experiments.probe_qwen_embedding --output artifacts/qwen-embedding-migration/reproduce/availability.json
& $taskPython -m experiments.qwen_embedding_smoke --output artifacts/qwen-embedding-migration/reproduce/real-smoke
& $taskPython -m experiments.qwen_embedding_smoke --chroma-host localhost --chroma-port 8001 --redis-db 15 --output artifacts/qwen-embedding-migration/reproduce/http-smoke
```

These commands make real API calls using the local embedding credential and public synthetic inputs. Choose a fresh output directory for a cold template measurement; an existing template file measures cache reuse. The smoke uses an isolated persistent Chroma index, real Qwen vectors, real MiniLM fallback and controlled provider faults. It does not call the chat LLM or measure retrieval/classification quality. Actual results and usage are in [Qwen migration report](QWEN_EMBEDDING_MIGRATION_REPORT.md).

HTTP smoke uses the existing Chroma service, a fresh collection namespace, a separate source SQLite file and Redis DB 15 with a unique cache prefix. It verifies deletion of controlled empty legacy collections, cross-worker cache hits and update invalidation, then removes only its own service collections and Redis keys. Existing collection names/counts are recorded before and after. It does not repair or encode user knowledge.

## Qwen second-round optimization

Run engineering tests in `learn_claude` first. From the Python backend directory:

```powershell
$taskPython='D:\anaconda3\envs\learn_claude\python.exe'
$taskOutput='artifacts/qwen-optimization-v2/reproduce'
& $taskPython -m experiments.qwen_optimization --stage all --output $taskOutput
& $taskPython -m experiments.qwen_optimization --stage byte_baseline --output $taskOutput
& $taskPython -m experiments.summarize_optimization_v2 --output $taskOutput
```

The pipeline validates frozen data, smokes three development questions, compares recursive/structural chunks at 256/512/768 service tokens, overlap 0/32/64, main and backup fusion, and intent thresholds. It writes `selection.json` before held-out retrieval/classification, then runs fixed answers and actual KnowledgeBase/Chroma/tool recovery. Identical groups and vectors are reused. The optional `byte_baseline` stage measures the former byte512 configuration after selection, without tuning from test results. API token counting is real embedding work, separately charged and recorded; its document vectors are reused.

`application` uses the actual FastAPI lifespan, native Agent tools, HTTP Chroma and Redis. It confirms that the six existing business documents exactly match the repository's public built-ins before repairing them, then removes only its temporary documents, memory collections and Redis prefix. It refuses unfamiliar business content. No external application deployment or private-data experiment is performed.

Stages can run individually with `--stage`; `--resume` skips completed stages and retains cumulative usage and failed attempts. The ledger deducts historical 685 embedding HTTP attempts, 264 attempted text inputs and 343 LLM gateway calls from the authorized 20000/10000/600 caps. Creating another output directory does not grant another budget; reproduction may need a newly authorized budget if the remaining allowance is insufficient. Matching 248 historical real LLM classifications and unchanged MiniLM results are reused, explicitly marked in raw outputs.

Measured selection, answers, family-cluster paired intervals, production and application boundaries are in [Qwen optimization report](QWEN_RAG_OPTIMIZATION_REPORT.md). The previous [Gemini second-round report](RAG_OPTIMIZATION_V2_REPORT.md) and `experiments.optimization_v2` remain historical evidence with their original failed/blocked states; they require Gemini configuration and cannot be resumed using Qwen credentials.

The 2026-10-07 regression run passed 86 tests, including the deterministic E0–E5 gate. API contract and Compose checks also passed. The earlier paid DeepSeek E0/E5 experiment failed quality, judge-completeness and token-cost gates; no new paid result supersedes it.

## 当前第二轮评测口径

检索复用原 `evaluation.retrieval_metrics.query_metrics`：来源命中Recall@5/10、相关来源去重Precision@3、来源MRR/nDCG，所有题参与总体均值；原无答案Recall=1及固定Precision分母3保持。逐题标注及30题回答审核不改。当前主备120题的Recall@5/10均1.00、Precision@3均0.2778；答案模型审核单独统计事实、条件、无依据主张、引用及无答案编造。

仅重算已保存排名及汇总已有LLM审核，零新增模型调用：

```powershell
& 'D:\anaconda3\envs\learn_claude\python.exe' -m experiments.summarize_optimization_v2 --output artifacts/qwen-optimization-v2/20261010-qwen-01
```

当前结果见 [Qwen正式报告](QWEN_RAG_OPTIMIZATION_REPORT.md)。所有旧回答、原审核与必要失败保留；已有历史运行中的旧派生口径不作为当前指标引用。

## 意图接 Agent 路由统计

```powershell
& 'D:\anaconda3\envs\learn_claude\python.exe' -m experiments.intent_routing
```

从Python后端目录运行。读取已完成Qwen运行的真实分类输出，在同152题上比较标准意图路由和实际三路/两路分类结果接生产路由；另复测240题路由回归。新增模型调用为0，决策后的回答和工具执行不计分。默认升级节点关闭时人工请求由General接待并标记升级；评分遵守当前功能配置。[计算口径、标注修订、结果及边界](INTENT_ROUTING_REPORT.md)，[逐题结果](../artifacts/qwen-optimization-v2/20261010-qwen-01/routing/current_test.json)。路由映射为业务策略派生标准，不是独立人工标注。
