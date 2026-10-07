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

The 2026-10-07 regression run passed 83 tests, including the deterministic E0–E5 gate. API contract and Compose checks also passed. The earlier paid DeepSeek E0/E5 experiment failed quality, judge-completeness and token-cost gates; no new paid result supersedes it.
