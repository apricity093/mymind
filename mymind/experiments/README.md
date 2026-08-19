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
