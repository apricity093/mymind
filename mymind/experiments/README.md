# Cache and Memory Experiments

Run commands from the `mymind/` directory with the project on `PYTHONPATH`.

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer offline
```

Docker integration uses an isolated Redis database and temporary Chroma collections:

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer integration --redis-url redis://:mymind123@localhost:6379/15 --chroma-port 8001
```

The real-model layer is opt-in because it incurs API cost:

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer real --confirm-cost
```

Every layer writes timestamped JSON and Markdown reports under `artifacts/experiments/`.
