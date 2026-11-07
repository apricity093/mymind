# 数据集与离线评估

在 `mymind/` 目录使用 `python -m benchmarks.scripts.<入口>`。构建器不调用模型或外部服务。

| 入口 | 用途与输出 |
| --- | --- |
| `build_rag_dataset` | RAG 语料和标注查询，默认写入 `data/eval/` |
| `build_agent_migration_dataset` | 主辅 Agent 路由合成数据，默认写入 `data/eval/` |
| `build_optimization_v2` | 合成检索与意图评测材料，默认写入 `data/eval/` |
| `build_mymind_results` | 根据本地正式记录生成根目录 `output/mymind-results.html` |
| `export_demo` | 将配置模板导出到 `output/demo/`；支持 `--output-dir`，默认工作根目录取 `MYMIND_WORK_ROOT` 或仓库根目录 |
| `summarize_evaluation_latency` | 汇总保留评测记录中的耗时与 token |
| `analyze_optimization_failures`、`audit_optimization_data` | 逐题失败与监督数据分析，需要本地历史记录 |
| `verify_agentic_retained` | 正式任务、示范及监督 mask 的零模型复核 |

结果页与历史分析需要本地 `agentic_runs/` 的正式记录；公开仓库未附这些原件。模板保存在 `templates/`，生成的 HTML、JSON 和其他导出物放在 Git 忽略的 `output/`。演示配置的模型与索引需另行准备。

## 本地正式记录的使用

以下命令从仓库根目录执行。输入为单独保留的本地正式记录，结果页及演示配置不随 Git 提交：

```powershell
python mymind/benchmarks/scripts/verify_agentic_retained.py --root . --host local
python mymind/benchmarks/scripts/build_mymind_results.py
python mymind/benchmarks/scripts/export_demo.py
```

生成后在本地打开 `output/mymind-results.html`。演示模板导出后仍须配置实际模型、索引、凭据及新的输出目录。

数据来源审计可写入独立目录：`python mymind/benchmarks/scripts/audit_optimization_data.py --root . --output output/data-audit`。`analyze_optimization_failures.py --root <工作根目录>` 当前固定写入该根下的 `agentic_runs/optimization-20261022/`，应使用包含输入原件的工作副本，避免覆盖已有报告证据。两个分析脚本均不调用模型。

历史报告内的旧 `scripts/` 命令和测试文件记录当时运行；当前测试统一从根目录执行 `python -m pytest -q`。零模型核验检查输入、来源与监督 mask，不代表历史 GPU 更新或当前模型服务已重新运行。

## 当前候选与先导数据

当前生成逻辑集中于balanced_policy_*、success_pool_contract及teacher_contract_candidate。prepare_mainline_candidates、prepare_source_expansion生成候选；verify_mainline_candidates、verify_source_expansion检查公开/私有一致性。数据先筛选教师成功，再整族划分。

prepare_three_branch_pilot、configure_three_branch_pilot分别准备固定先导划分及运行配置；学生监督通过rl.sft_data编码。已有实验目录不能覆盖后重跑。参考回答只是检查夹具，不能当教师示范。数据规模、筛选偏差和当前失败实例见[合成数据说明](../docs/synthetic_data.md)。
