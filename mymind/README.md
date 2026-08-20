# mymind（Python 主版本）

mymind 是面向客服场景的多 Agent 智能客服系统。本目录为 Python / FastAPI 主版本实现，承担细粒度意图识别、多 Agent 路由、RAG 检索、三级记忆、在线监控与实验评测。

## 项目目录架构

```text
mymind/
├── agents/                            # 多 Agent 路由与编排
│   └── agent_orchestrator.py          # 主/辅 Agent 结构化路由决策与执行
├── api/                               # Web 服务入口
│   └── main.py                        # FastAPI 应用（/chat、/search、/knowledge/* 等）
├── artifacts/                         # 实验产物（评测报告、Chroma 实验库、模型等，Git 忽略）
│   └── experiments/
├── config/                            # 部署与可观测性配置
│   ├── alerts/                        # Prometheus 告警规则
│   ├── grafana/                       # Grafana dashboards 与 provisioning
│   ├── nginx/                         # nginx.conf 与 ssl/
│   └── prometheus.yml                 # Prometheus 抓取配置
├── core/                              # 核心业务组件
│   ├── cache_metrics.py               # 应用 / Provider 级缓存指标
│   ├── cache_store.py                 # 运行时与实验共享的缓存接口
│   ├── intent_recognizer.py           # LLM / 字符 n-gram / 关键词三路融合意图识别
│   ├── knowledge_policy.py            # 按意图与业务信号决定是否执行 RAG
│   ├── llm_gateway.py                 # 多 Provider 统一 LLM 网关
│   ├── llm_utils.py                   # LLM 响应处理工具
│   ├── prompt_cache.py                # Anthropic prompt-cache 边界构造
│   ├── retrieval.py                   # RAG 检索核心（切块、去重、BM25 + RRF、重排）
│   └── skill_loader.py                # Skill 热加载器
├── data/                              # 数据目录
│   ├── demo_docs/                     # 示例知识文档
│   ├── eval/                          # 评测语料、标注数据与构建脚本
│   └── chroma/                        # 运行时 ChromaDB 数据（Git 忽略）
├── evaluation/                        # 评测框架
│   ├── evaluator.py                   # 端到端 Agent 评测
│   └── retrieval_metrics.py           # RAG 检索指标与统计
├── experiments/                       # 可复现实验（说明见 experiments/README.md）
│   ├── offline.py                     # 离线 / 字符代理实验
│   ├── integration.py                 # 集成实验
│   ├── real_model.py                  # 真实模型实验
│   ├── rag_retrieval.py               # R0–R4 检索消融实验（离线确定性层）
│   ├── rag_chroma.py                  # 真实 Chroma + all-MiniLM 检索实验
│   ├── reporting.py                   # 实验报告生成
│   └── run_experiments.py             # 实验运行入口
├── logs/                              # 运行日志（Git 忽略）
├── mcp/                               # MCP 工具与知识库
│   ├── knowledge_base.py              # 基于 ChromaDB 的 RAG 知识库
│   └── tool_manager.py                # MCP 工具调用管理
├── memory/                            # 多轮对话记忆
│   ├── conversation_memory.py         # 三级记忆（工作记忆 / 情景记忆 / 用户画像）
│   └── context_builder.py             # 上下文预算构建
├── monitor/                           # 在线表现监控
│   └── performance_monitor.py         # 性能与业务指标监控
├── skills/                            # 可热加载技能定义（每个目录一个 SKILL.md）
│   ├── billing_support/               # 账单支持技能
│   ├── general_customer_service/      # 通用客服技能
│   └── technical_support/             # 技术支持技能
├── tests/                             # pytest 测试（含 e2e_contract_server.py 契约测试）
├── tools/                             # 预留工具目录（当前为空）
├── Dockerfile                         # 镜像构建
├── docker-compose.yml                 # 本地依赖编排（Redis、ChromaDB 等）
├── build-image.sh / run-image.sh      # 镜像构建 / 运行脚本
├── docker-deploy.sh                   # 部署脚本
├── requirements.txt / requirements-dev.txt
├── .env.example / .env.example.env    # 环境变量样例（真实 .env 不入库）
└── .gitignore
```

> 说明：`.venv/`、`.idea/`、`.pytest_cache/`、`__pycache__/` 等运行环境、IDE 与缓存目录未在上图中列出。

## 目录职责速览

| 目录 | 职责 |
| --- | --- |
| `agents/` | 多 Agent 路由决策与编排 |
| `api/` | FastAPI Web 服务入口 |
| `artifacts/` | 实验与评测产物（Git 忽略） |
| `config/` | Prometheus、Grafana、Nginx 等部署配置 |
| `core/` | 意图识别、LLM 网关、RAG、缓存、Skill 加载等核心能力 |
| `data/` | 示例文档、评测数据与运行时 ChromaDB 数据 |
| `evaluation/` | 端到端 Agent 评测与检索指标 |
| `experiments/` | 可复现的缓存、记忆与 RAG 实验 |
| `logs/` | 运行日志（Git 忽略） |
| `mcp/` | 知识库与 MCP 工具调用 |
| `memory/` | 多轮对话三级记忆与上下文构建 |
| `monitor/` | Agent 在线表现与性能监控 |
| `skills/` | 按业务域划分的可热加载技能 |
| `tests/` | pytest 单元、集成与契约测试 |
| `tools/` | 工具扩展预留目录 |
