# mymind（Python 主版本）

mymind 是面向客服场景的多 Agent 智能客服系统。本目录为 Python / FastAPI 主版本实现，承担细粒度意图识别、多 Agent 路由、RAG 检索、三级记忆、在线监控与实验评测。

## 项目目录架构

```text
mymind/
├── agents/                            # 多 Agent 路由与编排
│   ├── agent_orchestrator.py          # Profile、主/辅路由、工具循环、升级与 Composer
│   ├── tools.py                       # 角色白名单工具、Schema 校验和 Trace 脱敏
│   └── trace_store.py                 # 内存 / Redis 请求级工具 Trace
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
│   ├── agent_migration.py             # 240 条数据的 E0–E5 确定性消融
│   ├── agent_migration_real.py        # 有费用确认和硬预算的 DeepSeek 配对实验
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

## Agent Profile、工具调用与人工升级

Python 主版本在原有路由、Skills、记忆和前置 RAG 之上增加了请求级 Agent 运行时：

- General、Technical、Billing 使用不同的 `AgentProfile`、温度、Token 上限和工具白名单。
- Anthropic、OpenAI 与 DeepSeek 的工具调用统一转换为 `ToolCall`，每个 Agent 最多续轮 3 次。
- 未授权工具和不符合 JSON Schema 的参数只会产生脱敏 Trace，不会执行处理函数。
- `EscalationAgent` 确定性生成待人工处理摘要，不调用 LLM，也不声称创建真实工单。
- 两个或更多 Agent 成功返回时才调用 `ResponseComposer`；调用失败时确定性保留全部结论。
- 工具结果、工具 Trace 和去重缓存均保存在 `Request` 上，不写入共享 Agent 实例。

角色默认参数为 General `0.3/900`、Technical `0.1/1200`、Billing `0.0/1100`、Composer `0.1/1000`（温度/最大输出 Token）。角色模型可由 `MYMIND_GENERAL_MODEL`、`MYMIND_TECHNICAL_MODEL`、`MYMIND_BILLING_MODEL`、`MYMIND_COMPOSER_MODEL` 覆盖；未配置时继续使用统一网关模型。

## 功能开关与知识检索模式

```dotenv
AGENT_PROFILE_ENABLED=false
AGENT_TOOL_USE_ENABLED=false
AGENT_ESCALATION_ENABLED=false
AGENT_COMPOSER_ENABLED=false
AGENT_TRACE_ENABLED=false
TRACE_API_ENABLED=false
KNOWLEDGE_TOOL_MODE=disabled
AGENT_MAX_TOOL_ROUNDS=3
TRACE_TTL_SECONDS=86400
TRACE_MAX_ENTRIES=200
```

`KNOWLEDGE_TOOL_MODE=disabled` 是生产默认值，继续使用 `KnowledgePolicy + 前置 RAG`；`supplemental` 允许 Agent 补充检索并在请求内去重；`tool_only` 只用于实验。Trace 采集与 Trace 查询接口可独立开关，生产默认不开放查询接口。

新能力默认全部关闭。当前 DeepSeek E0/E5 正式实验未通过质量、Judge 完整性和 Token 成本门槛，因此不得整体开启 E5；后续应按 Profile → Tool Use → Escalation → Composer → 内部 Trace 的顺序逐项复验和灰度。

## API 增量契约

`POST /chat` 保留所有旧字段，并增加完整 UUID `request_id` 与去重后的 `tools_used`。完整工具参数和结果不会直接返回给聊天客户端。

- `GET /trace/tool/{request_id}`：返回单请求脱敏 Trace；过期或不存在时 `found=false`。
- `GET /trace/tools?limit=20`：按时间倒序返回最近 Trace，`limit` 会限制到 1–100。
- `TRACE_API_ENABLED=false` 时两个接口均返回 404。

生产环境使用 Redis 键 `mymind:trace:tool:{request_id}` 和 Sorted Set `mymind:trace:recent`；默认 TTL 为 24 小时、最近列表最多 200 条。密码、Token、验证码、密钥和支付凭证不会写入 Trace。

## 验证

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m pytest -q
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer agent-migration
```

迁移正式评测集位于 `data/eval/agent_migration_dataset.json`：共 240 条，六类各 40 条，其中 60 条多轮、60 条对抗或缺失信息场景。原 EchoMind 的 11 条意图和 5 组对话保存在 `data/eval/echomind_smoke.json`，明确标记为 `formal_gate=false`，只用于 Smoke Test。
