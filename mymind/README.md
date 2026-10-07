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
│   ├── knowledge_policy.py            # R0–R4 检索实验的确定性知识门控
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
├── wiki/                              # 使用、业务流程、技术讲解和 PNG / SVG 流程图
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
| `wiki/` | 当前 Python 实现的完整文档与流程图 |
| `tools/` | 工具扩展预留目录 |

## Agent Profile、工具调用与人工升级

Python 主版本用请求级 Agent 运行时连接路由、Skills、记忆与工具式 RAG：

- General、Technical、Billing 定义不同的 `AgentProfile` 和工具白名单；角色温度和 Token 上限在启用 Profile 时生效。
- Anthropic、OpenAI 与 DeepSeek 的工具调用统一转换为 `ToolCall`，每个 Agent 最多续轮 3 次。
- 未授权工具和不符合 JSON Schema 的参数只会产生脱敏 Trace，不会执行处理函数。
- 启用独立升级节点时，`EscalationAgent` 确定性生成待人工处理摘要，不调用 LLM；默认由 GeneralAgent 接待升级请求并标记 `escalated`。
- Composer 默认关闭，主辅结果按既有方式拼接；开启后由 `ResponseComposer` 合成，失败时保留成功结论。
- `AgentCallResult` 在单次模型调用中保存工具名称、轨迹和用量，`Request` 保存请求内工具结果缓存。模型续轮失败和 GeneralAgent 回退后继续保留已发生的工具轨迹。
- 监控降权达到阈值的专业实例退出选择；同类实例全部降权时移除该专业路由并记录 `monitor_fallback=[...]`，有健康实例时继续使用健康实例。

角色默认参数为 General `0.3/900`、Technical `0.1/1200`、Billing `0.0/1100`、Composer `0.1/1000`（温度/最大输出 Token）。角色模型可由 `MYMIND_GENERAL_MODEL`、`MYMIND_TECHNICAL_MODEL`、`MYMIND_BILLING_MODEL`、`MYMIND_COMPOSER_MODEL` 覆盖；未配置时继续使用统一网关模型。

## 功能开关与知识检索模式

```dotenv
AGENT_PROFILE_ENABLED=false
AGENT_TOOL_USE_ENABLED=true
AGENT_ESCALATION_ENABLED=false
AGENT_COMPOSER_ENABLED=false
AGENT_TRACE_ENABLED=false
TRACE_API_ENABLED=false
KNOWLEDGE_TOOL_MODE=tool_only
AGENT_MAX_TOOL_ROUNDS=3
MYMIND_MONITOR_FALLBACK_PENALTY=0.5
TRACE_TTL_SECONDS=86400
TRACE_MAX_ENTRIES=200
```

`KNOWLEDGE_TOOL_MODE=tool_only` 是默认值，模型在工具循环中自主调用 `search_knowledge_base`；`disabled` 关闭知识工具。`/chat` 始终不按意图预检索。`MYMIND_MONITOR_FALLBACK_PENALTY` 优先于 `ECHOMIND_MONITOR_FALLBACK_PENALTY`，未配置时采用 `0.5`。

工具调用与工具式 RAG 默认启用；Profile、独立升级、Composer、内部 Trace 与 Trace 查询保持默认关闭。历史 DeepSeek E0/E5 实验未通过质量、Judge 完整性和 Token 成本门槛；当前默认配置不等于完整 E5，也不代表真实模型质量门禁已经通过。

## API 增量契约

`POST /chat` 保留所有旧字段，并增加完整 UUID `request_id` 与去重后的 `tools_used`。完整工具参数和结果不会直接返回给聊天客户端。

`knowledge_used` 表示实际进入检索工具处理。`knowledge_status` 区分 `used/empty/degraded/error/skipped`，多次检索按 `used → degraded → error → empty` 汇总；`knowledge_reason` 为 `agent_tool:<状态>` 或 `agent_did_not_search`。`knowledge_gate_accuracy` 评测实际工具调用。

检索调用评测偏低时，建议核对用例预期、Agent 工具使用规则与工具配置。Anthropic 协议扩容重试成功后，用量包含首轮与重试响应返回的 Token；工具续轮用量在 Agent 层进一步聚合。

- `GET /trace/tool/{request_id}`：返回单请求脱敏 Trace；过期或不存在时 `found=false`。
- `GET /trace/tools?limit=20`：按时间倒序返回最近 Trace，`limit` 会限制到 1–100。
- `TRACE_API_ENABLED=false` 时两个接口均返回 404。
- 同时开启 `AGENT_TRACE_ENABLED=true` 才会保存 Trace；仅开放查询接口不会自动开始采集。

生产环境使用 Redis 键 `mymind:trace:tool:{request_id}` 和 Sorted Set `mymind:trace:recent`；默认 TTL 为 24 小时、最近列表最多 200 条。密码、Token、验证码、密钥和支付凭证不会写入 Trace。

## 验证

```powershell
D:\anaconda3\envs\learn_claude\python.exe -m pytest -q
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer agent-migration
```

迁移正式评测集位于 `data/eval/agent_migration_dataset.json`：共 240 条，六类各 40 条，其中 60 条多轮、60 条对抗或缺失信息场景。原 EchoMind 的 11 条意图和 5 组对话保存在 `data/eval/echomind_smoke.json`，明确标记为 `formal_gate=false`，只用于 Smoke Test。

2026-10-07 的程序同步回归为 `86 passed`，包含 E0–E5 离线门禁；API 契约和 Compose 配置检查通过。该结果基于假模型与内存组件，不替代真实服务或付费模型实验。

## 文档入口

- [完整使用指南](wiki/完整使用指南.md)：环境、HTTP/CLI、接口、知识库、记忆、监控与评测。
- [业务流程说明](wiki/业务流程说明.md)：请求执行顺序与五张流程图。
- [技术亮点](wiki/技术亮点.md)：设计动机、实现方式与能力边界。
- [重点代码](wiki/重点代码.md)：当前源码中的关键数据结构和方法。
- [学习文档](wiki/EchoMind学习文档.md)：从架构到模块的系统阅读路线。
