# mymind

mymind 是一个面向客服场景的多 Agent 系统，包含工具式 RAG、意图与路由评测，以及 Qwen3-4B 的 LoRA SFT／在线 GRPO 后训练实验。本仓库包含 Python 主版本、Java 重构版本和一个可切换双后端的 Vue 调试前端。

`/chat` 负责意图识别、主辅 Agent 编排、记忆与工具检索；独立的 `api.policy_app:app` 运行经过 SFT／RL 的共享客服策略。后训练实验给定账单或技术角色，未训练主辅路由，也未将所选策略接入 `/chat`。两条链路的评测分别报告。

## 项目与结果入口

| 阅读目的 | 入口 |
| --- | --- |
| 正式选定配置、三阶段结果与应用主辅路由 | [交互结果展示](mymind-results.html)，可直接用浏览器打开 |
| 两条业务链路、训练协议与独立推理 | [策略架构与运行](mymind/docs/agentic_rl_architecture.md) |
| SFT、GRPO、评分器与结果范围 | [训练评测说明](mymind/experiments/agentic_rl/README.md) |
| 客服任务、证据与处置术语 | [领域词汇表](CONTEXT.md) |

## 项目组成

| 目录 | 定位 | 默认端口 |
|---|---|---:|
| `mymind/` | Python / FastAPI 主版本，包含细粒度意图、RAG、三级记忆、评测与多模型缓存 | 8000 |
| `mymindJava/` | Java 21 / Spring Boot 重构版，使用专用知识工具与本地 Hybrid RAG | 8080 |
| `mymindFrontend/` | Vue / Vite 调试前端，可切换 Python 和 Java 后端 | 5173 |

## 文件结构

Python 后端位于仓库根目录下的 `mymind/`。以下列出主要文件；实验权重、索引和真实服务配置按运行环境保留，部分本地产物由 Git 忽略。

### 项目根目录

```text
mymind/                                   # 项目根目录
├── mymind/                               # Python 后端，结构见下文
├── exports/                              # 展示与独立推理导出
│   └── showcase/                         # 结果页的 HTML 片段
├── scripts/                              # 数据构建、训练诊断、结果汇总与本地证据复核
│   ├── verify_agentic_retained.py        # 保留证据的零模型复核
│   └── verify_stage7_retained.py         # 阶段7保留证据复核
├── README.md                             # 项目介绍、结构、启动与接口
├── CONTEXT.md                            # 领域术语与系统边界
├── mymind-results.html                   # 交互结果展示
├── environment-local.yml                 # 本地 agentic_rl Conda 环境
├── pytest.ini                            # 根目录 pytest 配置
├── LICENSE
└── .gitignore
```

### Python 后端 `mymind/`

```text
mymind/
├── agents/                               # 多 Agent 应用与共享客服策略
│   ├── agent_orchestrator.py             # 意图路由、主辅编排、工具循环与 Composer
│   ├── tools.py                          # 角色工具、参数校验与 Trace 脱敏
│   ├── trace_store.py                    # 请求级工具 Trace 存储
│   └── customer_policy/                  # SFT/RL 共用的客服策略运行时
│       ├── protocol.py                   # 动作、观察与终止输出协议
│       ├── environment.py                # 工具交互环境与预算
│       ├── generation.py                 # 模型生成与 LoRA 推理
│       ├── retrieval.py                  # 策略原子检索及可见证据
│       └── runner.py                     # 单任务策略执行
├── api/
│   ├── main.py                           # 多 Agent FastAPI 服务（/chat 等）
│   ├── policy_app.py                     # 所选 SFT/RL 策略的独立推理服务
│   └── configs/agent-dev.json            # 策略服务开发配置
├── core/                                 # 意图、LLM 网关、Embedding、切块、检索与缓存
├── mcp/                                  # 知识库与 MCP 工具管理
│   ├── indexed_knowledge_base.py         # 当前主/备双索引、原文保存与补齐
│   ├── knowledge_base.py                 # 原知识库与检索实验基线
│   └── tool_manager.py                   # 工具执行、查询改写、重排与缓存
├── memory/                               # 三级记忆与上下文预算
├── monitor/                              # 在线表现与性能监控
├── evaluation/                           # 端到端评测与检索指标
├── experiments/                          # 实验实现、配置与报告入口
│   ├── agentic_rl/                       # 教师示范、SFT、GRPO、评分与正式评测
│   ├── intent_routing.py                 # 主辅Agent路由回归与历史分类重放
│   ├── qwen_application_acceptance.py    # 当前应用检索与服务验收
│   └── README.md                         # 各实验入口及适用范围
├── data/                                 # 后端本地数据
│   ├── demo_docs/                        # 示例知识文档
│   ├── eval/                             # 应用意图、路由与RAG评测语料
│   └── chroma/                           # 运行时 Chroma 数据
├── docs/                                 # 策略训练架构与 ADR
├── wiki/                                 # 使用指南、业务流程与技术讲解
├── skills/                               # 通用、账单和技术客服技能定义
├── tests/                                # 单元、集成、协议与实验契约测试
├── config/                               # Prometheus、Grafana、Nginx 部署配置
├── artifacts/                            # 本地产物与实验缓存
├── logs/                                 # 运行日志
├── tools/                                # 工具扩展预留目录
├── README.md                             # 后端详细结构与运行说明
├── requirements-local.txt                # 本地开发与 CPU 检查依赖
├── requirements-agentic-rl.txt           # 服务器模型推理与训练依赖
├── requirements.txt                      # 后端基础依赖
├── requirements-dev.txt                  # 开发依赖
├── Dockerfile
├── docker-compose.yml                    # Redis、Chroma 等依赖服务
├── build-image.sh                        # 镜像构建
├── run-image.sh                          # 镜像运行
├── docker-deploy.sh                      # 部署脚本
├── .env.example                          # 环境变量示例
└── .env.example.env                      # 补充环境变量示例
```

`mymind/data/` 保存随附的后端示例与应用评测数据；策略实验的数据资源位于工作根目录的 `data/agentic_rl/`，单独保留。详细模块职责见 [Python 后端说明](mymind/README.md)。环境、IDE、缓存目录与真实 `.env` 未展开。

仓库提供实现、少量合成开发样例、配置模板、测试与结果摘要。正式及校准数据集、各版本评分标准和来源记录，以及原始轨迹、158 条教师示范及 token 导出、冻结作业配置、模型权重和知识索引均单独保存。结果页内嵌已核对的指标，公开仓库本身不包含重跑全部正式训练或独立核验每条历史轨迹所需的资源。

## Python 主链路

```text
POST /chat
  -> Redis 工作记忆 + ChromaDB 情景记忆/用户画像
  -> LLM / Qwen embedding / 关键词三路意图识别
  -> ContextBuilder 控制上下文预算
  -> RoutingDecision 选择主 Agent 和辅助 Agent
  -> General / Technical / Billing Agent 工具循环
       -> Agent 按需调用 search_knowledge_base
       -> 查询改写、并行召回、去重、重排后返回工具结果
  -> 写回记忆并异步更新用户画像
  -> 监控与评测记录结构化诊断信息
```

Python 支持 Anthropic、OpenAI、DeepSeek 原生接口和 DeepSeek Anthropic 兼容接口。具体缓存与实验说明见 [`mymind/experiments/README.md`](mymind/experiments/README.md)。

工具调用和工具式 RAG 默认开启；Profile、独立人工升级节点、Composer、Trace 采集与 Trace 查询默认关闭。专业 Agent 的全部实例降权达到默认阈值 `0.5` 时，该类型退出当次路由；阈值由 `MYMIND_MONITOR_FALLBACK_PENALTY` 配置。

详细说明见 [完整使用指南](mymind/wiki/完整使用指南.md)、[业务流程说明](mymind/wiki/业务流程说明.md)、[技术亮点](mymind/wiki/技术亮点.md)、[重点代码](mymind/wiki/重点代码.md) 和 [学习文档](mymind/wiki/EchoMind学习文档.md)。

## 快速启动

### Python

本地开发、运行和测试统一使用 Conda 环境 `agentic_rl`（Python 3.10）。在仓库根目录创建环境：

```powershell
conda env create -f environment-local.yml
conda activate agentic_rl
```

环境已存在时，使用 `python` 执行下文命令。补装或更新本地依赖：

```powershell
& python -m pip install -r mymind/requirements-local.txt
```

本地清单覆盖 Python 后端、Chroma/ONNX 检索、离线实验和 pytest；模型推理及训练在服务器 `mindagent` 中运行，使用 `mymind/requirements-agentic-rl.txt`。Redis/Chroma 服务仍按下面的 Docker 命令启动。

在 `mymind/` 目录准备 `.env`，至少配置：

```env
LLM_PROVIDER=deepseek
LLM_API_KEY=replace_with_your_key
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
REDIS_URL=redis://:mymind123@localhost:6379/0
CHROMA_HOST=localhost
CHROMA_PORT=8001
```

不要提交真实 `.env`、API Key 或运行态数据库。

启动依赖和 Python 服务：

```powershell
cd mymind
docker compose up -d redis chromadb
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

访问：

- Swagger: `http://localhost:8000/docs`
- 健康检查: `http://localhost:8000/health`
- Prometheus metrics: `http://localhost:8000/metrics`

### Vue 前端

```powershell
cd mymindFrontend
npm install
npm run dev
```

访问 `http://localhost:5173`，在侧栏选择 Python 或 Java 后端。

Java 的详细启动方式见 [`mymindJava/README.md`](mymindJava/README.md)，前端部署说明见 [`mymindFrontend/README.md`](mymindFrontend/README.md)。

## API 示例

### 对话

```http
POST /chat
Content-Type: application/json

{
  "message": "订单 #A123 登录时报 401，而且被重复扣款",
  "user_id": "u1001",
  "conv_id": "optional-conversation-id"
}
```

Python 响应保留原字段，并增量返回诊断信息：

```json
{
  "conv_id": "...",
  "request_id": "...",
  "response": "...",
  "intent": "payment_issue",
  "intent_group": "billing",
  "agent_type": "billing",
  "agent_types": ["billing", "technical"],
  "primary_agent": "billing",
  "supporting_agents": ["technical"],
  "routing_reason": "intent=payment_issue, ...",
  "routing_confidence": 1.0,
  "entities": {"order_id": ["A123"], "error_code": ["401"]},
  "intent_confidence": 0.9,
  "intent_source_scores": {"llm": 0.9, "embedding": 0.5, "pattern": 0.75},
  "knowledge_used": true,
  "knowledge_status": "used",
  "knowledge_reason": "agent_tool:used",
  "tools_used": ["search_knowledge_base"],
  "escalated": false,
  "latency_ms": 320.5
}
```

`knowledge_used=true` 表示实际进入检索工具处理；即使检索为空或失败，该字段仍可为 true。`knowledge_status` 从工具轨迹汇总，取值如下：

- `used`: 检索工具返回知识片段
- `skipped`: Agent 未发起检索工具调用
- `empty`: 已检索但没有可用结果
- `degraded`: 工具进入 fallback，fallback 不作为真实知识依据
- `error`: 检索链路异常

同一请求多次检索按 `used → degraded → error → empty` 汇总。有检索轨迹时 `knowledge_reason=agent_tool:<状态>`，否则为 `agent_did_not_search`。完整脱敏轨迹通过 `/trace/tool/{request_id}` 或 `/trace/tools` 查询，需要同时开启 `AGENT_TRACE_ENABLED` 和 `TRACE_API_ENABLED` 才能采集并读取。

### 导入知识

```http
POST /knowledge/add
Content-Type: application/json

{
  "documents": [
    {"title": "退款政策", "content": "退款审核通常需要 1-3 个工作日。"}
  ]
}
```

导入使用 Chroma `upsert`，重复提交相同文档不会产生重复片段。响应同时保留 `added_chunks` 并提供语义更准确的 `processed_chunks`。

### 运行评测

```http
POST /eval/run
Content-Type: application/json

{
  "intent_cases": [
    {"message": "退款多久到账", "expected_intent": "refund"}
  ],
  "dialog_cases": [
    {
      "question": "应用登录一直报 401",
      "expected_intent": "technical_login",
      "expected_primary_agent": "technical",
      "expect_knowledge_search": true
    }
  ]
}
```

报告包含 Accuracy、Macro-F1、LLM Judge 四维质量分、`routing_accuracy`、`knowledge_gate_accuracy`、baseline 回归项和优化建议。多轮用例的期望值应用于最后一轮。

`knowledge_gate_accuracy` 比较实际 `search_knowledge_base` 工具调用与用例预期；不再用意图策略代替实际执行结果。评测接口会调用所配置的模型与 Judge，可能产生 API 费用。

## 细粒度意图

新增业务意图包括：

- `order_status`、`logistics`
- `refund`、`invoice`、`payment_issue`
- `account_security`
- `technical_login`、`technical_crash`
- `human_handoff`

它们分别映射到 `query`、`billing`、`account`、`technical`、`escalation` 等泛化分组。旧客户端仍可依赖 `intent`、`agent_type` 等原字段。

## 测试

Python：

```powershell
cd mymind
python -m pytest -q
```

前端：

```powershell
cd mymindFrontend
npm run test
npm run build
```

真实 Redis/Chroma 集成实验必须使用非生产 Redis DB，例如 `/15`：

```powershell
cd mymind
python -m experiments.run_experiments --layer integration --redis-url redis://:mymind123@localhost:6379/15 --chroma-port 8001
```

## 常见问题

### 启动时报缺少 API Key

设置 `LLM_API_KEY`，或使用兼容旧配置的 `ANTHROPIC_API_KEY`。模型、Provider 和 Base URL 必须属于同一供应商配置。

### 宿主机无法连接 Redis 或 ChromaDB

宿主机运行 Python 时使用 `localhost:6379` 和 `localhost:8001`；容器内运行 Python 时使用 Compose 服务名 `redis:6379` 和 `chromadb:8000`。

### `knowledge_status=degraded`

表示知识工具超时、熔断或异常并进入降级。系统不会把降级提示当作知识内容。检查 Chroma 健康状态和 `/monitor` 工具统计。

### 重复导入文档

当前导入是幂等 upsert。若同一标题的文档结构发生变化，建议使用稳定且唯一的标题，并在导入后清理不再使用的旧内容。

### 评测报告出现回归

评测会把当前指标与上一份 baseline 比较；相对下降超过 5% 会记录在 `regressions`。先检查具体失败用例，再决定调整意图模板、路由阈值或 Agent 的检索工具使用规则。

## 部署安全

当前项目默认面向本地开发和演示。公开部署前至少需要：

- 为知识导入、Skills reload 和评测接口增加认证与访问控制
- 收紧 CORS
- 覆盖默认 Redis 密码并限制 Redis/Chroma 对外端口
- 使用外部 Secret 管理 API Key
- 确认 `.env`、Chroma SQLite、日志和评测数据未进入 Git
