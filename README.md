# mymind

mymind 是一个面向客服场景的多 Agent 系统。本仓库包含 Python 主版本、Java 重构版本和一个可切换双后端的 Vue 调试前端。

## 项目组成

| 目录 | 定位 | 默认端口 |
|---|---|---:|
| `mymind/` | Python / FastAPI 主版本，包含细粒度意图、RAG、三级记忆、评测与多模型缓存 | 8000 |
| `mymindJava/` | Java 21 / Spring Boot 重构版，使用专用知识工具与本地 Hybrid RAG | 8080 |
| `mymindFrontend/` | Vue / Vite 调试前端，可切换 Python 和 Java 后端 | 5173 |

本轮 EchoMind 能力融合落在 Python 和 Vue。Java 仍使用原有粗粒度意图与响应契约，前端 adapter 会兼容缺失的诊断字段。

## Python 主链路

```text
POST /chat
  -> Redis 工作记忆 + ChromaDB 情景记忆/用户画像
  -> LLM / 字符 n-gram / 关键词三路意图识别
  -> KnowledgePolicy 按意图和业务信号决定是否执行 RAG
  -> 查询改写、并行召回、去重、LLM 重排
  -> ContextBuilder 控制上下文预算
  -> RoutingDecision 选择主 Agent 和辅助 Agent
  -> General / Technical / Billing Agent 执行
  -> 写回记忆并异步更新用户画像
  -> 监控与评测记录结构化诊断信息
```

Python 支持 Anthropic、OpenAI、DeepSeek 原生接口和 DeepSeek Anthropic 兼容接口。具体缓存与实验说明见 [`mymind/experiments/README.md`](mymind/experiments/README.md)。

## 快速启动

### Python

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
D:\anaconda3\envs\learn_claude\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
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
  "knowledge_reason": "intent:payment_issue",
  "escalated": false,
  "latency_ms": 320.5
}
```

`knowledge_status` 取值：

- `used`: 成功取得并注入知识内容
- `skipped`: 策略判断无需检索
- `empty`: 已检索但没有可用结果
- `degraded`: 工具进入 fallback，fallback 不作为真实知识依据
- `error`: 检索链路异常

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
D:\anaconda3\envs\learn_claude\python.exe -m pytest -q
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
D:\anaconda3\envs\learn_claude\python.exe -m experiments.run_experiments --layer integration --redis-url redis://:mymind123@localhost:6379/15 --chroma-port 8001
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

评测会把当前指标与上一份 baseline 比较；相对下降超过 5% 会记录在 `regressions`。先检查具体失败用例，再决定调整模板、路由阈值或 RAG 门控，不要只追求单一总分。

## 部署安全

当前项目默认面向本地开发和演示。公开部署前至少需要：

- 为知识导入、Skills reload 和评测接口增加认证与访问控制
- 收紧 CORS
- 覆盖默认 Redis 密码并限制 Redis/Chroma 对外端口
- 使用外部 Secret 管理 API Key
- 确认 `.env`、Chroma SQLite、日志和评测数据未进入 Git
