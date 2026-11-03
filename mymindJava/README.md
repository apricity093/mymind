# MyMind Java 网关与编排层

Java 端定位为面向高并发生产场景的企业网关与多角色编排层，基于 Spring Boot 与 Spring AI，负责应用接入、知识工具熔断、并行 Agent 编排和监控反馈。Nginx 承担入口限流，Actuator/Micrometer 暴露 Prometheus 指标。

Python 端承担 SFT/GRPO、Customer Policy 推理、意图融合与混合检索实验。当前 Java 通过 Spring AI 独立调用模型供应商，没有调用 Python 策略服务，也不加载其 LoRA 权重。生产级是设计定位，尚未完成生产部署和高并发容量验收。双端调用链见[系统架构设计](../CONTEXT.md)。

## 技术栈

| 类型 | 技术 |
|------|------|
| 语言 | Java 21 |
| Web 框架 | Spring Boot 3.5 |
| AI 框架 | Spring AI 1.1 |
| LLM Provider | Anthropic、DeepSeek |
| 文档处理 | LangChain4j DocumentSplitter |
| 记忆缓存 | Spring Data Redis |
| RAG | BM25 + 本地 hash vector + LLM rerank |
| 持久化 | Redis 工作记忆 + JSON 知识库/长期记忆/用户画像 |
| 监控 | Spring Boot Actuator、Micrometer、Prometheus |
| API 文档 | Springdoc OpenAPI、Swagger UI |
| 部署 | Docker、Docker Compose、Nginx、Prometheus |
| 构建 | Maven Wrapper |

## 核心链路

```text
POST /chat
  -> MemoryManager 读取 Redis 工作记忆、会话摘要、长期记忆、用户画像
  -> KnowledgeToolManager 做查询改写、并行召回、LLM rerank
  -> AgentOrchestrator 做意图识别和 Agent 路由
  -> General / Technical / Billing Agent 生成回复
  -> AnswerVerifier 校验回复是否可信、是否需要转人工
  -> 写入 Redis，并异步更新用户画像
```

相关实现：

- `src/main/java/com/mymind/api/mymindController.java`
- `src/main/java/com/mymind/memory/MemoryManager.java`
- `src/main/java/com/mymind/tool/KnowledgeToolManager.java`
- `src/main/java/com/mymind/agent/AgentOrchestrator.java`
- `src/main/java/com/mymind/agent/AnswerVerifier.java`

## 双端分工

| 能力 | Python 算法引擎 | Java 网关与编排 |
| --- | --- | --- |
| 策略训练与独立推理 | SFT、GRPO、证据评分、Customer Policy | 当前不承载训练，也未代理策略推理 |
| 应用对话 | 细粒度意图融合、主辅 Agent 工具循环 | 记忆与知识预取、角色编排、回答校验 |
| 检索 | Qwen/MiniLM 双索引、BM25、融合与重排 | 本地字符 hash vector、BM25、融合与重排 |
| 状态存储 | Redis + Chroma | Redis + 本地 JSON |
| 接入与监控 | FastAPI、Prometheus、诊断接口 | Spring MVC、Nginx 限流、工具熔断、Actuator/Micrometer |
| 前端契约 | conv_id、top_k | conversation_id、topK；由前端适配 |

`CircuitBreaker` 管理知识工具失败后的熔断与恢复；它不是请求限流器。Nginx 按客户端 IP 配置 10 请求/秒、burst 20 和连接限制。Agent 与检索并行调用使用 CompletableFuture 的默认执行资源，目前不宣称线程资源隔离或稳定吞吐能力。

## 服务能力

### DeepSeek 兼容

Java 版通过 Spring profile 切换模型：

```env
SPRING_PROFILES_ACTIVE=deepseek
DEEPSEEK_API_KEY=your_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

缺少真实 API Key 时，启动占位值仅用于完成 Spring AI 自动配置。真实调用缺少凭据、模型服务异常或响应缺少消息时直接抛出错误；网关不生成本地业务回复。

Anthropic 配置：

```env
SPRING_PROFILES_ACTIVE=anthropic
ANTHROPIC_API_KEY=your_key
ANTHROPIC_BASE_URL=https://api.anthropic.com
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### 记忆和知识库持久化

Python 版：

- Redis 保存工作记忆。
- ChromaDB 保存情景记忆、用户画像和知识库。

Java 版：

- Redis 保存工作记忆。
- `data/java/memory-store.json` 保存情景记忆和用户画像。
- `data/java/knowledge-store.json` 保存知识库片段。
- Docker 中保存到 `/app/data/java`，由 `app-data` volume 持久化。

配置项：

```env
mymind_DATA_DIR=data/java
KNOWLEDGE_STORE_PATH=data/java/knowledge-store.json
MEMORY_STORE_PATH=data/java/memory-store.json
EVAL_BASELINE_PATH=data/eval/baseline.json
```

当前用户可见效果已经对齐：应用重启后，导入的知识库、长期记忆和画像可以恢复。底层差异仍然存在：Java 没有直接写入 ChromaDB collection，而是 JSON 持久化 + 本地 hash vector 检索。

### Hybrid RAG

Python 侧使用 Chroma 向量索引与 BM25 混合检索，另有主备索引恢复机制。

Java 版当前检索链路：

```text
文档导入
  -> LangChain4j recursive splitter
  -> JSON 持久化
  -> 本地 documents 索引

查询
  -> LLM 查询改写
  -> 多子查询并行召回
  -> BM25 关键词得分
  -> 本地 hash vector 语义得分
  -> 加权融合
  -> LLM rerank
  -> fallback 到融合分排序
```

Java 的融合基于本地检索实现，没有把 ChromaDB 作为主召回源；其召回质量需独立评估。

### 评测和监控

Java 版已经实现：

- Intent Accuracy
- Macro-F1
- per-class Precision / Recall / F1
- LLM-as-Judge 四维评分
- baseline 保存和回归检测
- `/monitor`
- `/metrics`
- `/actuator/prometheus`
- Webhook 告警
- Agent 路由惩罚反馈

相关实现：

- `src/main/java/com/mymind/evaluation/EndToEndEvaluator.java`
- `src/main/java/com/mymind/evaluation/LLMJudge.java`
- `src/main/java/com/mymind/monitor/PerformanceMonitor.java`

## 实现边界

Java 使用专用 KnowledgeToolManager，不承担 Python 通用工具实验和训练模块。Chroma 自动配置在当前 profile 中被排除；主召回为本地 Hybrid RAG。监控采用成功率与延迟阈值、Webhook 和路由惩罚反馈。CLI、算法实验和策略训练保留在 Python 端。

## 主要接口

默认端口：`8080`

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/chat` | 主对话接口 |
| POST | `/search` | 知识库检索 |
| POST | `/knowledge/add` | 批量添加知识文档 |
| POST | `/knowledge/upload` | 上传 `.txt` / `.md` / `.json` 文件 |
| GET | `/knowledge/stats` | 知识库统计 |
| GET | `/monitor` | 监控摘要 |
| GET | `/metrics` | Prometheus 指标，兼容 Python 版路径 |
| GET | `/actuator/prometheus` | Spring Actuator Prometheus 指标 |
| POST | `/eval/run` | 运行评测 |
| GET | `/docs` | Swagger UI，可在线调用接口 |
| GET | `/v3/api-docs` | OpenAPI JSON |

Java 版配置了 Jackson `SNAKE_CASE`，响应字段会输出为：

```json
{
  "conversation_id": "...",
  "response": "...",
  "intent": "...",
  "agent_type": "...",
  "escalated": false,
  "latency_ms": 123,
  "knowledge_used": true,
  "verified": true,
  "grounded": true
}
```

和 Python 版主要差异：

- Python 使用 `conv_id`。
- Java 使用 `conversation_id`。
- Java 额外返回 `verified` 和 `grounded`。

推荐请求字段：

```json
{
  "message": "我想申请退款",
  "user_id": "u1001",
  "conversation_id": "optional-conversation-id"
}
```

## 本地运行

### macOS / Linux

准备环境：

- JDK 21 或更高版本
- Docker Desktop 或 Docker Engine

启动依赖：

```bash
docker compose up -d redis chromadb
```

DeepSeek 启动：

```bash
export SPRING_PROFILES_ACTIVE=deepseek
export DEEPSEEK_API_KEY=your_key
./mvnw spring-boot:run
```

Anthropic 启动：

```bash
export SPRING_PROFILES_ACTIVE=anthropic
export ANTHROPIC_API_KEY=your_key
./mvnw spring-boot:run
```

健康检查：

```bash
curl http://localhost:8080/health
```

Swagger：

```text
http://localhost:8080/docs
```

### Windows PowerShell

启动依赖：

```powershell
docker compose up -d redis chromadb
```

DeepSeek 启动：

```powershell
$env:SPRING_PROFILES_ACTIVE="deepseek"
$env:DEEPSEEK_API_KEY="your_key"
.\mvnw.cmd spring-boot:run
```

Anthropic 启动：

```powershell
$env:SPRING_PROFILES_ACTIVE="anthropic"
$env:ANTHROPIC_API_KEY="your_key"
.\mvnw.cmd spring-boot:run
```

健康检查：

```powershell
curl http://localhost:8080/health
```

Swagger：

```text
http://localhost:8080/docs
```

## Docker 部署

复制配置：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，选择模型 profile 并填写对应 API Key。

启动：

```bash
docker compose up -d --build
```

查看状态：

```bash
docker compose ps
```

查看日志：

```bash
docker compose logs -f mymind-java
```

停止：

```bash
docker compose down
```

Compose 服务和端口：

| 服务 | 容器名 | 地址 |
|------|--------|------|
| Java App | `mymind-java-app` | `http://localhost:8080` |
| Nginx | `mymind-java-nginx` | `http://localhost:8081` |
| Prometheus | `mymind-java-prometheus` | `http://localhost:9091` |
| ChromaDB | `mymind-java-chromadb` | `http://localhost:8002` |
| Redis | `mymind-java-redis` | `localhost:6380` |

常用验证：

```bash
curl http://localhost:8080/health
curl http://localhost:8081/health
curl http://localhost:8080/metrics
```

Swagger：

```text
http://localhost:8080/docs
http://localhost:8081/docs
```

## API 示例

### 对话

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "我想申请退款，订单号是 #12345",
    "user_id": "u1001"
  }'
```

也可以打开 Swagger UI，通过页面直接调用：

```text
http://localhost:8080/docs
```

### 检索

```bash
curl -X POST "http://localhost:8080/search?query=退款多久能到账&topK=3"
```

### 添加知识库

```bash
curl -X POST http://localhost:8080/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {
        "title": "退款补充政策",
        "content": "大促期间退款审核时间可能延长到 3-5 个工作日。"
      }
    ]
  }'
```

### 上传知识库文件

```bash
curl -X POST http://localhost:8080/knowledge/upload \
  -F "file=@docs.md"
```

### 运行评测

```bash
curl -X POST http://localhost:8080/eval/run \
  -H "Content-Type: application/json" \
  -d '{}'
```