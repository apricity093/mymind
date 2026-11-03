# MyMind：客服 Agent 算法引擎与网关编排

MyMind 研究客服任务中的证据获取与处置决策，并用 Python、Java 和 Vue 展示训练、推理与应用工程的边界。

- **SFT/GRPO 单策略闭环**：Qwen3-4B Base → LoRA SFT → 在线工具交互 RL，包含 assistant-only 监督、证据判分、验证选模和独立 Customer Policy 推理。
- **意图路由融合与混合 RAG**：LLM、embedding、关键词三路融合，主辅 Agent 路由，向量 + BM25 召回、查询改写、重排与缓存失效。
- **Python 算法底座 + Java 网关编排**：Python 承担 AI Native 算法与 Agentic RL；Spring Boot + Spring AI 承担面向生产的接入治理、多角色编排与监控；Vue 统一适配双端应用接口。

Java 的高并发生产级定位目前体现为工程实现：知识工具熔断、Nginx 限流、并行编排和 Prometheus 指标。项目用于合成案例和个人环境演示，尚无生产容量验收；Java 也尚未接入 Python 的 SFT/RL 策略服务。

## 研究结果

正式配置：158 条教师示范 → SFT120 → RL80 步中选择 checkpoint-40；使用固定验证集选模，最终评估原 120 题。

| 原 120 题 | Base | SFT120 | RL40 |
| --- | ---: | ---: | ---: |
| 完整成功 | 0 | 43 | 43 |
| 严重错误 | 38 | 38 | 27 |
| 协议错误 | 76 | 24 | 19 |

RL 新增 3 道成功、失去 3 道，完整成功率未提升；严重错误与协议错误减少。结论限于单 seed、固定合成政策和题集，不能外推稳定提升或泛化。两类错误可重叠，提前返回未检查项保持未知。详见[训练评测说明](mymind/rl/README.md)。

## 系统组成

| 目录 | 职责与入口 |
| --- | --- |
| `mymind/rl/` | 教师示范、SFT、GRPO、评分与选模 |
| `mymind/agents/customer_policy/` | 训练与独立推理共用的策略协议、检索环境和执行器 |
| `mymind/api/policy_app.py` | 独立策略服务：/requests、/status |
| `mymind/api/main.py` | Python 应用验证：/chat、知识库、路由与工具诊断，默认 8000 |
| `mymind/core/`、`mymind/mcp/` | 意图融合、混合 RAG、模型网关与缓存 |
| `mymind/experiments/`、`mymind/benchmarks/` | 应用实验、数据构建与离线分析 |
| `mymindJava/` | Spring Boot 网关与编排，默认 8080；模型调用通过 Spring AI |
| `mymindFrontend/` | Vue 控制台，默认 5173；切换两端 /chat 与知识接口 |

`/chat` 的多 Agent 编排与 Customer Policy 独立服务分别运行。后训练实验给定账单或技术角色，未训练路由器，也未将所选策略接入任一后端的 `/chat`。Java 当前独立执行编排和检索，两端契约一致不代表内部算法或输出等价。

## 阅读入口

| 目的 | 文档 |
| --- | --- |
| 双端职责、调用链、前端契约与领域术语 | [系统架构设计](CONTEXT.md) |
| 策略协议、训练及独立服务启动 | [策略架构与运行](mymind/docs/agentic_rl_architecture.md) |
| Python 配置、模块与接口 | [Python 说明](mymind/README.md) |
| Java 接入治理、运行与监控 | [Java 说明](mymindJava/README.md) |
| 前端代理、适配与部署 | [控制台说明](mymindFrontend/README.md) |
| 检索与路由实验 | [实验入口](mymind/experiments/README.md) |
| 数据构建、分析和结果页生成 | [Benchmarks](mymind/benchmarks/README.md) |

公开仓库提供实现、小型合成开发样例、配置模板、测试及结果摘要。正式数据、私有评分标准、原始轨迹、模型权重和索引单独保留，不具备直接重跑全部正式实验的完整资源。结果页可由 `python mymind/benchmarks/scripts/build_mymind_results.py` 生成至 `output/`，需要本地正式记录。

## 启动

Python 本地环境使用 `environment-local.yml`；模型推理与训练依赖见 `mymind/requirements-agentic-rl.txt`。在仓库根目录创建并激活开发环境：

```powershell
conda env create -f environment-local.yml
conda activate agentic_rl
cd mymind
Copy-Item .env.example .env
```

根据模板填写模型供应商、API Key、Redis 和 embedding 配置后启动应用：

```powershell
docker compose up -d redis chromadb
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Java 使用 JDK 21，按 [Java 配置](mymindJava/README.md) 准备模型和 Redis；在 `mymindJava/` 执行 `./mvnw.cmd spring-boot:run`。限流需要经过其 Nginx 代理，直接访问 8080 不经过该限制。

前端在 `mymindFrontend/` 执行：

```powershell
npm install
npm run dev
```

控制台通过 `/api/python`、`/api/java` 代理到本地后端。切换后端开启新会话；Python 使用 `conv_id`，Java 使用 `conversation_id`，差异由适配层处理。两端应用文档均提供 `/docs`。

## 验证

在仓库根目录运行 Python 离线套件：

```powershell
python -m pytest -q
```

五个测试域覆盖意图路由、RAG 管线、Agent 编排、记忆与缓存、RL 核心；测试使用进程内客户端与轻量替身，不读取历史实验结果 JSON。

前端在 `mymindFrontend/` 运行 `npm test` 和 `npm run build`；Java 在 `mymindJava/` 使用 JDK 21 运行 `./mvnw.cmd compile`。本地检查不代表真实模型效果、远程 GPU 或高并发压测通过。

## 运行边界

模型调用和 `/eval/run` 评测可能产生 API 费用。知识导入、评测及管理接口默认用于本地演示，公开部署需另行完成认证、网络访问控制和容量验收。真实凭据、数据库、模型与运行产物不入库。
