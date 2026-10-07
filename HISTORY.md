# HISTORY

## fix(mymind): 同步 EchoMind 09.28/10.02 意图、轨迹和工具式 RAG — 2026-10-07

- 三路意图投票先合并领域分，再选组内具体意图；Pattern 仅细化同一领域。
- 使用请求局部 `AgentCallResult` 保留工具调用，模型续轮失败和专业 Agent 回退后继续保存已发生的轨迹；监控降权支持单实例回退和健康实例筛选。
- 按本次配置选择默认启用工具调用与工具式 RAG，删除 `/chat` 意图预检索；聊天接口和评测记录实际检索结果，其他 Agent 开关保持原值。
- 同步环境模板、Compose、API 契约测试服务和回归用例；提供 `待更新文档清单.md`，列出五份下载版文档、五张流程图和本地三份 README 的手动更新项。
- 验证：`learn_claude` 全量测试 `83 passed`，包含 E0–E5 离线门禁；健康检查、知识导入和工具式 RAG 对话契约通过；Compose 配置解析和 `git diff --check` 通过。未重新执行付费真实模型实验。

## feat(agent): 迁移 Profile、工具调用、升级、Composer 与 Trace — 2026-10-02

- 在 Python 主版本中新增结构化 Agent Profile、角色级工具白名单、统一多 Provider 工具协议、独立人工升级节点、多 Agent Composer 和请求级 Trace。
- 扩展 `/chat` 增量契约和 Trace 查询接口，增加 Redis Trace、工具参数校验、脱敏、请求级 RAG 去重及 Token 遥测。
- 新增 240 条正式迁移评测集、EchoMind Smoke 数据、E0–E5 离线/DeepSeek 实验和完整测试；生产默认保持新能力关闭，待真实门禁通过后分阶段启用。
- 验证：`learn_claude` 全量测试 `64 passed`；E0–E5 离线门禁通过；真实 DeepSeek 实验严格限制为 600 次调用但质量、Judge 完整性和 Token 成本门禁未通过。

## docs(mymind): 新增 Python 主版本 README 目录架构文档 — 2026-08-20

- 新增 `mymind/README.md`，以目录树和职责速览表描述 `agents/`、`api/`、`core/`、`mcp/`、`memory/`、`experiments/` 等目录及关键模块职责。
- 依据 `AGENTS.md` 协作规范同步更新 `report.md` 与 `HISTORY.md`。
## feat(rag): 完成 check.md R0-R4 检索实验、前端契约适配与评测体系 — 2026-08-19

- 新增 R0-R4 及消融检索变体（段落切块、稳定 ID、去重、BM25+RRF、确定性重排回退），以及 128 条标注查询与指标/统计/聚类 bootstrap 评测。
- 后端新增字段均为可选增量字段，旧 `/search`、`/knowledge/*`、`/chat` 契约保持；前端统一稳定 chunk id 并分别发送 `top_k`/`topK`。
- 验证：Python `45 passed`；前端 Node `6 passed`；生产构建成功；浏览器契约 E2E 通过；真实 Chroma all-MiniLM 层 R4 通过验收（rewrite/rerank 仍为确定性代理）；独立审查报告与后续修订项见 `review_rag_experiment.md`。


