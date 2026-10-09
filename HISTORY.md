# HISTORY

## fix(rag): 知识导入后同步删除旧缓存并阻止旧结果回填 — 2026-10-09

- 知识导入和上传接口等待旧缓存清理完成后返回成功；Redis 切换缓存版本后同步删除历史版本键，保留当前版本和其他命名空间数据。
- 工具检索记录执行前的缓存版本，内存锁与 Redis Lua 校验阻止更新前的在途结果回填；缓存清理失败时接口返回 503，提示知识已写入并可重试导入。
- 更新使用指南与改动记录。验证：`learn_claude` 相关回归 `58 passed`，全量 `107 passed`；共享 Redis 后端替身验证跨实例失效，本机真实 Redis 连接超时，真实服务验证未完成。

## fix(llm): 累计扩容重试用量并更新检索评测提示 — 2026-10-07

- Anthropic 协议扩容重试累计首轮和重试响应返回的输入、输出与缓存 Token；DeepSeek 兼容路径同步规范化用量 metadata，避免成本遥测漏计。
- 知识工具调用评测偏低时，建议核对用例预期、Agent 工具使用规则与工具配置。
- 同步说明文档、验证记录和两处下载文档副本；下载版 EchoMind README 修正随附 `.env` 启动说明与工具式 RAG 主流程。保留现有运行配置、示例模板和下载程序代码。
- 验证：`learn_claude` 相关回归 `18 passed`，全量 `86 passed`；52 个本地链接、14 组源码节选、下载副本一致性和 `git diff --check` 通过。未调用付费模型。

## docs(mymind): 完成说明文档、流程图与 README 同步 — 2026-10-07

- 新增 `mymind/wiki` 的业务流程、完整使用指南、技术亮点、重点代码和系统学习文档，按当前 Python 主版本说明工具式 RAG、请求轨迹、默认开关、监控回退和多 Provider 能力。
- 重绘五张原文件名 PNG 流程图，提供对应可编辑 SVG；修订仓库、Python 和实验 README，更新文档阅读入口。
- 将五份说明文档及流程图同步到下载包的 `EchoMind/wiki` 与 `详细文档+简历`，将 `待更新文档清单.md` 标记为全部完成。
- 验证：文档链接、十五个 Python/两个 JSON/二十九个 PowerShell 示例、十四组源码节选和 PNG/SVG 校验通过；两处下载目录文件内容与项目版本一致。只修改文档，无新增模型调用或业务测试运行。

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


