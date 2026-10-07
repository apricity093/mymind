# 项目改动记录

## 2026-10-07（维护完成与提交记录）

- 改动文件：`HISTORY.md`、`report.md`。
- 改动摘要：记录扩容重试用量、检索评测提示及过时文档修正，准备按协作规范提交和推送。运行配置、示例模板及既有未跟踪 `analysis.md` 保持现状。
- 验证结果：相关回归 18 项通过，全量 86 项通过；52 个本地链接和 14 组源码节选一致，两处下载文档与项目一致；EchoMind README 的链接和启动/RAG 文案检查通过；`git diff --check` 通过。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（维护文档与下载版 README 覆盖）

- 改动文件：下载目录 `EchoMind/wiki` 与 `详细文档+简历` 中五份 Markdown、`EchoMind/README.md`、`report.md`。
- 改动摘要：两处详细文档同步当前维护结果；下载版 README 改为编辑随附 `.env`，修正主流程为 Agent 自主检索，并区分 EchoMind 原程序与 mymind 详细文档的适用范围。未改下载目录程序与配置文件。
- 验证结果：待内容一致性、链接与过时文案检查。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（维护回归与完成清单）

- 改动文件：`mymind/README.md`、`mymind/experiments/README.md`、`mymind/wiki/` 五份 Markdown、`待更新文档清单.md`、`report.md`。
- 改动摘要：当前文档的测试记录更新为 86 项；完成清单补充 Token 与评测提示维护和 EchoMind README 修订，运行配置及示例模板保持现状。
- 验证结果：`learn_claude` 全量回归 `86 passed in 16.72s`，包含已有工具、缓存、API 与离线实验回归。未执行付费模型实验。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（评测与重试修正的文档同步）

- 改动文件：`mymind/README.md`、`mymind/wiki/业务流程说明.md`、`完整使用指南.md`、`技术亮点.md`、`重点代码.md`、`EchoMind学习文档.md`、`report.md`。
- 改动摘要：说明扩容重试与工具续轮分别累计用量、DeepSeek 兼容 metadata 的统一口径，以及与 Agent 自主检索一致的评测建议。
- 验证结果：相关假模型回归 `18 passed`；待全量回归后更新当前验证记录并同步下载目录。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（评测提示与重试用量修正）

- 改动文件：`mymind/core/llm_gateway.py`、`mymind/evaluation/evaluator.py`、`mymind/tests/test_multi_provider_cache.py`、`report.md`。
- 改动摘要：知识调用评测建议改为核对用例预期、Agent 工具使用规则和工具配置；Anthropic 协议扩容重试聚合首轮与重试响应的输入、输出和缓存用量，DeepSeek 兼容适配同步规范化 metadata。按用户要求保留 `.env` 及示例模板。
- 验证结果：新增空响应重试的 Anthropic 与两种 DeepSeek 缓存用量格式回归，并加强截断重试及指标断言，待 `learn_claude` 验证。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档更新提交记录）

- 改动文件：`HISTORY.md`、`report.md`。
- 改动摘要：按项目协作规范记录五份说明文档、五张 PNG/SVG、三份 README 和清单完成状态的提交；两处下载目录已同步。
- 验证结果：文档与图像检查通过，待提交前检查 Markdown 差异格式及已完成清单链接。程序与运行配置未变更，保留既有未跟踪 `analysis.md`。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档清单完成与下载目录同步）

- 改动文件：`待更新文档清单.md`、`report.md`；下载目录 `EchoMind/wiki` 与 `详细文档+简历` 中对应五份说明文档、五张 PNG 和五份 SVG。
- 改动摘要：将已核对的 mymind 文档覆盖到两处下载资料目录；清单更新为已完成并提供本地阅读入口。下载包的其他文档与程序代码未改动。
- 验证结果：两处各 15 个文件与项目文档内容一致，复制后的相对链接有效；本轮只改文档与图像，不重复运行全量业务测试或付费模型实验。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档核对与图像布局验证）

- 改动文件：`mymind/wiki/完整使用指南.md`、`mymind/wiki/EchoMind学习文档.md`、`report.md`。
- 改动摘要：明确 1024 Token 为未启用 Profile 时的首轮参数，并说明 Anthropic 协议网关单次扩容重试；完成五张流程图的文字与分支布局检查。
- 验证结果：8 份 Markdown 的 39 个本地链接、15 个 Python 和 2 个 JSON 示例、29 个 PowerShell 语法块、14 组原文源码节选、5 个 PNG 和 5 个 SVG 校验通过。文档变更不修改程序，未重复运行模型或全量业务测试。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档清单落实：流程图）

- 改动文件：`mymind/wiki/assets/flows/01-chat-flow.png`、`02-intent-fusion.png`、`03-agent-routing.png`、`04-rag-tool-flow.png`、`05-memory-compression.png` 及对应 SVG、`report.md`。
- 改动摘要：按当前 Python 主链路重绘五张原文件名流程图，加入 ContextBuilder、默认开关、同领域细化、健康实例筛选、真实知识状态和消息数压缩条件；保留可编辑 SVG。
- 布局修订：英文标识按单词换行；调整主辅和记忆分支连线，补齐澄清绕过 Agent、非法工具回送以及模型轮数判断分支。
- 验证结果：图像生成包含文字边界检查，待渲染后逐张检查布局与箭头。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档清单落实：当前源码讲解）

- 改动文件：`mymind/wiki/重点代码.md`、`report.md`。
- 改动摘要：以当前源码的数据结构和十四个关键方法节选说明 HTTP 执行、领域投票、请求工具状态、监控路由、失败回退、RAG、upsert、记忆压缩、背景预算与实际工具调用评测；补充配置、Skills 和 Trace 阅读位置。
- 验证结果：代码节选按 AST 定位当前方法，待统一校验语法与源码一致性。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档清单落实：系统学习文档）

- 改动文件：`mymind/wiki/EchoMind学习文档.md`、`report.md`。
- 改动摘要：保留学习资料文件名，按本地实现说明架构阅读路线、数据结构、意图、路由、角色工具与多 Provider、知识链路、并发记忆、缓存、Skills、监控、Trace 和评测；提供与现有回归对应的阅读练习。
- 验证结果：已对照本地模块说明默认配置与接口边界，待与其余文档统一检查。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档清单落实：完整使用指南）

- 改动文件：`mymind/wiki/完整使用指南.md`、`report.md`。
- 改动摘要：提供本地 Python、Compose、CLI 和 Vue 入口说明，完整 API 操作、UTF-8 PowerShell 示例、知识与记忆查看、Skills 热加载、监控回退、Trace 和评测排查；保留真实服务与付费验证边界。
- 验证结果：已对照 FastAPI 路由、环境配置、记忆键与数据模型，示例语法和链接待统一检查。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档清单落实：业务流程与技术亮点）

- 改动文件：`mymind/wiki/业务流程说明.md`、`mymind/wiki/技术亮点.md`、`report.md`。
- 改动摘要：新增与本地源码一致的流程和技术说明，涵盖 ContextBuilder、领域投票、主辅路由、监控过滤、工具式 RAG、知识状态、并发工具轨迹、三级记忆、Skills 和分层验证。
- 验证结果：已按当前业务模块核对阈值和执行顺序，图片与相对链接待统一检查。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（文档清单落实：README）

- 改动文件：`README.md`、`mymind/README.md`、`mymind/experiments/README.md`、`report.md`。
- 改动摘要：同步工具式 RAG 主链路与默认开关、请求局部工具轨迹和失败回退、监控降权、知识状态和实际调用评测；保留离线实验与真实模型门禁的区分；增加完整文档入口。
- 验证结果：待五份文档和流程图完成后统一检查链接、代码片段与配置说明。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（同步更新提交记录）

- 改动文件：`HISTORY.md`、`待更新文档清单.md`、`report.md`。
- 改动摘要：记录本轮程序同步与验证结果；补充 Compose 配置解析通过和 CLI 尚未初始化知识库管理器的文档边界，修正文档清单的本地文件链接。
- 验证结果：`docker compose -f mymind/docker-compose.yml config --quiet` 成功；此前全量回归 83 项通过，契约测试服务通过。按协作规范提交并推送本轮改动，不包含既有未跟踪 `analysis.md`。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（程序同步完成与文档待更新清单）

- 改动文件：`待更新文档清单.md`、`report.md`。
- 改动摘要：整理下载版五份 Markdown、五张流程图和本地三份 README 的手动更新位置，写明本地工具式 RAG 默认值、知识状态、监控回退、开关和实验差异；原 README 和下载目录文档未覆盖。
- 验证结果：使用 `learn_claude` 在允许 Windows 本地套接字的执行环境运行全量测试，`83 passed in 9.87s`；其中 E0–E5 离线实验门禁随测试通过。前端契约测试服务健康检查、知识导入和 Agent 工具式 RAG 对话通过。`git diff --check` 通过；未执行付费真实模型实验。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（同步回归测试与前端契约测试服务）

- 改动文件：`mymind/tests/test_echomind_updates.py`、`mymind/tests/test_intent_routing_fusion.py`、`mymind/tests/test_agent_tool_migration.py`、`mymind/tests/test_rag_retrieval.py`、`mymind/tests/test_knowledge_policy_and_eval.py`、`mymind/tests/e2e_contract_server.py`、`report.md`。
- 改动摘要：补充领域合分和 Pattern 细化、单实例/多实例/辅助路由监控回退、可配置阈值、模型失败后轨迹保留、同一 Agent 并发工具隔离、Agent 自主检索及 API 状态回归测试；评测测试区分实际调用与未调用，契约测试服务改为使用共享知识工具。
- 知识库重复导入测试使用只支持 `upsert` 的集合，确认相同稳定 ID 和正文被再次提交写入；本地知识库已符合 09.28 写入语义，无需替换实现。
- 测试环境：受限执行中 Windows `asyncio` 创建事件循环的本地 `socketpair` 在 `accept` 阻塞，诊断确认尚未进入异步业务代码；测试将继续使用 `learn_claude` 在允许本地套接字的执行环境运行。
- 验证结果：待 `learn_claude` 执行本轮针对性与全量回归。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（同步 EchoMind 09.28：统一工具式 RAG）

- 改动文件：`mymind/agents/agent_orchestrator.py`、`mymind/agents/tools.py`、`mymind/api/main.py`、`mymind/evaluation/evaluator.py`、`mymind/docker-compose.yml`、`mymind/.env.example`、`mymind/.env.example.env`、`report.md`。
- 改动摘要：按用户选择默认启用工具调用和 `tool_only` RAG，保留其他 Agent 开关；删除 `/chat` 意图预检索和已预加载去重路径，保留请求内工具结果缓存。检索工具返回实际知识状态，聊天接口从工具轨迹区分使用、空结果、降级和错误；评测改为核对实际检索工具调用。同步 Compose、环境模板和监控回退阈值。
- 验证结果：待 API 契约、工具状态和评测回归测试验证；本地 `.env` 没有覆盖上述两个开关。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（同步 EchoMind 09.28：失败路径工具轨迹）

- 改动文件：`mymind/agents/agent_orchestrator.py`、`report.md`。
- 改动摘要：新增请求局部 `AgentCallResult`，模型续轮失败时保留已执行工具、轨迹和用量；专业 Agent 失败回退到 GeneralAgent 时合并此前工具记录，避免最终响应和持久化 Trace 丢失调用历史。保留原有成功路径及请求并发隔离设计。
- 验证结果：待工具成功、续轮失败、并发和回退回归测试验证。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-07（同步 EchoMind 09.28 / 10.02：意图融合与监控路由）

- 改动文件：`mymind/core/intent_recognizer.py`、`mymind/agents/agent_orchestrator.py`、`report.md`。
- 改动摘要：三路投票先累计领域分，再选领域内具体意图；Pattern 细化限定为同一领域。监控降权达到阈值的专业实例不再参与当次执行，全部实例降权时移除该专业路由并记录 `monitor_fallback`；支持 `MYMIND_MONITOR_FALLBACK_PENALTY` 和下载版的 `ECHOMIND_MONITOR_FALLBACK_PENALTY` 配置。
- 验证结果：待新增针对性回归测试后在 `learn_claude` 环境运行。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-10-02（Agent 工具迁移：提交与推送准备）

- 改动文件：`HISTORY.md`、`report.md`。
- 改动摘要：按项目协作规范记录本次 Agent Profile、工具调用、Escalation、Composer、Trace、API、评测集和实验迁移的提交说明；明确真实 DeepSeek 门禁未通过，因此新能力生产默认关闭。
- 验证结果：提交前最近一次 `learn_claude` 全量测试为 `64 passed`，E0-E5 离线消融通过，真实 DeepSeek 600 次调用实验结果不通过并已执行默认关闭回滚。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：DeepSeek 正式结果与发布回滚）

- 改动文件：`mymind/agents/agent_orchestrator.py`、`mymind/docker-compose.yml`、`mymind/.env.example`、`mymind/.env.example.env`、`mymind/README.md`、`mymind/tests/test_agent_tool_migration.py`、`mymind/tests/test_intent_routing_fusion.py`、`report.md`。
- 改动摘要：完成 60 条、E0/E5 各 3 次的 360 顶层请求真实 DeepSeek 实验，并严格将 Agent、工具续轮、Composer 和配对 Judge 共用调用限制在 600 次。生成阶段 E0 207 次、E5 361 次，剩余预算不足以完成全部 Judge；按发布回滚规则将 Profile、Tool Use、Escalation、Composer 和内部 Trace 的代码/容器默认值全部设为关闭，保留显式配置和 E0-E5 实验入口，Trace API 与补充 RAG 继续关闭。
- 验证结果：正式报告 `agent-migration-real-20260831-131624.json`，`overall_passed=false`。配对质量 E5-E0=-0.0844，95% CI [-0.1325,-0.0427]；E0/E5 平均 Token 1179.2/3214.4（2.73 倍，失败）；p95 21.49s/33.36s（1.55 倍，通过）；禁止声明率 1.67%/0.56%（通过）；总调用 600/600，但 Judge 失败 40/60。失败门禁：质量 +5pp、质量 CI>0、Judge 零失败、Token 成本≤1.6倍。回滚默认值并补完整 UUID 断言后，`learn_claude` 最终全量回归 `64 passed`、Python 编译检查和 `git diff --check` 均通过；最终 E0-E5 离线消融再次 `overall_passed=true`。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：下载版 Smoke 数据隔离）

- 改动文件：`mymind/data/eval/echomind_smoke.json`、`mymind/tests/test_agent_tool_migration.py`、`mymind/README.md`、`report.md`。
- 改动摘要：从下载版代码中核对并单独保存 11 条默认意图样本与 5 组默认对话，元数据明确标记 `smoke_only` 和 `formal_gate=false`；新增数量与用途约束测试，避免其被误作 240 条正式迁移门禁集或替代现有 128 条 RAG 回归集；同步补全 README 中新增 Agent 工具、Trace 和迁移实验模块的目录树。
- 验证结果：待本轮真实实验结束后与全量测试一并复核。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：Token 与成本门禁遥测）

- 改动文件：`mymind/core/llm_gateway.py`、`mymind/agents/agent_orchestrator.py`、`mymind/core/cache_metrics.py`、`mymind/experiments/agent_migration_real.py`、`mymind/tests/test_multi_provider_cache.py`、`report.md`。
- 改动摘要：预算版真实实验复核时发现原统一缓存遥测只统计输入 Token，且 E0/E5 会误复用最后一个编排器快照，已在 Anthropic/OpenAI/DeepSeek 统一用量契约中增加输出 Token，工具续轮聚合输入/输出/缓存用量，缓存指标同步记录输出 Token；真实实验现在独立保存每个变体的调用数、Token 和缓存快照，并以平均总 Token 作为成本代理执行 E5 不超过 E0 1.6 倍门禁。
- 验证结果：第二次不完整真实运行已主动中止且不作为结果；修正后全量测试 `63 passed`；1 次 DeepSeek 配对 Judge 烟雾测试成功返回 E0=0.575、E5=0.825 且解析无失败。正式运行增加每 20 个请求的非敏感进度输出，随后从头执行唯一正式实验。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：真实实验硬预算）

- 改动文件：`mymind/core/llm_gateway.py`、`mymind/experiments/agent_migration_real.py`、`mymind/experiments/README.md`、`mymind/tests/test_agent_tool_migration.py`、`report.md`。
- 改动摘要：首次真实 DeepSeek 运行发现兼容端点频繁因 `max_tokens` 自动二次调用，按趋势会超过 600 次预算，已主动中止且不作为实验结果；新增请求级不完整响应重试开关，真实实验统一首轮 2048 Token 且关闭隐藏重试，所有 Agent、工具续轮、Composer 与 Judge 共享 600 次硬上限；Judge 改为每个样本一次同时独立评分 E0/E5，仍保留 60 组配对分数和 Bootstrap 置信区间。
- 验证结果：预算版代码待无费用单测和全量回归后重新执行真实实验。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：运行配置与降级语义）

- 改动文件：`pytest.ini`、`mymind/agents/agent_orchestrator.py`、`mymind/api/main.py`、`mymind/docker-compose.yml`、`mymind/experiments/agent_migration_real.py`、`mymind/tests/test_agent_tool_migration.py`、`report.md`。
- 改动摘要：限定 pytest 正式收集目录，避免误扫描历史 ONNX 产物；统一 Provider 不支持工具时的无工具降级与标准错误码，区分非法参数和执行失败；`tool_only` 模式跳过 API 前置 RAG；统一 Trace 前缀配置并补齐容器最大工具轮数；真实迁移实验可从项目 `.env` 读取 DeepSeek 凭据但仍要求显式费用确认。
- 验证结果：限定 `mymind/tests` 的全量回归为 `60 passed`；另补 Provider 工具协议降级和非法参数标准错误码测试；首次用新增 pytest 配置复跑发现旧契约 Fake 无 `features` 属性，API 已改为兼容读取并待再次复跑。最终 E0-E5 离线实验已成功生成报告。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：边界加固、并发验证与文档）

- 改动文件：`mymind/agents/tools.py`、`mymind/agents/agent_orchestrator.py`、`mymind/tests/test_agent_tool_migration.py`、`mymind/README.md`、`mymind/experiments/README.md`、`mymind/.env.example`、`mymind/.env.example.env`、`report.md`。
- 改动摘要：修正 JSON 数值 Schema 接受布尔值的边界，扩展 Trace 字符串内嵌凭证脱敏，阻止补充 RAG 重复执行已完成的同查询前置检索，避免无 Profile 路径向 SDK 传递空温度，并增加 Composer 模型覆盖；补充三轮工具上限、RAG 去重、Composer 故障、Redis 跨实例可见性和 200 并发请求隔离测试；文档新增功能开关、Provider/角色模型、Trace API、数据集和实验命令。
- 验证结果：边界测试 `15 passed`；此前全量回归 `55 passed`，文档更新后将再次统一执行。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：实验与 API 契约测试）

- 改动文件：`mymind/tests/test_agent_tool_migration.py`、`mymind/tests/test_rag_retrieval.py`、`report.md`。
- 改动摘要：新增 240 条数据完整性、E0-E5 离线门禁和真实模型费用确认测试；现有 `/chat` 契约测试增加 `request_id`、`tools_used`，确认仅做向后兼容字段扩展。
- 验证结果：待全量 pytest 统一验证。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：真实模型配对实验）

- 改动文件：`mymind/experiments/agent_migration_real.py`、`mymind/experiments/run_experiments.py`、`report.md`。
- 改动摘要：新增显式 `--confirm-cost` 的 DeepSeek E0/E5 配对实验，固定六类各 10 条、每变体重复 3 次（360 个顶层请求），首轮使用四维 LLM Judge，输出质量配对 Bootstrap 置信区间、延迟、工具率、禁止声明率、Judge 失败和缓存遥测门禁。
- 验证结果：代码待通过无费用的参数/导入测试；真实调用仅在凭据存在时执行。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：E0-E5 离线消融）

- 改动文件：`mymind/experiments/agent_migration.py`、`mymind/experiments/run_experiments.py`、`report.md`。
- 改动摘要：新增基于 240 条 gold 数据的 E0-E5 可复现消融实验，使用确定性 Policy Gateway 验证路由、主辅协作、升级、工具 Precision/Recall/F1、越权执行、禁止声明和延迟；实验接入统一 JSON/Markdown 报告与退出码门禁。
- 验证结果：首轮 E5 路由准确率 100%、升级准确率 100%、越权执行 0，但辅助 Agent 准确率 83%、工具 F1 81%；据逐条失败明细补充跨域协作判定后复跑通过全部门禁：E5 路由 100%、辅助 Agent 98%、升级 100%、工具 F1 96%、越权执行 0。实验报告生成在工作区外的可视化产物目录，未写入仓库 artifacts。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：240 条评测集生成器）

- 改动文件：`mymind/data/eval/build_agent_migration_dataset.py`、`mymind/data/eval/agent_migration_dataset.json`、`report.md`。
- 改动摘要：新增六类各 40 条的 Agent/Tool/Trace 迁移评测集生成器，覆盖通用、技术、账单、复合、人工升级和越权安全场景；每条包含路由、工具、升级、知识门控、禁止声明等 gold 字段，并机械校验多轮与对抗样本比例。
- 验证结果：已生成 240 条 JSON；其中多轮 60 条、对抗样本 60 条，满足预设比例。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：运行时测试）

- 改动文件：`mymind/tests/test_intent_routing_fusion.py`、`mymind/tests/test_agent_tool_migration.py`、`report.md`。
- 改动摘要：将人工转接旧契约更新为真实 `EscalationAgent`；新增 E0-E5 配置、角色白名单、Schema/脱敏、Anthropic/OpenAI 工具解析、工具闭环、越权阻断、确定性升级、Trace TTL/容量与 Trace API 开关测试。
- 验证结果：现有测试在更新前为 44 通过、1 个预期契约失败；新增测试待统一运行。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：API 与 Redis 接线）

- 改动文件：`mymind/api/main.py`、`mymind/docker-compose.yml`、`report.md`。
- 改动摘要：启动时将编排器接入 Redis Trace Store 和已注册的 RAG 工具管理器；`/chat` 向后兼容地新增完整 `request_id` 与 `tools_used`；新增默认关闭的脱敏 Trace 查询接口，并补充 Agent 功能开关、知识工具模式、Trace TTL/容量的容器配置。
- 验证结果：待新增 API、Redis 多 Worker 与完整回归测试后确认。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：编排运行时）

- 改动文件：`mymind/agents/agent_orchestrator.py`、`report.md`。
- 改动摘要：在现有路由评分基础上完成结构化 `AgentProfile`、E0-E5 功能配置、请求局部 Tool Use 循环、角色白名单、完整 UUID、独立人工升级节点、统一网关 Response Composer、请求 Trace 聚合和 RAG 工具模式接线；保留现有 Skills、Provider 网关、Prompt Cache 和缓存指标能力。
- 验证结果：待 API 与 Trace Redis 接线后运行现有回归及新增工具、并发、升级和 Composer 测试。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：请求 Trace 存储）

- 改动文件：`mymind/agents/trace_store.py`、`report.md`。
- 改动摘要：新增请求级 Trace Store 协议、线程安全内存实现和 Redis 跨 Worker 实现；单请求使用独立键，最近请求使用 Sorted Set 索引，支持 24 小时 TTL、最多 200 条裁剪和单次/最近记录查询。
- 验证结果：待编排器和 API 接线后运行 TTL、并发可见性和脱敏测试。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：角色白名单工具）

- 改动文件：`mymind/agents/tools.py`、`report.md`。
- 改动摘要：新增角色级确定性工具定义、统一 JSON Schema 校验、Trace 参数脱敏、通用/技术/账单/升级工具白名单，以及复用现有 `MCPToolManager` 的共享知识检索工具；请求级结果缓存用于避免同一请求重复检索。
- 验证结果：待编排器接线后运行工具协议、越权、并发与 RAG 去重测试。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-31（Agent 工具迁移：统一网关协议）

- 改动文件：`mymind/core/llm_gateway.py`、`report.md`。
- 改动摘要：在现有多 Provider 网关上新增统一 `ToolCall`/`LLMResult.tool_calls` 契约；Anthropic、OpenAI 和 DeepSeek 路径现在能够发送工具定义、解析工具调用并保留可续轮的 assistant 消息；同时避免把 Anthropic 正常 `tool_use` 错判为空响应重试，原缓存遥测字段保持兼容。
- 验证结果：待完成 Agent 运行时接线后统一运行契约测试与全量回归。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-20（RAG 对比分析）

- 改动文件：`analysis.md`、`report.md`。
- 改动摘要：拉取并审计开源项目 CampusMind-Agent（commit `9674404`）的 Python 源码，核实其完整 RAG 链路（512/64 固定滑窗切块、MySQL/SQLite 事实源 + Chroma 派生索引、OpenAI-compatible embedding、CHAT/CONSULT/RISK 门控、单 query 改写、BM25+向量加权融合、本地确定性 rerank、邻块扩展、分区上下文预算、检索证据 trace、60 条检索评测与 RAGAS 12 条生成评测），并与本地 mymind Python 主版本（commit `fbe3c71`）逐项对比；在 `analysis.md` 中给出可借鉴点（P0：检索证据持久化、RAGAS 生成质量评测、生产 Hybrid+真实 BM25 降级、事实源/可重建索引；P1：分区上下文预算、邻块窗口、知识库管理接口；P2：可插拔 embedding 与按意图检索配置）、不建议照搬点，以及找实习视角的简历/面试建议。临时下载的源码与压缩包已在分析完成后清理，不进入版本控制。
- 验证结果：结论均基于两仓库源码逐文件核对，关键机制附文件行号；`analysis.md` 编码为 UTF-8；`git status` 确认仅新增 `analysis.md`。
- 后续修订：按审查意见将查询边界约束改为"期望行为而非代码保证"，修正实验分层、评测胜负、生产 BM25 兜底与生产 rerank 校验等表述，补充 trace 脱敏/保留期/访问边界，并修正 Embedding 依赖与 CampusMind query 结构化程度描述。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-20

- 改动文件：`mymind/README.md`、`report.md`、`HISTORY.md`。
- 改动摘要：新增 `mymind/README.md`，以目录树和职责速览表记录 Python 主版本的目录架构及各模块职责；按协作规范同步更新 `report.md` 与 `HISTORY.md`。
- 验证结果：已确认 README 目录树与 `mymind/` 实际目录结构一致，Markdown 编码为 UTF-8。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-08

- 改动文件：`mymind/agents/agent_orchestrator.py`、`mymind/api/main.py`、`mymind/core/intent_recognizer.py`、`mymind/core/llm_utils.py`、`mymind/core/cache_metrics.py`、`mymind/core/llm_gateway.py`、`mymind/evaluation/evaluator.py`、`mymind/experiments/`、`mymind/mcp/tool_manager.py`、`mymind/memory/conversation_memory.py`、`mymind/requirements.txt`、`mymind/tests/test_multi_provider_cache.py`、`mymind/.gitignore`、`AGENTS.md`。
- 改动摘要：增加多模型 LLM 网关与缓存指标支持，扩展缓存和记忆相关调用链、实验脚本及测试；实验运行产物改为由 Git 忽略，并从版本控制中移除已有产物。
- 验证结果：使用 `learn_claude` 虚拟环境运行测试，结果为 `17 passed`。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-10

- 改动文件：`README.md`、`report.md`、`mymind/core/intent_recognizer.py`、`mymind/core/knowledge_policy.py`、`mymind/core/llm_gateway.py`、`mymind/agents/agent_orchestrator.py`、`mymind/api/main.py`、`mymind/mcp/tool_manager.py`、`mymind/mcp/knowledge_base.py`、`mymind/memory/conversation_memory.py`、`mymind/monitor/performance_monitor.py`、`mymind/evaluation/evaluator.py`、`mymind/experiments/integration.py`、`mymind/tests/fakes.py`、`mymind/tests/test_cache_memory_optimization.py`、`mymind/tests/test_intent_routing_fusion.py`、`mymind/tests/test_knowledge_policy_and_eval.py`、`mymind/tests/test_multi_provider_cache.py`、`mymindFrontend/src/App.vue`、`mymindFrontend/src/styles.css`、`mymindFrontend/src/lib/backends.js`、`mymindFrontend/src/lib/backends.test.js`、`mymindFrontend/package.json`、`mymindFrontend/README.md`。
- 改动摘要：在保留本地多 Provider、缓存、上下文预算、记忆并发控制和实验体系的基础上，融合 EchoMind 的九类细粒度业务意图、意图分组与来源分数、结构化主辅 Agent 路由、按意图触发的知识检索策略、RAG 降级状态语义、幂等知识导入、异步 Redis 与非阻塞 Chroma 访问、受生命周期管理的画像任务，以及路由和知识门控评测指标。`/chat` 以增量字段暴露诊断信息，Vue 同步展示并兼容 Java 旧响应；Java 实现未修改。真实模型验收进一步补充了 Windows 非 UTF-8 终端的启动横幅兼容、DeepSeek Anthropic 响应为空或达到 token 上限时的单次扩容重试、监控统计元数据过滤、人工升级主 Agent 一致性，以及高主分场景下明确辅助领域不被相对阈值过滤。新增根 README，补充三版本定位、启动配置、完整 API 示例、评测闭环、故障排查和安全注意事项。
- 验证结果：使用 `learn_claude` 运行 Python 全量测试，结果为 `31 passed`；关键 Python 模块导入检查通过；Vue Node 测试结果为 `2 passed`；Vite 生产构建成功；浏览器桌面和 390px 移动视口检查均无横向溢出，控制台无错误或警告。使用项目 `.env` 中的真实 API Key、Redis 和 Chroma 完成端到端验收：退款请求正确返回细粒度意图、实体、billing 路由和 `knowledge_status=used`；人工请求返回 `primary_agent=general`、`escalated=true` 和 `knowledge_status=skipped`；复合技术/扣款请求并行返回 billing/technical 主辅 Agent；最小评测 2/2 通过，`routing_accuracy`、`knowledge_gate_accuracy`、`intent_accuracy` 均为 1.0；同一知识文档重复导入两次后总片段数保持 6。测试用户记忆数据已定向清理。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。

## 2026-08-19

- 改动文件：`check.md`（新增方案，本次任务来源）、`mymind/core/retrieval.py`、`mymind/evaluation/retrieval_metrics.py`、`mymind/mcp/knowledge_base.py`、`mymind/mcp/tool_manager.py`、`mymind/api/main.py`、`mymind/experiments/rag_retrieval.py`、`mymind/experiments/rag_chroma.py`、`mymind/experiments/run_experiments.py`、`mymind/experiments/reporting.py`、`mymind/experiments/README.md`、`mymind/data/eval/build_rag_dataset.py`、`mymind/data/eval/rag_corpus.json`、`mymind/data/eval/rag_dataset.json`、`mymind/tests/test_rag_retrieval.py`、`mymind/tests/e2e_contract_server.py`、`mymindFrontend/src/lib/backends.js`、`mymindFrontend/src/lib/backends.test.js`、`mymindFrontend/src/App.vue`、`mymindFrontend/src/styles.css`、`mymindFrontend/package.json`、`mymind/.gitignore`、`.gitignore`、`review_rag_experiment.md`、`report.md`。
- 改动摘要：按 `check.md` 完成 RAG 优化实验与前后端契约改造。Python 侧新增 R0-R4 及四个消融变体配置、Markdown/段落感知 `500+80` 切块、稳定 `chunk_id/source_id/section_path`、内容去重、纯 Python BM25、加权 RRF、确定性重排回退、检索统计；`KnowledgeBase` 支持独立版本化 cosine collection 与真实 all-MiniLM-L6-v2；`/search`、`/knowledge/stats` 仅新增可选字段，旧字段全部保留；`/search` 继续使用 `top_k`。构建 22 篇 Markdown 语料与 128 条标注查询（8 大类别、每分区 4 改写、8 条无答案校准查询），实现 Recall@5/10、MRR@10、nDCG@10、P@3、关键事实覆盖率、无答案误召回、Top-3 重复率、冷/热 p50/p95 延迟、调用次数与 rerank fallback 率，以及按 partition 聚类 bootstrap 95% CI。前端搜索结果统一为稳定 `id` 作为 Vue key，按后端类型分别发送 `top_k`/`topK`，展示章节/融合分/来源，导入后自动刷新统计。
- 验证结果：`learn_claude` Python 全量测试 `45 passed`；前端 `npm test` `6 passed`；`npm run build` 生产构建成功；前端契约 E2E（真实 FastAPI 路由处理器 + 内存组件 + Edge headless）完成 Python 健康检查、Markdown 多章节导入、统计 12→16、非默认 Top-K=2 返回 2 个不同 chunk、聊天 `knowledge_status=used`、切回 Java 无回归、桌面 1440 与移动 390 无横向溢出、控制台 0 错误 0 异常。离线字符代理实验产物 `rag-retrieval-20260819-075528.json`：R1 Recall@5 已饱和至 1.0，R4 无法满足“相对 R1 +5pp”门槛（0.9833 vs 1.0），`overall_passed=false`。真实 Chroma all-MiniLM-L6-v2 层产物 `rag-retrieval-chroma-20260819-075549.json`：R4 Recall@5 0.9833 vs R1 0.9083（+7.5pp，cluster bootstrap 95% CI 0.025~0.142），MRR/nDCG 不下降、无答案误召回 0、Top-3 重复 0、p95 延迟 310.7ms vs R1 280.7ms（<1.2 倍），`overall_passed=true`；但该层 rewrite/rerank 仍为确定性代理，真实 LLM 层需另行确认。独立审查报告 `review_rag_experiment.md` 总体结论“不通过”，指出离线代理不可作为生产结论、source 级标注饱和、R1 +5pp 门槛结构矛盾、无答案指标被门控短路、缺少最终回答质量评测等问题；其中前端 E2E 缺失一项已在本轮补做并记录于 `artifacts/experiments/frontend-e2e-contract.json`，其余问题列为后续修订项。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。
