# 项目改动记录

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
