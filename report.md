# 项目改动记录

## 2026-08-08

- 改动文件：`mymind/agents/agent_orchestrator.py`、`mymind/api/main.py`、`mymind/core/intent_recognizer.py`、`mymind/core/llm_utils.py`、`mymind/core/cache_metrics.py`、`mymind/core/llm_gateway.py`、`mymind/evaluation/evaluator.py`、`mymind/experiments/`、`mymind/mcp/tool_manager.py`、`mymind/memory/conversation_memory.py`、`mymind/requirements.txt`、`mymind/tests/test_multi_provider_cache.py`、`mymind/.gitignore`、`AGENTS.md`。
- 改动摘要：增加多模型 LLM 网关与缓存指标支持，扩展缓存和记忆相关调用链、实验脚本及测试；实验运行产物改为由 Git 忽略，并从版本控制中移除已有产物。
- 验证结果：使用 `learn_claude` 虚拟环境运行测试，结果为 `17 passed`。
- 是否触及冻结清单：否；当前项目规则未定义冻结清单。
