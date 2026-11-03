# MyMind Python 学习路线

当前系统分为应用编排与共享客服策略两条链路。先读[系统架构](../../CONTEXT.md)，再按本页入口追踪实际代码。

## 1. 请求与路由

从 [api/main.py](../api/main.py) 的 lifespan 和 `/chat` 开始，观察记忆读取、意图识别、ContextBuilder、Request、AgentOrchestrator 与响应。当前问题独立于背景预算；应用按实际工具轨迹报告知识状态。

[intent_recognizer.py](../core/intent_recognizer.py) 使用 LLM、独立 embedding 与关键词进行领域融合。当前 HTTP 共享 Qwen embedding 组件，正常权重为 70/20/10；不可用时降级处理。结果缓存键保留完整消息及最近三轮历史。

## 2. Agent 与工具

[agent_orchestrator.py](../agents/agent_orchestrator.py) 管理主辅路由、监控可用性和模型工具循环；[tools.py](../agents/tools.py) 在调用边界验证模型参数，内部按结果契约读取字段。同请求的相同查询和 Top-K 使用 Tuple 键缓存。

工具轨迹与用量保存在请求内，专业 Agent 回退时保留已发生的调用。业务角色的退款或升级建议不表示执行了真实退款或创建工单。

## 3. 当前检索与历史基线

[indexed_knowledge_base.py](../mcp/indexed_knowledge_base.py) 是 HTTP 的当前知识库：原文保存在 SQLite，Qwen 与 MiniLM 分别建立主备索引，每条路线使用向量和 BM25 融合。主备预算默认 512/240 token，overlap 为 0。

[tool_manager.py](../mcp/tool_manager.py) 执行查询改写、并行召回、缓存、熔断、去重与重排。备用正文可进入结果并保留 degraded 标记；工具故障提示不作为依据。重排失败使用确定性排序。

[knowledge_base.py](../mcp/knowledge_base.py) 和 [retrieval.py](../core/retrieval.py) 保留 R0–R4 的历史基线构件；其默认 collection 和切块方式不等同于当前 HTTP 配置。

## 4. 记忆、缓存与 Skill

[conversation_memory.py](../memory/conversation_memory.py) 使用 Redis 保存工作记忆与摘要、Chroma 保存情景记忆与画像。[context_builder.py](../memory/context_builder.py) 默认按 8000 字符预算裁减背景。Skill 由角色与关键词选择，是模型输入规范，不保证输出符合所有业务规则。

[cache_store.py](../core/cache_store.py) 将运行时结构键与 Redis 持久化键转换分开。知识更新使旧缓存失效，并阻止更新前开始的查询把旧值写入新版本。应用缓存、Provider 缓存和请求缓存分别计量。

## 5. 单策略 SFT 与 RL

阅读 [customer_policy 协议](../agents/customer_policy/protocol.py)、环境与 runner，再进入 [rl](../rl/README.md)。角色由输入给定，策略决定检索和终止。SFT 仅直接监督 assistant token；正式 RL 按同题四轨迹计算 r−组均值，并采用 DAPO token 归一化。

终止输出要同时满足协议、业务事实与证据支持。结构合法、检索命中、完整成功属于不同指标。正式 RL 的完整成功未超过 SFT，严重错误有所减少；当前重构后的本地检查不代表重新取得历史 GPU 成绩。

## 6. 练习与验证

从根目录执行 `python -m pytest -q`，按 `test_intent_routing.py`、`test_rag_pipeline.py`、`test_agent_orchestration.py`、`test_memory_and_cache.py`、`test_rl_core.py` 阅读对应行为。需要本地历史记录的离线复核见 [Benchmarks](../benchmarks/README.md)。

启动和接口示例见[完整使用指南](完整使用指南.md)，实际实现入口见[重点代码](重点代码.md)。
