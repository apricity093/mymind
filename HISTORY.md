# HISTORY

## feat(rag): 完成 check.md R0-R4 检索实验、前端契约适配与评测体系 — 2026-08-19

- 新增 R0-R4 及消融检索变体（段落切块、稳定 ID、去重、BM25+RRF、确定性重排回退），以及 128 条标注查询与指标/统计/聚类 bootstrap 评测。
- 后端新增字段均为可选增量字段，旧 `/search`、`/knowledge/*`、`/chat` 契约保持；前端统一稳定 chunk id 并分别发送 `top_k`/`topK`。
- 验证：Python `45 passed`；前端 Node `6 passed`；生产构建成功；浏览器契约 E2E 通过；真实 Chroma all-MiniLM 层 R4 通过验收（rewrite/rerank 仍为确定性代理）；独立审查报告与后续修订项见 `review_rag_experiment.md`。
