# mymind RAG 优化实验、前端兼容与模型审查方案

## 一、目标与约束

验证切块、混合召回、查询改写和重排是否提高正确证据召回率，同时保证 Python 后端修改后，现有 Vue 前端可以正常完成：

- Python/Java 后端切换
- 知识库检索
- 文本、Markdown、JSON 文档导入
- 知识库统计刷新
- 聊天及 RAG 状态展示
- 桌面端和移动端使用

本轮保留 Chroma 默认 `all-MiniLM-L6-v2`，允许建立新 collection 并从原始文档重建索引。

## 二、实验设计

固定相同语料、embedding、回答模型、prompt、Top-K 和测试查询，仅改变目标变量。

| 变体 | 配置                                                         |
| ---- | ------------------------------------------------------------ |
| R0   | 当前生产实现                                                 |
| R1   | R0 + cosine collection、稳定 chunk ID、正确去重及确定性重排回退 |
| R2   | R1 + Markdown/段落感知的 `500 字符 + 80 overlap` 切块        |
| R3   | R2 + BM25/向量加权 RRF                                       |
| R4   | R3 + 查询改写 + LLM rerank                                   |

消融实验：

- `R4 - rewrite`
- `R4 - rerank`
- `R4 - BM25`
- `R4 - overlap`

每个变体使用独立、版本化 Chroma collection。检索质量实验关闭结果缓存；延迟实验分别报告冷缓存和热缓存。

## 三、数据与指标

建立不少于 120 条标注查询，覆盖退款、物流、支付、账户、订阅、API 接入、401/403/500 错误码及无答案问题。

每条查询标注：

- `relevant_source_ids`
- 相关章节
- 必须召回的关键事实
- `0/1/2` 相关性等级
- 查询类型和业务类别

同一问题的不同改写必须位于同一数据分区，防止 dev/test 泄漏。LLM 可以生成候选问题，但标签必须人工确认。

主要指标：

- Recall@5、Recall@10
- MRR@10、nDCG@10
- Precision@3
- 关键事实覆盖率
- 无答案误召回率
- 重复 chunk 率

守护指标：

- p50/p95 检索延迟
- Chroma 和 LLM 调用次数
- token 消耗
- rerank 解析失败率及 fallback 率
- 前端契约与端到端操作通过率

采用逐查询配对比较和 bootstrap 95% 置信区间，并按查询类型、业务类别分别报告。

## 四、后端兼容契约

现有前端依赖的字段不得删除、改名或改变类型：

- `/search`：继续返回 `{query, results, reranked}`；每个结果保留 `title`、`content`、`score`、`chunk`。
- `/knowledge/add`、`/knowledge/upload`：保留 `message`、`added_chunks`、`processed_chunks`、`total_chunks`。
- `/knowledge/stats`：保留数值型 `total_chunks`。
- `/chat`：保留 `conv_id`、`response`、`knowledge_used`、`knowledge_status`、`knowledge_reason` 及现有意图和 Agent 字段。

新增字段只能是可选增量字段：

- `chunk_id`
- `source_id`
- `section_path`
- `fusion_score`
- `retrieval_sources`
- 索引版本和切块配置

Python `/search` 使用 `top_k`，Java 使用 `topK`。前端适配层必须根据后端类型发送正确参数，不能继续依赖 Python 忽略 `topK` 后使用默认值。

## 五、前端调整

在 `mymindFrontend/src/lib/backends.js` 增加搜索结果规范化：

- 将 `chunk_id`、`chunkId` 或旧 `id` 统一为稳定 `id`。
- 将 snake_case/camelCase 的新增诊断字段统一。
- 缺少新增字段时保持 Java 旧响应兼容。
- Python 请求发送 `top_k`，Java 请求发送 `topK`。

检索列表使用稳定 chunk ID 作为 Vue key；不能只使用 `title`，因为同一文档可能返回多个 chunk。界面继续显示标题、正文和综合 `score`，可补充章节、向量/BM25 来源，不直接展示内部冗长元数据。

导入成功后自动刷新知识库统计；索引构建或导入失败时显示明确错误，不能留下永久 busy 状态。

## 六、验收规则

R4 成为候选方案必须满足：

- Recall@5 `>= 90%`。
- 相对 R1 至少提升 5 个百分点，且 95% CI 不跨 0。
- MRR@10 和 nDCG@10 相对 R1 不下降超过 2 个百分点。
- 无答案误召回率 `<= 5%`。
- Top-3 重复 chunk 率为 0。
- p95 延迟不超过 R1 的 1.2 倍。
- 所有 Python 单元、集成和前端契约测试通过。

前端端到端验收必须使用真实 Python 后端完成：

1. 切换到 Python 后端并通过健康检查。
2. 导入一篇包含多个章节的 Markdown。
3. 知识库统计正确增加。
4. 检索命中新文档并显示多个不同 chunk，不出现 Vue duplicate-key 警告。
5. 使用非默认 Top-K 时，实际结果数量与参数一致。
6. 聊天请求显示 `knowledge_status=used` 和正常回答。
7. 切回 Java 后端后旧功能不回归。
8. 在桌面和 390px 移动视口检查无重叠、溢出和控制台错误。

验证命令分别使用 `learn_claude`、前端测试和生产构建。所有文件修改完成后更新根目录 `report.md`。

## 七、交给其他模型的审查 Prompt

```text
你是信息检索、RAG 实验方法和前后端契约方面的高级审查者。请严格审查下面的 mymind 实验与实施方案。

项目事实：
- Python RAG 当前链路为门控、查询改写、多路 Chroma 召回、合并、LLM rerank、Top-3 注入。
- 当前切块约 500 字符、无 overlap。
- 本轮保留 Chroma 默认 all-MiniLM-L6-v2。
- Vue 前端同时支持 Python 和 Java 后端。
- Python /search 参数为 top_k，Java 为 topK。
- 前端当前以 item.id || item.title 作为检索结果 key，而 Python 结果没有稳定 id。
- 后端升级必须保持现有聊天、搜索、导入和统计响应字段兼容。

待审查方案：
【粘贴完整方案】

请重点审查：
1. R0-R4 是否能正确归因切块、BM25、改写和 rerank 的收益。
2. R0 与修正基线 R1 的比较是否公平。
3. 数据集是否存在改写泄漏、测试集调参和 LLM 自我评判。
4. Recall、MRR、nDCG、无答案误召回及事实覆盖率定义是否充分。
5. 候选深度、RRF、chunk size 和 overlap 是否缺乏依据。
6. 120 条查询是否具备足够统计效力。
7. 缓存、模型随机性、索引版本和执行顺序是否污染实验。
8. 默认 embedding 对中文语料造成的结论边界。
9. Python 接口新增字段是否保持 Vue 和 Java 兼容。
10. 前端是否真实验证了搜索、导入、统计、聊天和后端切换。
11. 是否遗漏重复 Vue key、snake_case/camelCase、Top-K 参数或错误状态问题。
12. 验收门槛是否可能诱导只优化单一指标。

按以下格式输出：

A. 总体结论
- 通过 / 有条件通过 / 不通过
- 3-5 句话说明依据。

B. 严重问题
- 按 P0、P1、P2 排序。
- 每项说明问题、结论如何失真、具体修订办法。

C. 因果归因审查
- 列出每个变体改变的变量。
- 给出最小且充分的消融矩阵。

D. 数据、指标和统计审查
- 检查标注、泄漏、无答案样本、统计效力及置信区间。

E. 前后端契约审查
- 列出可能破坏现有 Vue 或 Java 兼容性的接口变化。
- 给出应增加的契约测试和浏览器场景。

F. 修订后的可执行方案
- 包含变体、数据、指标、统计方法、前端验收和发布规则。
- 不引入超出本项目规模的重型平台。

G. 阻碍实验成立的待确认问题
- 只列会实质改变实验设计的问题。

审查要求：
- 区分召回、排序、证据质量和最终回答质量。
- 不接受仅用 LLM Judge 证明检索提升。
- 不接受只测后端、不测现有前端工作流。
- 所有批评必须附带可执行修订。
```