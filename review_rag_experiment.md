# mymind RAG 实验与实施方案 高级审查报告

> 审查对象：`check.md` 方案 + 已实现代码（retrieval.py / knowledge_base.py / tool_manager.py / api/main.py / retrieval_metrics.py / rag_retrieval.py / reporting.py / build_rag_dataset.py + 两个 JSON / backends.js / backends.test.js / App.vue / test_rag_retrieval.py）+ 最新产物 `mymind/artifacts/experiments/rag-retrieval-20260819-073422.json`。

## A. 总体结论

**不通过（实验结论目前不成立，需按 F 节修订后重跑）**

1. 最新实验产物中 R4 作为候选在自身指标上全面不优于 R1：Recall@5 0.9833 < R1 的 1.0，fact_coverage 0.9833 < 1.0，MRR@10 差值 CI 跨 0（−0.036~+0.043）；`overall_passed=false`，按现有门槛 R4 被否决是“自洽”的，但底层是**确定性 char n-gram 代理 + 手写改写/重排代理 + 合成延迟模型**，这个否决与生产链路（Chroma all-MiniLM、LLM 改写/重排、真实延迟）之间不存在可辩护的因果连接。
2. 验收门槛存在结构性矛盾：要求 R4−R1 的 Recall@5 ≥ +5pp，而 R1 实测 1.0（封顶），该条件在数学上不可能满足；同时 Recall@5/@10 在几乎所有变体上≈1.0，指标饱和、无区分度，bootstrap“CI 不跨 0”失去意义。
3. 评测只覆盖 source 级粗粒度的召回/排序，完全没测证据定位与最终回答质量；且前端只有 mock 单测（backends.test.js），不存在真实 Python 后端驱动的浏览器端到端工作流，验收规则第八条“必须使用真实 Python 后端”一条都未执行。
4. 代码层（变体开关、稳定 ID、去重、RRF、前端字段规范化、契约增量字段）实现较完整、可读，这部分可以保留复用；问题集中在实验设计、度量定义、统计与验收可信度。
5. 结论：当前产物只能证明“离线代理上这套脚手架能跑通并自洽”，不能证明任何生产检索收益；补齐真实 embedding/LLM、修正指标与门槛、补前端 E2E 之前，不得据此做生产决策。

---

## B. 严重问题（按风险排序）

### P0-1：离线“确定性代理”替换了全部被测对象，结论无法迁移到生产

- **问题**：`experiments/rag_retrieval.py` 用 `DeterministicVectorStore`（字符 n-gram 余弦）代替 all-MiniLM-L6-v2，用 `REWRITE_MAP` 字典代替 LLM 改写，用 `rerank_score`（IDF 词覆盖）代替 LLM rerank，用 `LATENCY_MODEL` 常量代替真实延迟；而验收门槛（acceptance）与 `overall_passed` 全部建立在这套代理产出的数字上。
- **结论如何失真**：最关键的三个被优化对象——embedding 语义匹配、LLM 改写、LLM rerank——恰是代理与真实差异最大的部分。离线向量代理会刻意弱化中文二元组以“让 BM25 显收益”（`_vector` 注释明示），这是**为让结论好看而人为制造了检索难度差异**；改写代理直接注入语料词（“多久”→“时效”，而“时效”正是章节标题“退款审核时效”的成员），本质是查询→语料泄漏。报告虽“声明”了代理身份，但声明不能把代理结论变成生产结论。
- **修订**：把实验分层为两套互不混淆的产物：(a) **offline/unit 层**（当前这套）只用于验证脚手架逻辑正确、去重/RRF/chunk 行为正确，`artifact_type` 改名 `rag-retrieval-offline-unit`，**禁止其生成 acceptance/overall_passed**；(b) **真实层**：用 `KnowledgeBase(variant=...)` 真实 Chroma collection + 真实 embedding + `MCPToolManager.rewrite_query/_rerank` 跑同一数据集，只由真实层产出 acceptance 与 CI。无法联网/无 API Key 时，实验如实标记“未完成”，而不是用代理顶替。

### P0-2：验收门槛“R4 vs R1 提升 5pp”与事实自相矛盾

- **问题**：`evaluation/retrieval_metrics.acceptance_report` 中 `recall_at_5_vs_r1_gte_5pp = r4 - r1 >= 0.05`。R1 实测 Recall@5=1.0，任何候选都不可能再高 5pp；且 R4 实测 0.9833 反而低于 R1。
- **结论如何失真**：该门槛要么永久失败（不可能达成的目标），要么迫使实验者去“压低 R1”换取差值，属于诱导调参的验收设计。产物 `failures` 永久包含该项，`overall_passed` 失去开关意义。
- **修订**：把“相对 R1 提升 5pp”改为对**有区分度指标**的非劣/优势组合：主判据改 nDCG@10、MRR@10 与 Precision@3 的提升（这些 R1 未饱和），Recall@5 只做非劣约束（`r4 >= r1 − 0.02`，即不下降超 2pp）；或先把语料/查询难度提升到 R1 Recall@5 ≈ 0.75~0.85 再保留“5pp”门槛。

### P0-3：无答案误召回率实质测的是“门控”，不是“检索”

- **问题**：`run_variant` 对 `no_answer` 查询先执行 `KnowledgePolicy.decide(row["query"], None)`，被门控掉的查询直接把 `items=[]`，根本不进检索管道。`KnowledgePolicy._BUSINESS_KEYWORDS` 不含天气/电影/餐馆/药品，因此 120 条 test 里的 8 条 no_answer 查询**全部被门控**，`no_answer_false_positive` 恒等于 0。
- **结论如何失真**：`no_answer_fpr_lte_5` 是恒真的（0.0），它证明的是“关键词门控把无关问题挡了”，而完全没测“检索器对相似但无答案的问题是否会误召回无关文档”。check.md 要求的是检索层误召回率，这里被门控短路掩盖。
- **修订**：把“无答案误召回率”拆成两个量：(a) 门控精确率（可保留，单独命名 `gate_precision`）；(b) **旁路门控**直接跑 `pipeline.search` 的误召回率，配一组**语义相似但无答案**的干扰查询（如“退款会退到支付宝吗？”而语料只写“原支付账户”），用 `item_strength >= min_score` 阈值判定“误召回”，并报告阈值敏感性曲线而非单点。

### P1-1：Recall/MRR/nDCG/Precision 在 source 级标注下全部饱和，无法归因

- **问题**：数据集每条查询只标 1 个 source_id（少数标该 source 下多个 sections 仍合并为同一 source），Recall 定义为“Top-k 出现任一 relevant source”。语料 22 篇、每篇多章节，任一 source 命中即得 1。实测 Recall@10=1.0（全部变体）、Recall@5 0.95~1.0，Precision@3 贴天花板（单 source 时 1/3，实测 0.29~0.31）。
- **结论如何失真**：R0→R4 的任何差异（切块、BM25、改写、rerank）在这些指标上几乎为零，消融矩阵形同虚设；四个减法消融的差异主要落在噪声范围（见 C 节）。
- **修订**：改 **chunk 级标注**（每个相关 chunk 单独给 0/1/2 级 + 所属 section），指标改 chunk@k 的 Recall/nDCG；增加难度（多文档共现同主题词汇、需跨 section 综合、需精确数字），使 R1 Recall@5 降到 0.75 上下，并先做难度预检验（R1 不得封顶）再正式跑。

### P1-2：实验没有测“证据质量”与“最终回答质量”

- **问题**：整个指标栈（recall/precision/mrr/ndcg/fact_coverage/duplicate/fpr）都在检索排序层；`fact_coverage` 是“事实串出现在 Top-k 拼接文本里”，既不是证据定位（谁引用了谁），也不是答案正确性，更不是忠实度。check.md 要求“区分召回、排序、证据质量和最终回答质量”，实现完全没有落地后两层。
- **结论如何失真**：无法回答“Top-3 注入后客服回答是否更好/是否幻觉/是否引错 chunk”这一真实业务问题；且“不接受仅用 LLM Judge 证明检索提升”的另一面——**连 LLM Judge 都没有**，只有检索代理分数。
- **修订**：增加两层评估：(a) 证据层——对每条查询标注 `required_evidence_chunk_ids`，测 Top-3 是否包含能独立支撑答案的 chunk（`evidence_hit@3`），并做 citation 一致性抽查；(b) 回答层——固定 prompt/model，Top-3 注入 vs 不注入，对 date/数字/布尔/枚举型答案先跑**规则判分器**测 `answer_accuracy`，LLM Judge 仅作辅助，且必须带 3 人标注子集做 judge-agreement 校验，禁止单独作为证据。

### P1-3：前端 E2E 完全缺失，只有 mock 单测

- **问题**：仓库仅 `mymindFrontend/src/lib/backends.test.js`（Node `node:test` + 替换 `global.fetch` 的 mock）与 `mymind/tests/test_rag_retrieval.py`（`TestClient` + Fake 对象的单元/集成测试）。`package.json` 无 Playwright/Cypress；`.edge-e2e/` 只是浏览器用户数据目录而非测试代码。验收规则第八节“必须使用真实 Python 后端完成前端端到端验收”的 8 个步骤一条都未执行。
- **结论如何失真**：无法发现真实浏览器下 `top_k`/`topK` 参数、duplicate Vue key、snake_case 字段、导入后统计刷新、busy 状态卡死、390px 溢出等问题；“前端兼容”目前只被类型化 mock 断言，与实际工作流等价性未验证。
- **修订**：见 E 节 E2E 清单与 F 节落地步骤。

### P2-1：`cosine` 开关在离线实验里是空转变量

- **问题**：`build_items` 只把 `collection_metadata = {"hnsw:space": "cosine" if config.cosine else "l2"}` 写进报告，离线 `DeterministicVectorStore` 根本不读该配置，所有变体用同一 n-gram 余弦。R0(l2) 与 R1(cosine) 的差异全部来自 dedup/fallback，cosine 的收益为 0 却计入“R1 修正基线”。
- **结论如何失真**：check.md 明示 R1 = R0 + cosine，而实验实际没测 cosine；R0 vs R1 的公平比较前提被破坏。
- **修订**：离线层显式声明“不覆盖 cosine，该维度留待真实层”，并从离线 `VariantConfig` 归因中剥离此维度；真实层用 `KnowledgeBase(variant=...)` 新 collection 实测 cosine 差异。

### P2-2：Bootstrap 对“模板改写查询”假独立 + 二元饱和指标 CI 失效

- **问题**：`paired_bootstrap_delta` 以**查询**为抽样单元，但每个模板的 4 条改写共享同一条标注、词汇高度相关，实际有效独立样本 ≈ 28 个分区而非 120；且对 recall 这种 0/1 且几乎全为 1 的指标，CI 天然坍缩（产物中大量 `ci_low=0.0, ci_high=0.0`）。
- **结论如何失真**：CI 宽度被系统性低估，“不跨 0”与“+5pp”的统计判断不可信；128 条查询的“统计效力”是虚的。
- **修订**：改用 **cluster bootstrap（按 partition 重采样）**；先做功效分析（在 R1 未饱和的 nDCG/P@3 上，估计达到 80% 功效检测 +0.03 所需分区数）；CI 不得用于判定饱和指标。

### P2-3：改写代理泄漏 + 重排代理内置对 R2+ chunker 的偏好

- **问题**：`REWRITE_MAP` 的替换词直接取自语料关键词（“时效”“退回原支付账户”“权限不足”等），使改写链路在离线层天然偏向命中；离线重排代理 `rerank_score` 用 `section_path` 词命中加分，而 `section_path` 只在 markdown 模式（R2+）下存在——等于**重排代理内置了对 R2+ chunker 的偏好**，污染“chunking”的归因。
- **结论如何失真**：改写/重排收益在离线层被结构性抬升或偏置，即便数值接近 R1，也不代表真实 LLM 行为。
- **修订**：离线改写用**不含答案词的通用同义映射且与语料词汇去交集**；离线重排代理改为纯 query-vs-content 覆盖，不使用 section_path（或把 section 信号独立成查询变量单独消融）；并在报告里输出每个改写实例供人工审计。

---

## C. 因果归因审查

### C.1 每个变体实际改变的变量（依据 `variant_config`）

| 变体 | chunk_mode | overlap | cosine | stable_ids | dedup | hybrid(BM25+RRF) | rewrite | llm_rerank | 确定性回退 |
|---|---|---|---|---|---|---|---|---|---|
| R0 | sentence | 0 | 关 | 关 | legacy_repr | 关 | 开 | 开 | 关 |
| R1 | sentence | 0 | 开 | 开 | chunk_id | 关 | 开 | 开 | 开 |
| R2 | markdown | 80 | 开 | 开 | chunk_id | 关 | 开 | 开 | 开 |
| R3 | markdown | 80 | 开 | 开 | chunk_id | 开 | **关** | **关** | 开 |
| R4 | markdown | 80 | 开 | 开 | chunk_id | 开 | 开 | 开 | 开 |

### C.2 归因正确性问题（真实缺陷）

1. **R0→R1 一次翻转 4 个变量**（cosine、stable_ids、dedup、确定性回退）。按“一次只变一个变量”的消融要求，这 4 项从未被单独拆开测过；且 cosine 在离线层空转（P2-1），R1 相对 R0 的“提升”实际只剩 dedup + 回退的混合效应，无法拆解。
2. **R2→R3 同时翻转两个变量**：+BM25/RRF 的同时关闭了 rewrite 与 rerank。于是 R3 与 R2 的差异是“加 BM25 减改写减重排”的**净效应**，不能归因到 BM25。正确做法是 R3 = R2 + BM25（其余全同），另由 `r4-no-bm25`（= R4−BM25）承担 BM25 的消融角色。
3. **R4 的四个减法消融在饱和指标上无信号**：产物里 `r4-no-rewrite` 与 R4 的 Recall/MRR/nDCG 几乎逐位相同，说明改写、rerank、BM25、overlap 每个变量在本数据集上贡献≈0——这不是“变量无效”的可靠结论，而是**指标与难度饱和**导致的（P1-1）。
4. **生产链路未被“固定”**：check.md 要求固定 embedding/回答模型/prompt/Top-K。离线层固定了 Top-K=10 与同质代理；但真实生产 `/search` 默认 `top_k=5`、`_build_knowledge_context` 固定 Top-3 注入、改写数 `n=3`，离线用 recall_k=20 与 `_normalize_queries[:4]`，参数面不一致，真实层尚缺。

### C.3 R0 vs R1 是否公平：不公平

- R0 的问题不在“不公平”本身而在**回放不忠实**：离线 `build_items` 对 R0 也走 `Chunker`，`source_id_for(title, content)` 生成稳定 source_id，而生产 `KnowledgeBase._legacy_chunk_records` 用的 source_id 是 `"legacy:{md5}"`（逐 chunk 唯一）。因此离线 R0 的 source 级标注能命中，而真实生产 R0 的 source_id 与标注 source_id 永远对不上，真实层若照搬，R0 的 Recall 会被打成 0。离线把 R0 美化了。
- 另外，R0 在生产里走 `dedupe_items(mode="chunk_id")`（tool_manager 第 360 行）而非 `legacy_repr`——离线用 `legacy_repr` 回放，与当前生产代码不一致，R0 的 59.17% Top-3 重复率很可能是对一段已不存在旧代码的回放，夸大了 R0 的缺陷以便 R1“修复”看起来有理。

**修订**：R0 必须用当前生产代码路径（`KnowledgeBase().__init__` + `MCPToolManager.search_with_rewrite`）逐查询回放并记录真实 source_id 形态；source 匹配要兼容 `legacy:` 前缀（先按 title 归一化映射回 source_id），否则标注体系与 R0 不兼容。

---

## D. 数据、指标和统计审查

1. **标注质量**：`build_rag_dataset.py` 全由模板程序化生成，**无人工确认字段、无 annotator、无互评**，自检只用 `assert fact in source_doc`（事实是否在源文档），不检验事实是否只在唯一 chunk 出现，也不检验改写查询是否真的同义。check.md 要求“LLM 可生成候选、标签必须人工确认”，此点未落地。需在每条记录加 `reviewed/reviewer/verified_at`，未 review 者不进 test 集。
2. **改写泄漏（两种）**：(a) 同模板 4 条改写共享 partition——**做对了**（分区正确、改写同区，不会跨区泄漏）；(b) 离线改写代理向语料词泄漏——**没做对**（见 P2-3）；(c) 重排/BM25 代理能读全集 IDF（`build_idf`），生产 rerank 也全文可读，这不构成标签泄漏，但**改写代理**直接映射语料关键词属间接泄漏。
3. **无答案样本**：标定/测试分离做对了（noanswer-01/02 为 calibration、03/04 为 test，8+8）。但 120 条 test 中的 8 条无答案全部被门控短路（P0-3），真正的检索层误召回率为空测。
4. **120 条统计效力**：名义 120 条，实际有效分区 28 个（每模板 4 改写共享一标注），且主要指标饱和、差异接近 0；120 条对“检测 nDCG +0.03 量级差异”功效不足。必须做功效分析、按集群估计，并扩分区（建议 ≥40 分区、含 8~10 项难查询）。
5. **bootstrap CI**：实现方法本身正确（百分位法、固定 seed），但抽样单元应为 partition（cluster bootstrap），且对饱和二元指标失效；CI 边界取 `int(tail*len)-1` 的修正可保留。
6. **重复 chunk 定义**：`result_chunk_key` = chunk_id 优先、无 id 才用规范化内容。后果有二：(a) R0 用 legacy_repr 去重失败导致 59% 该指标——但这是旧代码回放问题；(b) **overlap 造成的“内容重复”完全测不出来**：R2/R4 的 overlap 会让相邻 chunk 共享 80 字符内容，但 chunk_id 不同，`duplicate_top3_rate` 仍为 0，等于宣告“无重复”而实际注入了重复内容。需要并存两个指标：`duplicate_chunk_id_rate`（同 id 重复）与 `near_dup_content_rate`（Top-k 间 Jaccard/最长公共子串相似度>阈值的占比），后者对 overlap 才有区分力。
7. **`relevant_sections` 是死字段**：标了但任何指标都没用；markdown chunker 的 `section_path` 与标注 section 字符串格式一致时，本可做 section 级 precision/recall（更有诊断价值），却未实现。

---

## E. 前后端契约审查

### E.1 可能破坏 Vue/Java 兼容的变化

1. **`/search` 新增 `chunk_config`（dict）与 `index_version`**：均为增加、不删旧字段，符合“仅增量”。前端 `normalizeSearchResponse` 正确收容。✅ 无破坏。
2. **`/knowledge/stats` 新增 `index_version/chunk_config`**：`total_chunks` 保持数值。✅ `loadStats` 里 monitor 分支可能覆盖 `statusText` 属旧行为，非契约破坏。
3. **`/search` 的 `query`/`top_k` 位置**：FastAPI 声明 `async def search(query: str, top_k: int = 5)`，二者都是 query string 参数；前端用 `URLSearchParams` 发 POST 无 body，与现契约一致。**风险**：若有客户端改用 JSON body `{query, top_k}` 会 422。建议显式声明 `Query(...)` 或同时支持 body 模型，并用契约测试冻结。⚠️ 低风险但应固定。
4. **`chunk` 字段歧义**：Python 返回 `chunk=chunk_index`（数值索引），Java 旧响应 `chunk` 语义可能不同。`normalizeSearchResults` 用 `toNumber` 归一基本安全，但若 Java `chunk` 是字符串 ID 会被转 0，可能触发 Vue key 回退。需在 `backends.test.js` 补“Java 返回 `chunk:'abc'`”用例，明确预期行为。⚠️
5. **`score` 在非 cosine 默认 collection 下可能为负**：`_format_results` 恒用 `1.0 - dist`，生产默认 collection 是 L2 距离（`cosine=False`），距离可 >1 → 负 score，前端直接展示。R0 既有行为，但引入 cosine 前应先修 score 归一化（l2 用 `1/(1+dist)` 或按 metric 分支），否则排序展示把负分显示为混乱分数。⚠️
6. **Top-K 参数**：已按后端类型发送 `top_k`/`topK`，有单测覆盖，方向正确；但端到端没验证“Python 用非默认 top_k=7 返回 7 条”（Python 单测只断言 `manager.calls[-1][2]==7`，是 mock）。需真实后端验证。

### E.2 应补的契约测试（后端 `tests/`）

- `/search`：query string 形式与 JSON body 形式的兼容性冻结；top_k=1/20 边界；`requested_top_k` 与 `returned` 一致。
- `/knowledge/stats`：`total_chunks` 必为 int 且 ≥0。
- `/knowledge/upload`：`.md` 多章节文导入后 `total_chunks` 增加且查询命中新 doc，`chunk_id/source_id/section_path` 均非空。
- `/chat`：`knowledge_status` 枚举值集合冻结（skipped/used/empty/error/degraded），前端对未知态需有兜底展示。
- 前端 `backends.test.js` 补 `fusion_score: null`、`retrieval_sources` 缺省两例。

### E.3 应补的浏览器 E2E 场景（真实 Python 后端驱动）

用 Playwright（`learn_claude` 环境 + `mymindFrontend` 构建产物），逐条跑 check.md 第八节 8 步，并额外加：

- 断言 `searchResults` 中两个 item 的 Vue `key` 不同（无 duplicate-key console 警告，用 `page.on('console')` 监听）。
- 网络拦截断言：发给 Python 的请求 URL 含 `top_k=7`、给 Java 的含 `topK=7`（而非仅单测）。
- 导入失败（构造 413/400）时 UI 显示错误且 `busy` 恢复、按钮可再次点击（防“永久 busy”）。
- 390px 移动视口截图 + 横向溢出检测（`scrollWidth <= clientWidth`），复查 `.result-item`、`.topk-control` 无换行溢出。

---

## F. 修订后的可执行方案（不引入重型平台）

### F.1 实验分层（最关键）

1. `experiments/rag_retrieval.py` 输出 `artifact_type="rag-retrieval-offline-unit"`，并**删除其 acceptance/overall_passed 计算**（离线只输出指标表，不带“通过/不通过”结论）。
2. 新增 `experiments/rag_retrieval_prod.py`：用 `KnowledgeBase(chroma_*, variant=name)` 建 9 个版本化 collection → `MCPToolManager` 注册 `knowledge_search` → 对 128 条逐查询跑 `search_with_rewrite`（真实改写/重排、缓存关闭、固定 seed、所有变体在同一进程串行以固定模型温度与索引状态）→ 用同一套 `retrieval_metrics` 产出 acceptance + cluster bootstrap。**只有该层能出结论**，报告明确区分 `prod` 与 `offline-unit` 两份产物。

### F.2 指标与门槛修订

- 改用 chunk 级 `relevant_chunk_ids` + `relevance`（chunk_id→0/1/2）与 section 级标签。
- 主指标：`recall_chunk@5/@10`、`nDCG@10(chunk)`、`MRR@10`、`Precision@3(chunk)`、`evidence_hit@3`、`answer_accuracy(rule)`。
- 门槛改为（可执行）：
  - `recall_chunk@5 >= 0.88` 且 `recall_chunk@5 >= R1 − 0.02`（非劣，替代不可能成立的 +5pp）；
  - `nDCG@10`、`MRR@10` 相对 R1 至少一项 **+0.03 且 cluster-bootstrap 95%CI 下界 > 0**；
  - `no_answer_FPR(旁路门控) <= 5%`；
  - `near_dup_content_rate(top3) <= 5%` 且 `duplicate_chunk_id_rate == 0`；
  - `p95 真实延迟 <= R1 × 1.2`（真实层用 `asyncio` 实测冷/热，禁止合成 LATENCY_MODEL）。
- 先跑一次 **R1 难度预检**：若 `recall_chunk@5 >= 0.95` 或 nDCG ≥ 0.95，则回 F.3 加难，不得带饱和数据进正式验收。

### F.3 数据修订（`build_rag_dataset.py`）

- 增加难查询模板：跨 section 综合、需精确数值、多篇同主题干扰（如两篇都有“退款时效”不同数值），使 R1 recall_chunk@5 落 0.70~0.85。
- 每条记录加 `reviewed/reviewer/verified_at`；事实与 `required_evidence_chunk_ids` 由人工确认；保留 `must_recall_facts`，另建 `answer`（可规则判分的 gold 答案）字段。
- `no_answer` 增加“语义相似无答案”干扰子集；改写分区保持现状（已正确，不需改）。
- 自检升级：断言每个 `evidence_chunk_id` 在对应变体的切块集中唯一存在；人工复核表以 CSV 存到 `data/eval/`。

### F.4 变量与消融重排（修正 C.2）

- R1 = production 基线（直接调生产函数，不改任何开关）；R1' = 仅工程修正（cosine + 稳定 ID + chunk_id 去重），并逐项拆分：R1a（仅 cosine）、R1b（仅 stable id）、R1c（仅 dedup），确保“一次只变一个变量”。
- R2 = R1' + markdown/overlap；R3 = R2 + BM25/RRF（**保留 rewrite/rerank 不变**）；R4 = R3 + 全开。四个减法消融：`R4−rewrite`、`R4−rerank`、`R4−BM25`、`R4−overlap`，保证每个减法只关一个开关。
- 所有变体固定：embedding（all-MiniLM）、回答模型、prompt、Top-K、rewrite n=3、recall_k 一致、关闭结果缓存；延迟实验冷/热分开报告并注明是否命中 LLM 缓存。

### F.5 统计与报告

- `paired_comparison` 改 cluster bootstrap（按 `partition` 重采样），保留固定 seed 与百分位法。
- 每个变体的报告页增加：per-partition 指标表（暴露同一分区 4 改写的相关性）、改写实例清单、fallback/failure 率、`rerank_failure_rate` 的分项。
- 报告结论模板显式声明四层：召回 / 排序 / 证据质量 / 最终回答质量——缺一层则写“未测”，不得以相邻层指标替代。

### F.6 前端落地（代码已完成，缺验证）

- `backends.js`/`App.vue` 的规范化与稳定 id 已实现正确，保留；补 `backends.test.js` 的 `fusion_score:null`、`chunk:'abc'`、Java 旧响应三例。
- 用 Playwright + 真实 Python 后端完成 E.3 的 8 步 + 4 项额外检查；将 E2E 脚本纳入 `learn_claude` 的测试命令，输出 pass/fail 给报告。
- `report.md` 记录本次修订（按 AGENTS.md 约定）。

---

## G. 阻碍实验成立的待确认问题

以下问题会实质改变实验设计，需先确认再继续，否则 F 节方案仍需相应调整：

1. **真实 Chroma + all-MiniLM 是否可在本环境离线/在线运行？** 若嵌入模型对中文语料的语义区分度使 R1 的 chunk 级 Recall 初始即饱和（≥0.95），则 F.3 的加难方向（语料规模、同义词干扰、跨 section 综合）必须据此确定；若无法运行真实 embedding，则“检索收益”实验无法成立，只能降级为单元级行为验证。
2. **LLM 改写/重排的稳定性与调用预算是否允许固定温度下对 128 条 × 9 变体重复跑？** 若模型随机性或 API 配额使重排结果不可复现，需要明确每次评测的重复次数（如 n=3 取均值）与成本上限，否则真实层结论没有可复现性。
3. **生产 R0 的 source_id 现状确认**：还原后生产 chunk 的 source_id 究竟是不可跨 chunk 的 `legacy:{md5}` 还是会在本轮顺带升级到稳定 source_id？这决定 C.3 的“R0 回放 + source 映射兼容层”是否需要写、以及 R0 与 R1 的对齐口径。
4. **验收目标是否允许“重新定义 R1 为工程修正后基线”**：若业务上 R1（cosine+稳定ID+去重）是应当直接上线的确定性修复，则 R0 与 R1 的对比是“修复验证”而非“消融”，+5pp 门槛应整体废除——需确认这一口径。
5. **是否接受用规则判分 + 人工标注子集替代 LLM Judge 作为回答质量层证据**：若团队坚持 LLM Judge，则必须同步提供 3 人标注子集做一致率校验，二者选其一，不能单独用 LLM 自评证明检索或回答提升。
