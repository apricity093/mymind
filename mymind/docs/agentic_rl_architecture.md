# 共享客服策略架构与运行

应用层 `/chat` 使用意图识别、主辅 Agent、工具式 RAG 和记忆。独立 `api.policy_app:app` 加载经过 SFT/RL 的 Qwen 模型，接收已给定的账单或技术角色。共享策略学习检索与终止决策，应用层负责业务路由；两条链路分别评测。

## 模块与数据流

| 模块 | 职责 |
| --- | --- |
| `agents/customer_policy/protocol.py` | 公共任务、动作、观察与终止结果协议 |
| `environment.py`、`retrieval.py` | 每请求独立轨迹、检索预算、真实工具结果和可见证据 |
| `generation.py`、`runner.py` | 模型生成、工具往返、预算控制与原始结果保存 |
| `experiments/agentic_rl/sft_data.py`、`sft.py` | assistant 监督、工具观察 mask 与 LoRA SFT |
| `grpo.py`、`training.py`、`grpo_diagnostics.py` | 在线采样、同题组奖励、有效 token 损失、参考策略与更新诊断 |
| `tasks.py`、`grading.py` | 任务标准与生成后的评分 |
| `api/policy_app.py` | 启动装配、完整轨迹串行执行、状态与请求记录查询 |

任务输入包含角色、用户问题和可选材料。模型可以调用 `search`，取得实际可见的政策正文，再生成回答、请求补充信息或升级人工的结构化结果。策略支持根据工具观察继续决策；正式最终测试中未观察到补检，因而没有该能力的正向实验证据。工具状态、正文、引用及原始输出均记录，呈现模板不补答案。

标准中的“私有”指对被评测策略隔离：评分标准用于训练奖励或生成后的评测，不进入策略提示。公开的合成回归标准可用于检查评分器，不作为正式测试已经独立复核的证明。

## 所选训练与评测

基座为 `Qwen/Qwen3-4B-Instruct-2507`，固定 revision `cdbee75f17c01a7cc42f958dc650907174af0554`。LoRA 为 r16、α32、dropout 0，覆盖七类投影。

SFT 使用训练侧筛选的 158 条教师示范，训练 120 步；system、用户、工具观察及 padding 不参与直接监督。GRPO 从 SFT120 adapter 开始，新建优化器，在线同题采样四条轨迹，实际训练 80 步，按原 80 题开发验证选择 checkpoint-40。

正式 RL 采用 DAPO 有效模型 token 归一化、优势 `r−组均值`、`scale_rewards=none`、β=0、概率比裁剪 0.8～1.28。训练采样 temperature 0.8、top_p 0.95，正式推理贪心。训练器另支持逐轨迹归一化、组标准差和固定 SFT 参考 adapter 的 GRPO 式对照，详见[训练评测说明](../experiments/agentic_rl/README.md)。

原 600 题按 400 训练、80 验证、120 测试划分。158 示范来自训练侧 20 个组合；验证用于选 checkpoint，最终测试验证冻结选择。阶段 5 仅 C01 有人工确认，其余相关审核按实际 AI 来源报告。

正式 120 题 Base/SFT/RL 完整成功为 0/43/43，严重错误为 38/38/27，协议失败为 76/24/19。RL 新增 3 道成功也失去 3 道，D09/F04 各 0/15。单 seed、固定合成政策与有限评测规模限制保持，不据此声称稳定提升或未知政策泛化。配置对照与推理开销见[结果页](../../mymind-results.html)。

## 独立推理

推理需要匹配 revision 的本地基座、SFT120 或 RL40 adapter，以及按相同政策与检索参数构建的知识索引。上述资源和真实凭据不随仓库提供。配置模板为 [agent-dev.json](../api/configs/agent-dev.json)；配置内的模型、adapter、知识库和输出路径均相对 `MYMIND_WORK_ROOT`。

在具备 GPU 和模型依赖的环境安装 `mymind/requirements-agentic-rl.txt` 及后端基础依赖，然后从仓库根目录设置：

```powershell
$env:MYMIND_WORK_ROOT = (Resolve-Path '<运行资源目录>').Path
$env:MYMIND_AGENT_CONFIG = (Resolve-Path 'mymind/api/configs/agent-dev.json').Path
Set-Location mymind
python -m dotenv -f "$env:MYMIND_WORK_ROOT/service.env" run --no-override -- python -m uvicorn api.policy_app:app --host 127.0.0.1 --port 18000 --workers 1
```

`service.env` 自行配置 Embedding 与 `DEEPSEEK_API_KEY`，不提交凭据。运行前为配置选择新的 `output_dir`，并确认资源路径与索引就绪；模板不会下载所选 adapter 或自动重建正式索引。

`GET /status` 返回实际加载信息；`POST /requests` 接受 `role`、`user_text` 和可选 `records`；`GET /requests/{request_id}` 查询保存结果；`/docs` 提供接口示例。模型生成在工作线程执行，单 GPU 完整请求串行，每个请求新建环境。模型正常答错、协议失败和运行错误按实际结果保存。

## 仓库验证与资源边界

在根目录创建并激活 `environment-local.yml` 定义的环境后，运行 `python -m pytest -q`。协议、评分、mask、预算、模型接口和服务契约使用合成输入或替身，不需要本地 CUDA 训练依赖。依赖未公开历史轨迹和冻结作业文件的复核测试，在缺少对应记录时跳过并说明原因；本地保留证据完整时仍执行。

正式与校准数据集、各版本标准和来源记录，以及 158 条实际教师示范、token 导出和作业账本单独保存。少量自包含的合成开发样例位于 `experiments/agentic_rl/resources/`，用于协议与评分器回归。`scripts/verify_agentic_retained.py`、`verify_stage7_retained.py` 与结果页构建器用于持有这些记录的本地复核，不能在缺少原始记录时重建历史实验证据。

本地 CPU 回归不替代 GPU 训练验证。原 SFT 精确续训状态已清理，RL 断点恢复未经 GPU 实测；独立推理不依赖 optimizer 或 RNG 状态。
