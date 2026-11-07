# MyMind Python 后端

本目录提供客服Agent、混合RAG、记忆与工具编排，以及独立的Customer Policy训练和推理。`api.main`的`/chat`与`api.policy_app`分别运行，训练策略尚未接入`/chat`。系统边界见[项目说明](../README.md)和[架构设计](../CONTEXT.md)。

## 环境与启动

从仓库根目录创建本地环境：

```powershell
conda env create -f environment-local.yml
conda activate agentic_rl
cd mymind
Copy-Item .env.example .env
```

首次创建`.env`后填写模型、Embedding及Redis配置；已有文件直接编辑，不覆盖。配置项及默认值见[环境模板](.env.example)。

只启动依赖、在宿主机运行后端：

```powershell
docker compose up -d redis chromadb
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

用Docker运行完整应用：

```sh
docker compose up -d --build
docker compose logs -f mymind
docker compose stop
```

单独构建镜像可执行`docker build --target production -t mymind:local .`。Compose读取`.env`；`.dockerignore`避免把真实环境文件和本地缓存打进镜像。

## 两套依赖

| 文件 | 环境 |
| --- | --- |
| [requirements.txt](requirements.txt) | 本地应用、离线分析与测试；Conda和Docker共用 |
| [requirements-agentic-rl.txt](requirements-agentic-rl.txt) | 服务器CUDA、SFT/RL及独立策略服务 |

两套环境分别安装，不叠加。服务器的OpenAI SDK和Chroma版本与本地不同。应用镜像采用统一依赖清单，也包含离线分析和测试包。整理清单不代表已安装或升级环境。

## 模块与接口

| 目录 | 职责 |
| --- | --- |
| `api/` | 应用`/chat`、知识接口及独立策略`/requests`服务 |
| `agents/` | 角色路由、工具循环、升级与回复组合 |
| `agents/customer_policy/` | 训练与推理共用的协议、检索环境和执行器 |
| `core/`、`mcp/` | 模型网关、Embedding、混合检索与缓存 |
| `memory/`、`monitor/` | 会话记忆、上下文与监控 |
| `rl/` | 原始教师轨迹、SFT、RL、评分和评测 |
| `benchmarks/`、`experiments/` | 数据构造、离线分析和应用实验 |
| `skills/`、`config/` | 业务技能与部署配置 |
| `tests/` | 轻量替身测试及本地材料回归 |

应用默认使用工具式检索：模型主动调用知识工具，`/chat`不预先按意图检索。Profile、独立升级、Composer及Trace默认关闭，工具调用默认启用。Trace查询须同时开启采集与API开关；具体变量以环境模板为准。

接口交互见`http://127.0.0.1:8000/docs`。独立策略服务的模型、adapter、索引和工作根目录见[策略架构与运行](docs/agentic_rl_architecture.md)。

## 检查与阅读

仓库根目录运行`python -m pytest -q`。基础测试使用轻量替身，部分评分/采集回归需要本地保留材料；本地检查不代表真实服务、GPU或模型质量验收。

- [训练与评测](rl/README.md)
- [数据与离线分析](benchmarks/README.md)
- [检索与路由实验](experiments/README.md)
- [完整使用指南](wiki/完整使用指南.md)
- [业务流程](wiki/业务流程说明.md)

真实凭据、模型权重、索引和运行产物单独保存。真实模型调用、检索和训练可能产生费用，按项目授权规则执行。
