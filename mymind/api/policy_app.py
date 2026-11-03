"""独立客服策略接口，启动时复用模型和检索，每次请求创建轨迹。"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
import asyncio
import json
import os
import sys
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import Field

from agents.customer_policy.configuration import load_configuration, rerank_gateway
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import QwenGeneration
from agents.customer_policy.protocol import ProtocolModel, PublicTask, VisibleMaterial
from agents.customer_policy.retrieval import build_retrieval
from agents.customer_policy.runner import run_task


class PolicyRequest(ProtocolModel):
    role: Literal["billing", "technical"]
    user_text: str = Field(min_length=1)
    records: list[VisibleMaterial] = Field(default_factory=list)

    def public_task(self, request_id):
        return PublicTask(task_id=request_id, **self.model_dump())


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class ServiceGeneration:
    """生成在工作线程执行，检索继续复用服务事件循环。"""

    def __init__(self, generation):
        self.generation = generation

    def token_count(self, messages):
        return self.generation.token_count(messages)

    def _generate(self, messages, tools_enabled, options):
        return asyncio.run(self.generation.generate(messages, tools_enabled=tools_enabled, **options))

    async def generate(self, messages, tools_enabled=True, **options):
        pending = asyncio.create_task(asyncio.to_thread(self._generate, messages, tools_enabled, options))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            # 取消 HTTP 等待不能提前释放 GPU 锁；线程中的生成无法取消。
            await pending
            raise


@asynccontextmanager
async def lifespan(app):
    config_path = Path(os.environ["MYMIND_AGENT_CONFIG"])
    if not config_path.is_absolute():
        raise ValueError("MYMIND_AGENT_CONFIG 须为绝对路径")
    root, config, output = load_configuration(config_path)
    if config["retrieval"]["rerank_model"] != "deepseek-flash":
        raise ValueError("固定重排须为 deepseek-flash")
    gateway = rerank_gateway(config["retrieval"])
    retrieval = build_retrieval(root, config["retrieval"], gateway)
    generation = QwenGeneration(root / config["model_dir"],
        {**config["generation"], "model_id": config["model_id"], "model_revision": config["model_revision"]},
        adapter_path=root / config["adapter_dir"])
    app.state.generation, app.state.retrieval = generation, retrieval
    app.state.output, app.state.config = output, config
    app.state.lock = asyncio.Lock()
    app.state.requests_completed = 0
    app.state.started_at = datetime.now(timezone.utc).isoformat()
    write_json(output / "configuration.json", config)
    write_json(output / "model_load.json", generation.load_info)
    write_json(output / "startup.json", {"pid": os.getpid(), "argv": sys.argv,
        "python": sys.executable, "cwd": str(Path.cwd()), "work_root": str(root),
        "config_path": str(config_path), "started_at": app.state.started_at})
    (output / "requests").mkdir(exist_ok=True)
    try:
        yield
    finally:
        await gateway.client.close()
        retrieval.knowledge.embedding.close()
        retrieval.knowledge.close()
        write_json(output / "shutdown.json", {"pid": os.getpid(),
            "stopped_at": datetime.now(timezone.utc).isoformat(),
            "requests_completed": app.state.requests_completed})


app = FastAPI(
    title="MyMind Customer Policy 推理引擎",
    description="给定账单或技术角色，执行 SFT/RL 共享策略的检索与终止决策。",
    lifespan=lifespan,
)


@app.get("/status")
async def status():
    return {"ready": True, "pid": os.getpid(), "started_at": app.state.started_at,
        "requests_completed": app.state.requests_completed,
        "model": app.state.generation.load_info, "retrieval": app.state.config["retrieval"],
        "generation": app.state.config["generation"], "budget": app.state.config["budget"]}


@app.post("/requests")
async def handle_request(request: PolicyRequest):
    request_id = str(uuid4())
    try:
        task = request.public_task(request_id)
    except ValueError as ex:
        raise HTTPException(422, detail=str(ex)) from ex
    # 单个 GPU 的完整轨迹串行执行；锁内仍为每个请求创建新环境。
    async with app.state.lock:
        row = await run_task(task, ServiceGeneration(app.state.generation), PolicyEnvironment(app.state.retrieval),
                             budget=app.state.config["budget"])
        row.update(request_id=request_id, request=request.model_dump(),
                   completed_at=datetime.now(timezone.utc).isoformat())
        write_json(app.state.output / "requests" / f"{request_id}.json", row)
        app.state.requests_completed += 1
    return row


@app.get("/requests/{request_id}")
async def read_request(request_id: str):
    from uuid import UUID
    try:
        request_id = str(UUID(request_id))
    except ValueError:
        raise HTTPException(404, detail="请求不存在") from None
    path = app.state.output / "requests" / f"{request_id}.json"
    if not path.exists():
        raise HTTPException(404, detail="请求不存在")
    return json.loads(path.read_text(encoding="utf-8"))
