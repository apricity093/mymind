"""按启动配置装配运行路径和固定检索参数。"""
import json
import os
from pathlib import Path


def load_configuration(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    root = Path(os.environ["MYMIND_WORK_ROOT"]).resolve()
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    os.environ["RAG_ONNX_PATH"] = str(root / "cache/minilm")
    os.environ["RAG_CHUNK_TOKENS"] = str(config["retrieval"]["chunk_tokens"])
    os.environ["RAG_EMBEDDING_QUERY_BUDGET_S"] = str(config["retrieval"]["embedding_budget_s"])
    os.environ["RAG_RERANK_MAX_CHARS"] = str(config["retrieval"]["rerank_chars"])
    os.environ["RAG_RERANK_MAX_TOKENS"] = str(config["retrieval"]["rerank_tokens"])
    return root, config, output


def rerank_gateway(config):
    from core.llm_gateway import DeepSeekGateway
    from openai import AsyncOpenAI
    gateway = DeepSeekGateway(api_key=os.environ["DEEPSEEK_API_KEY"],
                              model=config["rerank_model"], base_url=config["rerank_base_url"],
                              thinking=config["rerank_thinking"])
    # 服务重试和超时明确记录在工具配置中。
    gateway.client = AsyncOpenAI(api_key=gateway.api_key, base_url=gateway.base_url,
        timeout=config["rerank_timeout_s"], max_retries=config["rerank_retries"])
    return gateway
