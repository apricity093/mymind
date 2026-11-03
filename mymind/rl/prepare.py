"""导入附录 A，核对实际主索引切块和私有条款支持。"""
import asyncio
from pathlib import Path
import shutil

from agents.customer_policy.configuration import rerank_gateway
from agents.customer_policy.retrieval import build_retrieval
from rl.runtime import initialize, write_json


async def main():
    root, config, output, tasks, standards, clauses = initialize()
    gateway = rerank_gateway(config["retrieval"])
    retrieval = build_retrieval(root, config["retrieval"], gateway)
    knowledge_path = Path(__file__).resolve().parents[1] / "agents/customer_policy/resources/knowledge.md"
    text = knowledge_path.read_text(encoding="utf-8")
    technical = text.index("### 接口认证")
    documents = [{"source_id": "billing-policy", "title": "退款到账与扣款咨询", "content": text[:technical], "format": "md"},
                 {"source_id": "technical-policy", "title": "接口认证、授权与故障咨询", "content": text[technical:], "format": "md"}]
    if config.get("dataset_mode") in {"calibration", "formal"}:
        outcome = {"status": "success", "pending_main": retrieval.knowledge.stats()["pending_main"],
                   "reused_existing_index": True}
        if not retrieval.knowledge._main.count():
            raise RuntimeError("校准及正式数据要求已有完整实验知识索引")
    else:
        destination = root / config["retrieval"]["knowledge_dir"] / "knowledge.md"
        shutil.copy2(knowledge_path, destination)
        outcome = retrieval.knowledge.import_documents(documents)
    write_json(output / "knowledge_import.json", outcome)
    if outcome["status"] != "success" or outcome["pending_main"]:
        raise RuntimeError("Qwen 主索引未完整导入，详见 knowledge_import.json")
    actual = retrieval.knowledge._main.get(include=["documents", "metadatas"])
    write_json(output / "main_chunks.json", actual)
    supported = {}
    compact = lambda value: "".join(value.split())
    for clause_id, clause in clauses.items():
        texts = [content for content, metadata in zip(actual["documents"], actual["metadatas"])
                 if metadata["source_id"] == clause.source_id]
        supported[clause_id] = compact(clause.text) in compact("".join(texts))
    write_json(output / "clause_coverage.json", supported)
    if not all(supported.values()):
        raise RuntimeError("实际切块缺少条款正文")
    probes = []
    for query in config["probe_queries"]:
        observation, diagnostics = await retrieval.search(query, 1)
        probes.append({"query": query, "observation": observation.model_dump(), "diagnostics": diagnostics})
    write_json(output / "retrieval_probes.json", probes)
    if any(p["observation"]["status"]["state"] != "candidates" or
           p["observation"]["status"]["degraded"] for p in probes):
        raise RuntimeError("真实主索引检索未通过")
    write_json(output / "summary.json", {"tasks_validated": len(tasks), "clauses_covered": len(supported),
        "main_chunks": len(actual["ids"]), "main_retrieval_probes": len(probes)})
    retrieval.knowledge.close()
    retrieval.knowledge.embedding.close()
    await gateway.client.close()


if __name__ == "__main__":
    asyncio.run(main())
