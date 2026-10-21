"""当前源码、固定环境、tokenizer 与完整索引的 CPU 接入检查。"""
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "mymind"))
from agents.customer_policy.configuration import rerank_gateway, load_configuration
from agents.customer_policy.retrieval import build_retrieval
from experiments.agentic_rl.sft_data import encode_trajectory
from experiments.agentic_rl.runtime import write_json
from experiments.agentic_rl.stage7 import read, check_inputs
from experiments.agentic_rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs


def main():
    import asyncio
    from transformers import AutoTokenizer
    from trl import GRPOConfig
    from experiments.agentic_rl.grpo import make_trainer_class
    root = Path(os.environ["MYMIND_WORK_ROOT"])
    run = root / "runs/stage7-preparation-20261014"
    baseline = read(run / "baseline.json")
    _,cfg,_ = load_configuration(run / "configs/preflight-sft.json")
    tasks = load_public_tasks(root / cfg["tasks"])
    standards = load_private_standards(root / cfg["standards"])
    clauses = load_policy_support(root / cfg["policy_support"])
    validate_task_pairs(tasks, standards, clauses)
    check_inputs(root,cfg,tasks,standards,"sft")
    provenance=read(root / cfg["model_dir"] / "base_revision.json")
    assert provenance["revision"]==cfg["model_revision"] and provenance["repo_id"]==cfg["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(root / cfg["model_dir"],local_files_only=True)
    assert tokenizer.chat_template==(root / baseline["tokenizer_template"]).read_text(encoding="utf-8")
    rows=read(root / cfg["student_data"])
    for row in rows:
        encoded=encode_trajectory(tokenizer,row["messages"])
        assert all(encoded[k]==row[k] for k in ("input_ids","labels","attention_mask","spans"))
    rl=read(run / "configs/preflight-grpo.json")
    options=dict(rl["grpo"]); options.pop("steps_per_generation")
    training={**rl["training"],"per_device_train_batch_size":2,"use_cpu":True}
    args=GRPOConfig(output_dir=str(run / "cpu-config"),**training,**options)
    assert args.generation_batch_size==4 and args.steps_per_generation==2
    cls=make_trainer_class()
    gateway=rerank_gateway(cfg["retrieval"])
    retrieval=build_retrieval(root,cfg["retrieval"],gateway)
    try:
        stats=retrieval.knowledge.stats()
        assert not stats["pending_main"] and retrieval.knowledge._main.count()>0
        assert (root / cfg["retrieval"]["knowledge_dir"] / "knowledge.md").read_bytes()==(SOURCE / "mymind/agents/customer_policy/resources/knowledge.md").read_bytes()
        actual=retrieval.knowledge._main.get(include=["documents","metadatas"])
        compact=lambda text:"".join(text.split())
        coverage={cid:compact(clause.text) in compact("".join(text for text,meta in zip(actual["documents"],actual["metadatas"]) if meta["source_id"]==clause.source_id)) for cid,clause in clauses.items()}
        assert all(coverage.values())
        write_json(run / "knowledge-index.json",actual)
        result={"versions":{n:version(n) for n in ("torch","transformers","peft","trl","accelerate","chromadb")},
            "base":provenance,"tokenizer":type(tokenizer).__name__,"template_matches":True,
            "student_rows_reencoded":len(rows),"private_standards":len(standards),
            "validation_tasks":sum(s.split=="validation" for s in standards.values()),
            "test_tasks":sum(s.split=="test" for s in standards.values()),
            "index":stats,"embedding_model":retrieval.knowledge.embedding.settings.model,
            "rerank_config":cfg["retrieval"],"trainer_class":cls.__name__,"model_calls":0,"api_requests":0}
        result.update(knowledge_source_equals_index_input=True,clause_coverage=coverage)
        write_json(run / "remote-cpu.json",result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:
        retrieval.knowledge.close(); retrieval.knowledge.embedding.close()
        asyncio.run(gateway.client.close())


if __name__=="__main__":
    main()
