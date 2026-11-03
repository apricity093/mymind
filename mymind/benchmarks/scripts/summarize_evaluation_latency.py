"""离线汇总保留的正式任务、模板加载与真实检索耗时。"""
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
STAGE8 = Path("agentic_runs/stage8-20261016")
QWEN = Path("mymind/artifacts/qwen-optimization-v2/20261010-qwen-01")
OUTPUT = ROOT / "agentic_runs/analysis-20261017-latency/metrics.json"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def timing(value):
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(value) and value >= 0, value
    return value


def summarize(values, unit):
    values = sorted(timing(value) for value in values)
    assert values
    return {"count": len(values), "unit": unit, "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": values[math.ceil(len(values) * 0.95) - 1], "sum": sum(values)}


def formal_tests():
    costs_source = STAGE8 / "COSTS.md"
    costs = {}
    for line in (ROOT / costs_source).read_text(encoding="utf-8").splitlines():
        if line.startswith("| "):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            costs[cells[0]] = cells
    groups, task_ids = {}, None
    for name, label in (("base", "Base三搜/全部"), ("sft", "SFT120三搜/全部"),
                        ("rl", "RL40三搜/全部")):
        source = STAGE8 / "test" / name / "first_records.jsonl"
        rows = [json.loads(line) for line in (ROOT / source).read_text(
            encoding="utf-8").splitlines() if line.strip()]
        ids = {row["task_id"] for row in rows}
        assert len(rows) == len(ids) == 120, name
        if task_ids is None:
            task_ids = ids
        assert ids == task_ids, name
        elapsed = summarize([row["elapsed_s"] for row in rows], "seconds")
        generation = []
        for row in rows:
            assert row["generations"], row["task_id"]
            seconds = sum(timing(item["latency_ms"]) / 1000 for item in row["generations"])
            assert math.isclose(seconds, timing(row["cost"]["gpu_generation_s"]),
                                rel_tol=1e-12, abs_tol=1e-9), row["task_id"]
            generation.append(seconds)
        generated = summarize(generation, "seconds")
        cells = costs[label]
        assert int(cells[1]) == len(rows), name
        assert f"{elapsed['sum']:.3f}" == cells[5], name
        assert f"{generated['sum']:.3f}" == cells[6], name
        assert 0 < generated["sum"] <= elapsed["sum"], name
        groups[name] = {"source": source.as_posix(), "elapsed_s": elapsed,
                        "generation_s": generated,
                        "generation_fraction_of_elapsed": generated["sum"] / elapsed["sum"],
                        "costs_reported_s": {"elapsed": float(cells[5]),
                                             "generation": float(cells[6])}}
    return {
        "costs_source": costs_source.as_posix(),
        "timing_code": "mymind/rl/formal_evaluate.py",
        "generation_timing_code": "mymind/agents/customer_policy/generation.py",
        "scope": "每组全部120条首次任务，包含答错与协议失败。模型加载后逐题计时，含预算预留、生成、真实检索、协议处理与任务内进度落盘；不含模型加载、最终评分和最终记录写盘。属于本批串行评测任务墙钟，非HTTP端到端或服务压测。",
        "generation_scope": "逐题累加generations[*].latency_ms，包含模型生成及输出转移、解码、解析等保存计时范围；不是GPU kernel时间或GPU利用率。共享GPU负载与输出长度会影响组间耗时，不能仅凭本表归因于训练算法。",
        "groups": groups,
    }


def template_cache():
    source = QWEN / "intent_dev.json"
    startup = read(source)["startup"]
    cold = timing(startup["template_prepare_wall_ms"])
    warm = timing(startup["cache_hit_startup_ms"])
    assert cold > 0
    assert startup["template_encoded_texts"] == 54
    assert startup["cache_hit_template_encoded_texts"] == 0
    return {
        "source": source.as_posix(), "source_field": "startup",
        "timing_code": "mymind/experiments/qwen_optimization.py:Runner.intent",
        "observations_per_condition": 1,
        "cold_template_prepare_ms": cold,
        "cache_hit_startup_ms": warm,
        "cold_template_encoded_texts": startup["template_encoded_texts"],
        "cache_hit_template_encoded_texts": startup["cache_hit_template_encoded_texts"],
        "observed_reduction_fraction": 1 - warm / cold,
        "scope": "同一路模板加载的单次冷准备与重启缓存读取观测，54条真实Qwen模板编码对比0条重复编码。仅为模板初始化组件，含实验环境的请求等待；不是完整recognize、聊天延迟或稳定生产降幅。",
    }


def retrieval():
    source = QWEN / "production.json"
    production = read(source)
    rows = production["rows"]
    assert production["status"] == "measured"
    assert len(rows) == len({row["id"] for row in rows}) == 20
    assert all(row["route"] == "main" and row["cached"] is False for row in rows)
    return {
        "source": source.as_posix(), "source_field": "rows[*].wall_ms",
        "timing_code": "mymind/experiments/qwen_optimization.py:Runner.production",
        "wall_ms": summarize([row["wall_ms"] for row in rows], "milliseconds"),
        "result_cache_hits": 0, "index_route": "main",
        "scope": "20道固定可答题，已准备索引的真实Qwen查询编码、混合检索与Persistent Chroma工具路径；全部结果缓存未命中。计时含工具调用与本地检索指标计算，不含文档建库、初始批量节流等待、LLM重排、查询改写及最终回答。是暖组件查询观测，不是完整RAG或HTTP服务延迟。",
    }


def main():
    result = {
        "method": {"p50": "statistics.median：偶数样本取中间两项平均。",
                   "p95": "nearest-rank：升序第ceil(0.95*n)项，索引从1开始。",
                   "sources": "路径相对仓库根目录；仅读取已有记录，不生成轨迹或重新评分。"},
        "formal_test": formal_tests(),
        "template_cache": template_cache(),
        "persistent_chroma": retrieval(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
