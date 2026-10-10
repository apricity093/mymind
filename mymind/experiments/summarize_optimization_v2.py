"""汇总实际产物、家族配对区间与改对/改错；未运行组保持缺失。"""
import argparse
import json
from pathlib import Path
from collections import Counter
import statistics
import sqlite3

from experiments.optimization_v2 import read, write
from evaluation.retrieval_metrics import paired_bootstrap_delta
from core.intent_recognizer import IntentRecognizer, IntentCategory
from core.document_chunking import BudgetChunker
from experiments.retrieval_scoring import (
    RETRIEVAL_METRICS, evaluate_retrieval, rescore_retrieval_group,
)


def qwen_overlap_observations(output):
    cache=output/'encoding_cache.sqlite3'
    paths={overlap:output/'chunks'/f'qwen-structure-512-{overlap}.json' for overlap in [0,32,64]}
    if not cache.exists() or not all(path.exists() for path in paths.values()):
        return None
    db=sqlite3.connect(cache)
    def count(text):
        if not text:
            return 0
        row=db.execute('SELECT tokens FROM counts WHERE text=?',(text,)).fetchone()
        if row is None:
            raise ValueError('overlap_real_count_missing')
        return row[0]
    result={'status':'measured','scope':'同章节相邻块；复用真实Qwen计数缓存，零新增API','groups':{}}
    try:
        for requested,path in paths.items():
            saved=read(path)
            chunker=BudgetChunker(count,512,requested,'qwen',reserve=16)
            rows=[]
            previous=None
            for item in saved['items']:
                if previous and (previous['source_id'],previous['section_path'])==(item['source_id'],item['section_path']):
                    tail=chunker._tail(previous['content']) if requested else ''
                    applied=bool(tail and item['content'].startswith(tail+'\n\n'))
                    rows.append({'chunk_id':item['chunk_id'],'requested':requested,'applied':applied,
                                 'actual_tokens':count(tail) if applied else 0,'actual_chars':len(tail) if applied else 0})
                previous=item
            values=[row['actual_tokens'] for row in rows]
            result['groups'][str(requested)]={'requested':requested,'pairs':len(rows),'applied_pairs':sum(r['applied'] for r in rows),
                'actual_distribution':dict(Counter(values)),'actual_min':min(values),'actual_max':max(values),'actual_mean':statistics.mean(values),
                'max_input_tokens_plus_reserve':max(saved['input_counts'])+16,'rows':rows}
    finally:
        db.close()
    write(output/'overlap_observations.json',result)
    return result


def summarize(output):
    data=read(Path(__file__).resolve().parents[1]/"data/eval/rag_optimization_v2.json")
    questions={q["id"]:q for q in data["questions"]}
    groups={}
    for path in (output/"retrieval").glob("*.json"):
        groups[path.stem]=rescore_retrieval_group(read(path),questions)
        write(path,groups[path.stem])
    for name in ["production.json","backup_production.json"]:
        value=read(output/name)
        if value and value.get("rows"):
            write(output/name,rescore_retrieval_group(value,questions))
    by_tag={}
    for name,value in groups.items():
        tags={tag for row in value["rows"] for tag in row["tags"] if row["answerable"]}
        by_tag[name]={tag:{"n":len(rows),**{metric:statistics.mean(r[metric] for r in rows) for metric in RETRIEVAL_METRICS}}
                      for tag in tags if (rows:=[row for row in value["rows"] if row["answerable"] and tag in row["tags"]])}
    selection=read(output/"selection.json",{})
    encoder='qwen' if read(output/'manifest.json',{}).get('models',{}).get('embedding')=='qwen3.7-text-embedding-flash' else 'gemini'
    comparisons={}
    primary=f"{encoder}-structure-512-0-test-0.5"
    candidate=f"{encoder}-structure-{selection.get('main_budget',512)}-{selection.get('main_overlap',0)}-test-{selection.get('main_weight',.5)}"
    backup="minilm-structure-240-0-test-0.5"
    backup_candidate=f"minilm-structure-240-0-test-{selection.get('backup_weight',.5)}"
    for name,a,b in [("main_candidate_minus_current",candidate,primary),("backup_candidate_minus_current",backup_candidate,backup),
                      ("backup_bm25_minus_hybrid","minilm-structure-240-0-test-bm25",backup),
                      ("main_bm25_minus_hybrid",f"{encoder}-structure-{selection.get('main_budget',512)}-{selection.get('main_overlap',0)}-test-bm25",candidate),
                      ("exact_tokens_minus_byte_baseline",candidate,'qwen_byte-structure-512-0-test-0.5')]:
        if a in groups and b in groups:
            ar={r["id"]:r for r in groups[a]["rows"]}
            br={r["id"]:r for r in groups[b]["rows"]}
            rows=[ar[qid] for qid in ar]
            comparisons[name]={"a":a,"b":b,"metrics":{},"improved":[],"regressed":[]}
            for metric in RETRIEVAL_METRICS:
                comparisons[name]["metrics"][metric]=paired_bootstrap_delta([{**r,"partition":r["family_id"]} for r in rows],
                    [float(ar[r["id"]][metric]) for r in rows],[float(br[r["id"]][metric]) for r in rows],samples=1000,seed=20261009)
            for r in rows:
                delta=ar[r["id"]]["recall_at_5"]-br[r["id"]]["recall_at_5"]
                if delta:
                    comparisons[name]["improved" if delta>0 else "regressed"].append({"id":r["id"],"family_id":r["family_id"],"delta":delta,
                         "candidate":ar[r["id"]]["retrieved"][:5],"current":br[r["id"]]["retrieved"][:5]})
    # 保存相关来源首次命中排名的变化。
    diagnostics=[]
    for split in ["dev","test"]:
        bm=groups.get(f"minilm-structure-240-0-{split}-bm25")
        fusion=groups.get(f"minilm-structure-240-0-{split}-0.5")
        vectors=groups.get(f"minilm-structure-240-0-{split}-vector")
        if not bm or not fusion or not vectors:
            continue
        fr={r["id"]:r for r in fusion["rows"]}
        vr={r["id"]:r for r in vectors["rows"]}
        for row in bm["rows"]:
            if row["answerable"] and row["mrr_at_10"]>fr[row["id"]]["mrr_at_10"]:
                evidence_ids=[item["chunk_id"] for item in row["retrieved"][:5]]
                diagnostics.append({"id":row["id"],"split":split,"bm25_first_relevant_rank":row["first_relevant_rank"],"fusion_first_relevant_rank":fr[row["id"]]["first_relevant_rank"],
                    "bm25":row["retrieved"],"vector":vr[row["id"]]["retrieved"],"fusion":fr[row["id"]]["retrieved"],
                    "bm25_top5_positions_in_fusion":{cid:next((i+1 for i,item in enumerate(fr[row["id"]]["retrieved"]) if item["chunk_id"]==cid),None) for cid in evidence_ids}})
    intent=read(output/"intent_test.json")
    intent_comparison={}
    if intent:
        levels=intent["thresholds"][str(selection["intent_threshold"])]
        two={r["id"]:r for r in levels["two"]["rows"]}
        three={r["id"]:r for r in levels["three"]["rows"]}
        improved=[qid for qid in two if two[qid]["predicted"]!=two[qid]["expected"] and three[qid]["predicted"]==three[qid]["expected"]]
        regressed=[qid for qid in two if two[qid]["predicted"]==two[qid]["expected"] and three[qid]["predicted"]!=three[qid]["expected"]]
        disagreements=[qid for qid in two if two[qid]["predicted"]!=three[qid]["predicted"]]
        intent_comparison={"improved":improved,"regressed":regressed,"disagreements":disagreements,"two":{k:v for k,v in levels["two"].items() if k!="rows"},
                           "three":{k:v for k,v in levels["three"].items() if k!="rows"}}
    elif (intent:=read(output/"intent_llm_test.json")):
        level=intent["thresholds"]["0.5"]
        intent_comparison={"two":{k:v for k,v in level["two"].items() if k!="rows"},"three":level["three"],"improved":None,"regressed":None,"disagreements":None}
    answers={p.stem:read(p) for p in (output/"answers").glob("*.json")}
    answer_summary={}
    answer_failures={}
    for name,value in answers.items():
        group=name[:name.rfind("-rag-v2")]
        summary=answer_summary.setdefault(group,Counter())
        summary["n"]+=1
        summary[value.get("status","unfinished")]+=1
        summary[f"rerank_{value.get('rerank_status','missing')}"]+=1
        if value.get("status")=="measured":
            audit=value["audit"]
            summary["facts_supported"]+=int(audit.get("facts_supported") is True)
            summary["conditions_complete"]+=int(audit.get("conditions_complete") is True)
            summary["unsupported_claims_n"]+=len(audit.get("unsupported_claims",[]))
            summary["citation_errors_n"]+=len(audit.get("citation_errors",[]))
            summary["citation_error_answers_n"]+=int(bool(audit.get("citation_errors")))
            summary["no_unsupported_claim_answers"]+=int(not audit.get("unsupported_claims"))
            passed=(audit["facts_supported"] and audit["conditions_complete"]
                    and not audit["unsupported_claims"] and not audit["citation_errors"]
                    and (value["question"]["answerable"] or not audit["unanswerable_fabricated"]))
            summary["all_checks_passed"]+=int(passed)
            if value["question"]["answerable"]:
                summary["answerable_n"]+=1
                summary["answerable_facts_supported"]+=int(audit["facts_supported"])
                summary["answerable_conditions_complete"]+=int(audit["conditions_complete"])
            if not passed:
                answer_failures.setdefault(group,[]).append({"id":value["question"]["id"],"audit":audit})
            if not value["question"]["answerable"]:
                summary["unanswerable_n"]+=1
                summary["unanswerable_fabricated"]+=int(audit.get("unanswerable_fabricated") is True)
    for summary in answer_summary.values():
        measured=summary["measured"]
        summary["rates"]={key:summary[key]/measured if measured else None for key in (
            "facts_supported","conditions_complete","all_checks_passed",
            "no_unsupported_claim_answers","citation_error_answers_n",
        )}
        summary["answerable_rates"]={key:summary[key]/summary["answerable_n"] if summary["answerable_n"] else None
            for key in ("answerable_facts_supported","answerable_conditions_complete")}
        summary["unanswerable_fabrication_rate"]=(summary["unanswerable_fabricated"]/summary["unanswerable_n"]
            if summary["unanswerable_n"] else None)
    raw_intent=read(output/"intent_raw_test.json",{}) or read(output/"intent_llm_raw_test.json",{})
    production=read(output/"production.json",{}) or read(output/"backup_production.json",{})
    production_group=candidate if production.get('rows',[{}])[0].get('route')=='main' else backup_candidate
    production_comparison={}
    if production.get("status")=="measured" and production_group in groups:
        offline={r["id"]:evaluate_retrieval(questions[r["id"]],r["retrieved"][:5]) for r in groups[production_group]["rows"]}
        rows=production["rows"]
        production_comparison={"n":len(rows),"ids":[r["id"] for r in rows],"offline":{},"production":{},
            "metric_changed_ids":[r["id"] for r in rows if any(r[metric]!=offline[r["id"]][metric] for metric in RETRIEVAL_METRICS)],
            "returned_top_k":5,
            "p50_ms":statistics.median(r["wall_ms"] for r in rows),
            "p95_ms":sorted(r["wall_ms"] for r in rows)[max(0,__import__('math').ceil(len(rows)*.95)-1)]}
        for metric in RETRIEVAL_METRICS:
            production_comparison["offline"][metric]=statistics.mean(offline[r["id"]][metric] for r in rows)
            production_comparison["production"][metric]=statistics.mean(r[metric] for r in rows)
    timings={}
    for label,key in [("llm_and_pattern","llm_ms"),("parallel_three","parallel_ms")]:
        values=[r[key] for r in raw_intent.values() if r["status"]=="measured" and "intent" in r["case"] and r.get(key) is not None]
        if values:
            timings[label]={"n":len(values),"p50_ms":statistics.median(values),"p95_ms":sorted(values)[max(0,__import__('math').ceil(len(values)*.95)-1)],
                            "scope":"同题真实LLM分支；复用结果时该耗时为原运行实测，不包含本轮模板准备" if label=="llm_and_pattern" else "真实LLM/embedding并行耗时，计入限速等待；不同于缓存复用排序"}
    summary={"manifest":read(output/"manifest.json"),"selection":selection,"api_usage":read(output/"api_usage.json"),
        "retrieval":{name:{k:v for k,v in value.items() if k!="rows"} for name,value in groups.items()},
        "retrieval_by_tag":by_tag,
        "comparisons":comparisons,"backup_regressions":diagnostics,"intent_comparison":intent_comparison,
        "intent_latency":read(output/"intent_latency.json"),"intent_branch_timings":timings,"answer_summary":answer_summary,
        "answer_failures":answer_failures,
        "answer_judge_scope":"同一deepseek-flash生成并审核；原始布尔/列表审核，无人工审核，不折算成未测的0-1评分",
        "production":read(output/"production.json"),"backup_production":read(output/"backup_production.json"),
        "production_comparison":production_comparison,"production_cache":read(output/"production_cache.json"),
        "endpoint_observations":read(output/"endpoint_observations.json")}
    summary['application']=read(output/'application.json')
    summary['overlap_observations']=qwen_overlap_observations(output) if encoder=='qwen' else None
    if intent and intent.get('diagnostics') and encoder=='qwen':
        voter=IntentRecognizer.__new__(IntentRecognizer)
        voter.threshold=selection.get('intent_threshold',.5)
        results=[]
        for value in intent['diagnostics']:
            votes={key:{**value[key],'intent':IntentCategory(value[key]['intent'])} for key in ['llm','embedding','pattern']}
            row={'id':value['case']['id'],'message':value['case']['message'],'allowed':value['case']['allowed_intents']}
            for mode in ['two','three']:
                voter._embedding_enabled=mode=='three'
                predicted,confidence,scores=voter._vote(votes['llm'],votes['embedding'],votes['pattern'])
                row[mode]={'predicted':predicted.value,'allowed_match':predicted.value in row['allowed'],'confidence':confidence,'scores':scores}
            results.append(row)
        summary['intent_diagnostics']={'scope':'20条歧义/弱监督诊断，单独报告，不并入152题准确率','rows':results,
            'two_allowed_matches':sum(r['two']['allowed_match'] for r in results),'three_allowed_matches':sum(r['three']['allowed_match'] for r in results)}
    write(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    summarize(args.output)


if __name__=="__main__":
    main()
