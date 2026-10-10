"""固定阶段的第二轮实验；真实模型、持久复用、预算与失败断点。"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import random
import re
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import chromadb
import numpy as np
import httpx
from types import SimpleNamespace
from dotenv import load_dotenv

from core.document_chunking import BudgetChunker, parse_sections
from core.embedding import EmbeddingError, EmbeddingSettings, GeminiEmbedding
from core.intent_recognizer import IntentRecognizer, IntentCategory
from core.llm_gateway import LLMRequest
from core.retrieval import BM25Index, weighted_rrf, dedupe_items, RRF_K
from experiments.gemini_rag import chat_gateway, normalize, duplicate_fraction, macro_f1
from experiments.retrieval_scoring import evaluate_retrieval, summarize_retrieval, rescore_retrieval_group
from mcp.indexed_knowledge_base import KnowledgeBase, prepare_minilm
from mcp.tool_manager import MCPToolManager, Tool

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["validate", "smoke", "chunking", "overlap", "fusion", "intent_dev", "select", "test", "intent_test", "answers", "production"]
LOCAL_STAGES = ["backup_dev", "intent_llm_dev", "local_select", "local_test", "intent_llm_test", "backup_answers", "backup_production", "production_cache"]


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class BudgetEmbedding(GeminiEmbedding):
    def __init__(self, settings, output, interval):
        super().__init__(settings, min_interval=interval)
        self.output = output
        self.ledger = read(output / "api_usage.json", {"google_http": 0, "google_encoded_texts": 0, "google_count_requests": 0,
                            "google_reported_tokens": 0, "llm_calls": 0, "throttle_sleep_ms": 0, "retry_wait_ms": 0})
        self.counts = read(output / "token_counts.json", {})
        self.saved_vectors = read(output / "vectors.json", {})
        self.reuse = Counter(read(output/"cache_reuse.json",{}))
        self.blocked = self.ledger.get("google_blocked_reason")
        if not self.blocked:
            failures=read(output/"api_failures.json",[])
            self.blocked=next((f["reason"] for f in reversed(failures) if "PerDay" in f["reason"]),None)
            if self.blocked:
                self.ledger["google_blocked_reason"]=self.blocked

    def flush(self):
        self.ledger["throttle_sleep_ms"] = self.base_wait + self.usage["throttle_sleep_ms"]
        write(self.output / "api_usage.json", self.ledger)
        write(self.output / "token_counts.json", self.counts)
        write(self.output / "vectors.json", self.saved_vectors)
        write(self.output / "cache_reuse.json",dict(self.reuse))

    @property
    def base_wait(self):
        return getattr(self, "_base_wait", 0)

    def _request(self, method, suffix, payload=None, units=1):
        if self.blocked:
            raise EmbeddingError(self.blocked)
        for attempt in range(3):
            texts = units if "embed" in suffix.lower() else 0
            if self.ledger["google_http"] >= 20000 or self.ledger["google_encoded_texts"] + texts > 10000:
                self.blocked = "authorized_budget_exhausted"
                raise EmbeddingError(self.blocked)
            self.ledger["google_http"] += 1
            self.ledger["google_encoded_texts"] += texts
            self.ledger["google_count_requests"] += int(suffix == ":countTokens")
            write(self.output / "api_usage.json", self.ledger)
            try:
                result = super()._request(method, suffix, payload, units)
                self.ledger["google_reported_tokens"] += result.get("usageMetadata", {}).get("promptTokenCount", 0)
                return result
            except EmbeddingError as ex:
                reason = str(ex)
                failures = read(self.output / "api_failures.json", [])
                failures.append({"operation": suffix or "model_metadata", "reason": reason, "attempt": attempt+1})
                write(self.output / "api_failures.json", failures)
                match = re.search(r"retry=([\d.]+)s", reason)
                if "HTTP_429" in reason and match and attempt < 2 and "PerDay" not in reason:
                    delay = min(float(match.group(1))+1, 60)
                    time.sleep(delay)
                    self.ledger["retry_wait_ms"] += delay*1000
                    continue
                if any(code in reason for code in ["HTTP_401", "HTTP_403", "HTTP_429"]):
                    self.blocked = reason
                    self.ledger["google_blocked_reason"] = reason
                    write(self.output / "api_usage.json", self.ledger)
                raise

    def count_tokens(self, text):
        key = json.dumps([self.identity, text], ensure_ascii=False)
        if key in self.counts:
            self.reuse["count"] += 1
            self._count_supported = self.counts[key][1] == "google_countTokens"
            return self.counts[key][0]
        value = super().count_tokens(text)
        self.counts[key] = [value, self.counting_mode]
        return value

    def cached_embed(self, texts, task, titles=None):
        keys = [json.dumps([self.identity, task, (titles[i] if titles else ""), text], ensure_ascii=False) for i, text in enumerate(texts)]
        pending = list(dict.fromkeys(key for key in keys if key not in self.saved_vectors))
        self.reuse[task] += len(keys)-len(pending)
        for offset in range(0, len(pending), 16):
            batch = pending[offset:offset+16]
            inputs = [json.loads(key) for key in batch]
            values = super().embed([x[3] for x in inputs], task, [x[2] for x in inputs])
            self.saved_vectors.update(zip(batch, values))
            self.flush()
        return [self.saved_vectors[key] for key in keys]


class Runner:
    def __init__(self, args):
        self.args, self.output = args, args.output.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        load_dotenv(ROOT / ".env")
        settings = EmbeddingSettings.from_env()
        if settings.dimensions != 768:
            raise ValueError("本轮固定 768 维")
        self.google = BudgetEmbedding(settings, self.output, args.api_interval)
        if getattr(args,"retry_google",False):
            self.google.blocked=None
            self.google.ledger.pop("google_blocked_reason",None)
            write(self.output/"api_failures_before_retry.json",read(self.output/"api_failures.json",[]))
        self.google._base_wait = self.google.ledger["throttle_sleep_ms"]
        self.rag = read(args.rag_data)
        self.intent_data = read(args.intent_data)
        self.corpus = self.rag["corpus"]
        self.by_source = {d["source_id"]: d for d in self.corpus}
        self.mini = self.mini_count = None
        self.mini_vectors = read(self.output / "mini_vectors.json", {})
        self.gateway, self.model = chat_gateway()
        if self.gateway:
            original = self.gateway.complete
            async def measured(request):
                if self.google.ledger["llm_calls"] >= 600:
                    raise RuntimeError("authorized_llm_budget_exhausted")
                self.google.ledger["llm_calls"] += 1
                write(self.output / "api_usage.json", self.google.ledger)
                try:
                    result = await original(request)
                except Exception as ex:
                    failures=read(self.output/"llm_failures.json",[])
                    failures.append({"model":request.model,"error_type":type(ex).__name__,"http_status":getattr(ex,"status_code",None)})
                    write(self.output/"llm_failures.json",failures)
                    raise
                saved = read(self.output / "llm_usage.json", [])
                saved.append({**self.gateway.usage[-1], "response_retry":result.metadata.get("response_retry"),
                              "empty_response_retry":result.metadata.get("empty_response_retry")})
                write(self.output / "llm_usage.json", saved)
                return result
            self.gateway.complete = measured
        self.manifest = read(self.output / "manifest.json", None)
        if self.manifest is None:
            self.manifest = {"date": "2026-10-09", "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "initial_git_status": subprocess.check_output(["git", "status", "--short"], text=True), "python": platform.python_version(),
                "environment": "learn_claude", "chroma": chromadb.__version__, "data_version": self.rag["version"],
                "rag_file": str(args.rag_data.resolve()), "intent_file": str(args.intent_data.resolve()),
                "source_ids": list(self.by_source), "rag_ids": [q["id"] for q in self.rag["questions"]],
                "intent_ids": [q["id"] for q in self.intent_data["cases"]], "answer_ids": self.rag["answer_ids"],
                "models": {"embedding": settings.model, "dimensions": 768, "backup": "all-MiniLM-L6-v2", "chat": self.model},
                "limits": {"google_texts": 10000, "google_http": 20000, "llm_calls": 600}, "rrf_k": RRF_K,
                "estimate": {"google_encoded_texts": "约 2500–4500，含六种分块、overlap、180查询、282分类文本及必要生产验证", "google_http": "约 6000–16000，实际计数与复用决定", "llm_calls": "最多 522，含248意图、90题组重排/回答/审核与改写验证"}, "stages": {}}
            write(self.output / "manifest.json", self.manifest)
        elif self.manifest["models"]["chat"]!=self.model:
            self.manifest.setdefault("chat_model_resolutions",[]).append({"previous":self.manifest["models"]["chat"],"effective":self.model,
                "reason":"服务实际支持的名称；进程配置覆盖，未改真实.env"})
            self.manifest["models"]["chat"]=self.model
            write(self.output/"manifest.json",self.manifest)

    def prepare_mini(self, warm=True):
        if self.mini is None:
            self.mini, self.mini_count = prepare_minilm(os.getenv("RAG_ONNX_PATH"), warm=warm)

    def chunks(self, encoder, kind="structure", budget=512, overlap=0):
        name = f"{encoder}-{kind}-{budget}-{overlap}"
        path = self.output / "chunks" / f"{name}.json"
        saved = read(path)
        if saved is not None:
            if saved["data_version"] != self.rag["version"] or saved["corpus"] != self.corpus:
                raise ValueError("saved_chunks_do_not_match_corpus")
            return saved["items"]
        if encoder == "minilm":
            self.prepare_mini(warm=False)
        counter = self.google.count_tokens if encoder == "gemini" else self.mini_count
        chunker = BudgetChunker(counter, budget, overlap, encoder, reserve=16 if encoder=="gemini" else 0)
        items = []
        for doc in self.corpus:
            items += [{**r.metadata, "content": r.text} for r in chunker.chunk_document(doc["title"], doc["content"], doc["source_id"], structural=kind=="structure")]
            print(f"chunks {name} source={doc['source_id']} total_chunks={len(items)}",flush=True)
        counts = [counter(BudgetChunker.embedding_title(i["title"], i["section_path"])+"\n"+i["content"]) for i in items]
        assert max(counts) + chunker.reserve <= budget
        assert len(items) >= 50
        write(path, {"data_version": self.rag["version"], "corpus": self.corpus, "items": items, "input_counts": counts,
                     "config": chunker.config, "counting_mode": self.google.counting_mode if encoder=="gemini" else "minilm_tokenizer"})
        self.google.flush()
        return items

    def mini_embed(self, texts):
        self.prepare_mini()
        pending = list(dict.fromkeys(text for text in texts if text not in self.mini_vectors))
        for offset in range(0, len(pending), 64):
            batch = pending[offset:offset+64]
            self.mini_vectors.update(zip(batch, [[float(v) for v in vec] for vec in self.mini(batch)]))
            write(self.output / "mini_vectors.json", self.mini_vectors)
        return [self.mini_vectors[text] for text in texts]

    def evaluate(self, question, ranked):
        result = evaluate_retrieval(question, ranked)
        result["context_chars"] = sum(len(i["content"]) for i in ranked[:5])
        result["sentence_duplicate_approximation"] = duplicate_fraction(ranked[:5])
        return result


    def retrieval(self, encoder, kind, budget, overlap, split, modes, limit=0):
        name=f"{encoder}-{kind}-{budget}-{overlap}-{split}"
        path=self.output / "rankings" / f"{name}.json"
        questions=[q for q in self.rag["questions"] if q["split"]==split][:limit or None]
        completed={mode:read(self.output/"retrieval"/f"{name}-{mode}.json") for mode in modes}
        if all(value and value.get("status")=="measured" and [r["id"] for r in value["rows"]]==[q["id"] for q in questions] for value in completed.values()):
            print(f"retrieval {name} reused_completed_groups={','.join(modes)}",flush=True)
            return {mode:value["rows"] for mode,value in completed.items()}
        items=self.chunks(encoder,kind,budget,overlap)
        cached=read(path, {"rows":{}})
        need_vector=any(mode!="bm25" for mode in modes)
        before=time.perf_counter()
        titles=[BudgetChunker.embedding_title(i["title"],i["section_path"]) for i in items]
        if need_vector:
            values=self.google.cached_embed([i["content"] for i in items],"RETRIEVAL_DOCUMENT",titles) if encoder=="gemini" else self.mini_embed([f"{t}\n{i['content']}" for t,i in zip(titles,items)])
            matrix=np.asarray(values,dtype=float)
            matrix/=np.linalg.norm(matrix,axis=1,keepdims=True)
        build_ms=(time.perf_counter()-before)*1000
        index=BM25Index()
        for item,title in zip(items,titles):
            index.add(item["chunk_id"], f"{title}\n{item['content']}",item)
        results={mode:[] for mode in modes}
        for n,q in enumerate(questions):
            stored=cached["rows"].get(q["id"], {})
            had_vector,had_bm25="vector" in stored,"bm25" in stored
            if need_vector and "vector" not in stored:
                started=time.perf_counter()
                query_reused=json.dumps([self.google.identity,"RETRIEVAL_QUERY","",q["question"]],ensure_ascii=False) in self.google.saved_vectors if encoder=="gemini" else q["question"] in self.mini_vectors
                qv=self.google.cached_embed([q["question"]],"RETRIEVAL_QUERY")[0] if encoder=="gemini" else self.mini_embed([q["question"]])[0]
                stored["encoding_wall_ms"]=(time.perf_counter()-started)*1000
                stored["query_vector_reused"]=query_reused
                scores=matrix @ (np.asarray(qv)/np.linalg.norm(qv))
                order=sorted(range(len(items)),key=lambda i:(-scores[i],items[i]["chunk_id"]))[:20]
                stored["vector"]=[{**items[i], "score":float(scores[i]),"retrieval_sources":["vector"]} for i in order]
            if any(mode!="vector" for mode in modes) and "bm25" not in stored:
                stored["bm25"]=[{**index.items[ident],"score":float(score),"retrieval_sources":["bm25"]} for ident,score in index.search(q["question"],20)]
            cached["rows"][q["id"]]=stored
            for mode in modes:
                fusion_started=time.perf_counter()
                ranked=stored[mode] if mode in {"vector","bm25"} else weighted_rrf(stored["vector"],stored["bm25"],vector_weight=float(mode),bm25_weight=1-float(mode))
                row=self.evaluate(q,dedupe_items(ranked))
                row["fusion_and_evaluation_wall_ms"]=(time.perf_counter()-fusion_started)*1000
                row["ranking_cache_reused"]=had_vector if mode=="vector" else had_bm25 if mode=="bm25" else had_vector and had_bm25
                results[mode].append(row)
            write(path,cached)
            if n % 10==0:
                print(f"retrieval {name} {n+1}/{len(questions)}",flush=True)
        for mode,rows in results.items():
            positive=[r for r in rows if r["answerable"]]
            summary=summarize_retrieval(rows)
            summary.update(status="measured",chunks=len(items),n=len(rows),answerable_n=len(positive),mode=mode,config=name,
                          vector_build_wall_ms=build_ms if need_vector else 0,rows=rows,
                          context_chars=statistics.mean(r["context_chars"] for r in rows),duplicate_approx=statistics.mean(r["sentence_duplicate_approximation"] for r in rows),
                          latency_scope="离线排名，编码缓存复用计时不能解释为在线延迟")
            write(self.output / "retrieval" / f"{name}-{mode}.json",summary)
        self.google.flush()
        return results

    def validate(self):
        questions=self.rag["questions"]
        assert len(self.corpus)==60 and len(questions)==180
        families={}
        for q in questions:
            assert families.setdefault(q["family_id"],q["split"])==q["split"]
            assert bool(q["evidence_groups"])==q["answerable"]
            for group in q["evidence_groups"]:
                for alt in group["alternatives"]:
                    assert self.by_source[alt["source_id"]]["content"][alt["start"]:alt["end"]]==alt["quote"]
            if not q["answerable"]:
                assert all(q["absent_topic"] not in d["content"] for d in self.corpus)
        assert Counter(q["split"] for q in questions)=={"dev":60,"test":120}
        cases=self.intent_data["cases"]
        assert set(self.intent_data["labels"])=={c.value for c in IntentCategory}
        assert Counter(q["split"] for q in cases)=={"dev":76,"test":152}
        for q in cases:
            assert families.setdefault(q["family_id"],q["split"])==q["split"]
        long=[]
        for doc in self.corpus[:16]:
            value=max(self.google.count_tokens(BudgetChunker.embedding_title(doc["title"],s.path)+"\n"+s.text) for s in parse_sections(doc["content"]))
            long.append({"source_id":doc["source_id"],"max_section_input_tokens":value})
        assert all(d["max_section_input_tokens"]>768 for d in long)
        assert any(d["max_section_input_tokens"]>1536 for d in long)
        result={"status":"measured","data_frozen":True,"long_sections":long,"families":90,"intent_families":114,
                "answer_ids":self.rag["answer_ids"],"review":"代理语义复核：版本/渠道明确限定，常规办理与故障例外分别处理；非人工审核。合成规则共享知识库问题泛化。"}
        write(self.output/"data_validation.json",result)
        self.google.flush()

    def choose_chunk(self):
        values=[read(self.output/"retrieval"/f"gemini-structure-{b}-0-dev-vector.json") for b in [256,512,768]]
        if any(v is None for v in values):
            raise RuntimeError("chunking_not_complete")
        best=max(values,key=lambda v:(v["recall_at_5"],v["precision_at_3"]))
        current=values[1]
        return int(best["config"].split("-")[2]) if best["recall_at_5"]>=current["recall_at_5"]+.02 and best["precision_at_3"]>=current["precision_at_3"] else 512

    def choose_overlap(self,budget):
        values=[read(self.output/"retrieval"/f"gemini-structure-{budget}-{o}-dev-vector.json") for o in [0,32,64]]
        if any(v is None for v in values):
            raise RuntimeError("overlap_not_complete")
        best=max(values,key=lambda v:(v["recall_at_5"],v["precision_at_3"],-int(v["config"].split("-")[3])))
        current=values[0]
        return int(best["config"].split("-")[3]) if best["recall_at_5"]>=current["recall_at_5"]+.02 and best["precision_at_3"]>=current["precision_at_3"] else 0

    async def intent(self,split):
        if self.gateway is None:
            raise RuntimeError("chat_credentials_missing")
        recognizer=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.google,
            template_cache_path=str(self.output/"intent_templates.json"),template_wait_s=180)
        cold_started=time.perf_counter()
        before=self.google.ledger["google_encoded_texts"]
        await recognizer._load_template_embeddings()
        template_time=(time.perf_counter()-cold_started)*1000
        template_texts=self.google.ledger["google_encoded_texts"]-before
        # 重启读取同一真实模板缓存，单独记录耗时和编码数。
        restarted=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.google,
            template_cache_path=str(self.output/"intent_templates.json"),template_wait_s=180)
        started=time.perf_counter()
        before=self.google.ledger["google_encoded_texts"]
        await restarted._load_template_embeddings()
        startup={"template_prepare_wall_ms":template_time,"template_encoded_texts":template_texts,
                 "cache_hit_startup_ms":(time.perf_counter()-started)*1000,"cache_hit_template_encoded_texts":self.google.ledger["google_encoded_texts"]-before}
        raw_path=self.output/f"intent_raw_{split}.json"
        raw=read(raw_path,{})
        earlier_llm=read(self.output/f"intent_llm_raw_{split}.json",{})
        cases=[c for c in self.intent_data["cases"] if c["split"]==split]
        if split=="test":
            cases+=self.intent_data["diagnostics"]
        try:
            for n,case in enumerate(cases):
                if case["id"] in raw and raw[case["id"]]["status"]=="measured":
                    continue
                if case["id"] in raw:
                    write(self.output/"failed_intent"/f"{case['id']}-{datetime.now().strftime('%H%M%S%f')}.json",raw[case["id"]])
                start=time.perf_counter()
                async def llm():
                    saved=earlier_llm.get(case["id"],{})
                    if saved.get("status")=="measured":
                        return {**saved["llm"],"intent":IntentCategory(saved["llm"]["intent"])},saved["llm_ms"]
                    t=time.perf_counter()
                    value=await recognizer._llm_recognize(case["message"],None)
                    return value,(time.perf_counter()-t)*1000
                async def emb():
                    t=time.perf_counter()
                    vector=(await asyncio.to_thread(self.google.cached_embed,[case["message"]],"CLASSIFICATION"))[0]
                    candidates=[(cat,max(float(np.dot(vector,v)/(np.linalg.norm(vector)*np.linalg.norm(v))) for v in vectors)) for cat,vectors in recognizer._tpl_embeddings.items()]
                    cat,score=max(candidates,key=lambda v:v[1])
                    return {"intent":cat,"confidence":score},(time.perf_counter()-t)*1000
                lv,ev=await asyncio.gather(llm(),emb())
                pattern=recognizer._pattern_recognize(case["message"])
                raw[case["id"]]={"case":case,"llm":{**lv[0],"intent":lv[0]["intent"].value},"embedding":{**ev[0],"intent":ev[0]["intent"].value},
                     "pattern":{**pattern,"intent":pattern["intent"].value},"llm_ms":lv[1],"embedding_ms":ev[1],
                     "llm_reused":earlier_llm.get(case["id"],{}).get("status")=="measured",
                     "parallel_ms":None if earlier_llm.get(case["id"],{}).get("status")=="measured" else (time.perf_counter()-start)*1000,
                     "status":"failed" if lv[0].get("failed") else "measured"}
                write(raw_path,raw)
                print(f"intent {split} {n+1}/{len(cases)} status={raw[case['id']]['status']}",flush=True)
                if lv[0].get("failed"):
                    raise RuntimeError("real_llm_classification_failed")
            thresholds=[.4,.5,.6] if split=="dev" else [read(self.output/"selection.json")["intent_threshold"]]
            result={"status":"measured","startup":startup,"thresholds":{},"diagnostics":[v for v in raw.values() if "allowed_intents" in v["case"]]}
            for threshold in thresholds:
                result["thresholds"][str(threshold)]={}
                for mode in ["two","three"]:
                    rows=[]
                    for value in raw.values():
                        if "intent" not in value["case"]:
                            continue
                        votes={k:{**value[k],"intent":IntentCategory(value[k]["intent"])} for k in ["llm","embedding","pattern"]}
                        recognizer.threshold=threshold
                        recognizer._embedding_enabled=mode=="three"
                        predicted,confidence,scores=recognizer._vote(votes["llm"],votes["embedding"],votes["pattern"])
                        rows.append({"id":value["case"]["id"],"family_id":value["case"]["family_id"],"expected":value["case"]["intent"],"predicted":predicted.value,"confidence":confidence,"scores":scores})
                    per_class={}
                    for label in self.intent_data["labels"]:
                        tp=sum(r["expected"]==label and r["predicted"]==label for r in rows)
                        predicted=sum(r["predicted"]==label for r in rows)
                        expected=sum(r["expected"]==label for r in rows)
                        per_class[label]={"precision":tp/predicted if predicted else 0,"recall":tp/expected if expected else 0,"n":expected}
                    result["thresholds"][str(threshold)][mode]={"macro_f1":macro_f1(rows),"accuracy":sum(r["predicted"]==r["expected"] for r in rows)/len(rows),
                        "other_fraction":sum(r["predicted"]=="other" for r in rows)/len(rows),"per_class":per_class,
                        "confusion":dict(Counter(f"{r['expected']}->{r['predicted']}" for r in rows)),"rows":rows}
            write(self.output/f"intent_{split}.json",result)
        finally:
            await recognizer.close()
            await restarted.close()

    def select(self):
        budget=self.choose_chunk()
        overlap=self.choose_overlap(budget)
        weights={}
        for encoder,b,ov,options in [("gemini",budget,overlap,[.25,.5,.75]),("minilm",240,0,[.1,.25,.5])]:
            values=[read(self.output/"retrieval"/f"{encoder}-structure-{b}-{ov}-dev-{w}.json") for w in options]
            if any(v is None for v in values):
                raise RuntimeError("fusion_not_complete")
            current=next(v for v in values if v["mode"]=="0.5")
            best=max(values,key=lambda v:(v["recall_at_5"],v["precision_at_3"]))
            weights[encoder]=float(best["mode"]) if best["recall_at_5"]>=current["recall_at_5"]+.02 and best["precision_at_3"]>=current["precision_at_3"] else .5
        intent=read(self.output/"intent_dev.json")
        if not intent or intent["status"]!="measured":
            raise RuntimeError("intent_dev_not_complete")
        scores={float(k):v["three"]["macro_f1"] for k,v in intent["thresholds"].items()}
        best=max(scores,key=scores.get)
        threshold=best if scores[best]>scores[.5]+.02 else .5
        write(self.output/"selection.json",{"status":"frozen_before_test","data_version":self.rag["version"],"main_budget":budget,"main_overlap":overlap,
            "main_weight":weights["gemini"],"backup_weight":weights["minilm"],"intent_threshold":threshold,"recursive_control_budget":512,
            "answer_ids":self.rag["answer_ids"],"rationale":"仅开发集判断；差异不足2个百分点保留起点，overlap没有清楚收益选0；纯BM25仅对照。正式测试不再搜索参数。"})

    async def intent_latency(self):
        """另测真实recognize路径，准备成本、缓存启动和热请求分别保留。"""
        path=self.output/"latency_templates.json"
        selection=read(self.output/"selection.json")
        threshold=selection["intent_threshold"]
        cases=[c for c in self.intent_data["cases"] if c["split"]=="test"][:4]
        cold=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.google,
            confidence_threshold=threshold,template_cache_path=str(path),template_wait_s=3)
        before=dict(self.google.ledger)
        started=time.perf_counter()
        value=await cold.recognize(cases[0]["message"])
        cold_request_ms=(time.perf_counter()-started)*1000
        prep_started=started
        await cold._template_task
        preparation_ms=(time.perf_counter()-prep_started)*1000
        cold_usage={k:self.google.ledger[k]-before[k] for k in ["google_http","google_encoded_texts","llm_calls"]}
        restarted=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.google,
            confidence_threshold=threshold,template_cache_path=str(path),template_wait_s=3)
        before=dict(self.google.ledger)
        started=time.perf_counter()
        await restarted._load_template_embeddings()
        hit_start_ms=(time.perf_counter()-started)*1000
        hit_template_texts=self.google.ledger["google_encoded_texts"]-before["google_encoded_texts"]
        rows=[]
        for case in cases[1:]:
            before=dict(self.google.ledger)
            value=await restarted.recognize(case["message"])
            rows.append({"id":case["id"],"wall_ms":value.latency_ms,"intent":value.intent.value,"scores":value.source_scores,
                         "usage":{k:self.google.ledger[k]-before[k] for k in ["google_http","google_encoded_texts","llm_calls"]}})
        write(self.output/"intent_latency.json",{"status":"measured","cold_first_request_ms":cold_request_ms,
            "cold_preparation_until_ready_ms":preparation_ms,"cold_total_usage":cold_usage,"cache_hit_startup_ms":hit_start_ms,
            "cache_hit_template_texts":hit_template_texts,"hot_requests":rows,"latency_includes_experiment_throttle":True,
            "note":"首次请求有界等待后可按85/15降级；后台模板准备成本另列，没有消失。"})
        await cold.close()
        await restarted.close()

    async def intent_llm(self, split):
        if self.gateway is None:
            raise RuntimeError("chat_credentials_missing")
        recognizer=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.google,
            template_cache_path=str(self.output/"intent_templates.json"))
        recognizer._embedding_enabled=False
        path=self.output/f"intent_llm_raw_{split}.json"
        raw=read(path,{})
        cases=[c for c in self.intent_data["cases"] if c["split"]==split]
        if split=="test":
            cases+=self.intent_data["diagnostics"]
        try:
            for n,case in enumerate(cases):
                if raw.get(case["id"],{}).get("status")=="measured":
                    continue
                previous=raw.get(case["id"])
                if previous:
                    write(self.output/"failed_intent"/f"{case['id']}-{datetime.now().strftime('%H%M%S%f')}.json",previous)
                started=time.perf_counter()
                value=await recognizer._llm_recognize(case["message"],None)
                raw[case["id"]]={"case":case,"llm":{**value,"intent":value["intent"].value},
                    "pattern":{**recognizer._pattern_recognize(case["message"]),"intent":recognizer._pattern_recognize(case["message"])["intent"].value},
                    "llm_ms":(time.perf_counter()-started)*1000,"status":"failed" if value.get("failed") else "measured",
                    "embedding_status":"blocked_by_google_daily_quota"}
                write(path,raw)
                print(f"intent_llm {split} {n+1}/{len(cases)} status={raw[case['id']]['status']}",flush=True)
                if value.get("failed"):
                    raise RuntimeError("real_llm_classification_failed")
            output={"status":"measured_two_route_only","three_route_status":"blocked_by_google_daily_quota","thresholds":{}}
            for threshold in ([.4,.5,.6] if split=="dev" else [.5]):
                recognizer.threshold=threshold
                rows=[]
                for value in raw.values():
                    if "intent" not in value["case"]:
                        continue
                    llm={**value["llm"],"intent":IntentCategory(value["llm"]["intent"])}
                    pattern={**value["pattern"],"intent":IntentCategory(value["pattern"]["intent"])}
                    predicted,confidence,scores=recognizer._vote(llm,{"failed":True},pattern)
                    rows.append({"id":value["case"]["id"],"family_id":value["case"]["family_id"],"expected":value["case"]["intent"],"predicted":predicted.value,"confidence":confidence,"scores":scores})
                per_class={}
                for label in self.intent_data["labels"]:
                    tp=sum(r["predicted"]==label and r["expected"]==label for r in rows)
                    predicted=sum(r["predicted"]==label for r in rows)
                    expected=sum(r["expected"]==label for r in rows)
                    per_class[label]={"precision":tp/predicted if predicted else 0,"recall":tp/expected if expected else 0,"n":expected}
                output["thresholds"][str(threshold)]={"two":{"macro_f1":macro_f1(rows),"accuracy":sum(r["predicted"]==r["expected"] for r in rows)/len(rows),
                    "other_fraction":sum(r["predicted"]=="other" for r in rows)/len(rows),"per_class":per_class,"rows":rows,
                    "confusion":dict(Counter(f"{r['expected']}->{r['predicted']}" for r in rows))},"three":{"status":"blocked","reason":self.google.blocked}}
            output["diagnostics"]=[v for v in raw.values() if "allowed_intents" in v["case"]]
            write(self.output/f"intent_llm_{split}.json",output)
        finally:
            await recognizer.close()

    def select_local(self):
        values=[read(self.output/"retrieval"/f"minilm-structure-240-0-dev-{w}.json") for w in [.1,.25,.5]]
        if any(v is None for v in values):
            raise RuntimeError("backup_development_not_complete")
        current=values[-1]
        best=max(values,key=lambda v:(v["recall_at_5"],v["precision_at_3"]))
        weight=float(best["mode"]) if best["recall_at_5"]>=current["recall_at_5"]+.02 and best["precision_at_3"]>=current["precision_at_3"] else .5
        write(self.output/"selection.json",{"status":"partial_frozen_before_test","data_version":self.rag["version"],"main_budget":512,"main_overlap":0,"main_weight":.5,
            "backup_weight":weight,"intent_threshold":.5,"recursive_control_budget":512,"answer_ids":self.rag["answer_ids"],
            "main_selection_status":"blocked_by_google_daily_quota_keep_current","intent_three_selection_status":"blocked_keep_original_templates_and_threshold",
            "rationale":"备用权重仅用开发集选择；Google每日额度阻塞主分块/overlap/三路意图，主参数与意图阈值暂保持当前，未声称完成选择。"})

    async def answers(self, backup_only=False):
        if self.gateway is None:
            raise RuntimeError("chat_credentials_missing")
        selection=read(self.output/"selection.json")
        if not selection:
            raise RuntimeError("selection_missing")
        specs=[("gemini",512,0,.5),("gemini",selection["main_budget"],selection["main_overlap"],selection["main_weight"]),("minilm",240,0,selection["backup_weight"])]
        if backup_only:
            specs=specs[-1:]
        manager=MCPToolManager("experiment",model=self.model,gateway=self.gateway)
        os.environ["RAG_RERANK_MAX_CHARS"]="12000"
        questions={q["id"]:q for q in self.rag["questions"]}
        for encoder,budget,overlap,weight in dict.fromkeys(specs):
            name=f"{encoder}-structure-{budget}-{overlap}-test-{weight}"
            retrieved=read(self.output/"retrieval"/f"{name}.json")
            if not retrieved:
                raise RuntimeError("formal_retrieval_missing")
            by_id={r["id"]:r for r in retrieved["rows"]}
            for qid in self.rag["answer_ids"]:
                path=self.output/"answers"/f"{name}-{qid}.json"
                if read(path,{}).get("status")=="measured":
                    continue
                previous=read(path)
                if previous:
                    write(self.output/"failed_answers"/f"{path.stem}-{datetime.now().strftime('%H%M%S%f')}.json",previous)
                q=questions[qid]
                items=by_id[qid]["retrieved"]
                before_success=manager.rerank_stats["llm_success"]
                ranked=await manager._rerank(q["question"],items,5)
                chosen=[]
                remaining=6000
                for item in ranked:
                    block=f"[{len(chosen)+1}] {item['title']} / {item['section_path']}\n{item['content']}"
                    if len(block)<=remaining:
                        chosen.append(block)
                        remaining-=len(block)+2
                context="\n\n".join(chosen)
                record={"question":q,"retrieved":items,"reranked":ranked,"context":context,"context_budget_chars":6000,
                    "rerank_input_budget_chars":12000,"rerank_status":"measured" if manager.rerank_stats["llm_success"]>before_success else "failed_fallback",
                    "evaluation":"real_llm_audit_not_human","model":self.model}
                try:
                    answer=await self.gateway.complete(LLMRequest(self.model,"仅依据所给知识回答并引用编号，完整说明版本渠道、条件和例外；没有标准时明确材料未规定，不凭常识填数。",messages=[{"role":"user","content":f"问题：{q['question']}\n证据：\n{context}"}],max_tokens=4096,temperature=0))
                    record["answer"]=answer.text
                    record["answer_status"]="measured"
                    write(path,record)
                    audit=await self.gateway.complete(LLMRequest(self.model,"审核回答是否由原文支持。仅输出JSON。不得把检索非空算作无答案编造。",messages=[{"role":"user","content":json.dumps({"question":q,"answer":answer.text,"context":context,
                        "output":{"facts_supported":"布尔","conditions_complete":"布尔","unanswerable_fabricated":"布尔","unsupported_claims":[],"citation_errors":[],"explanation":"依据"}},ensure_ascii=False)}],max_tokens=4096,temperature=0))
                    record["audit_raw"]=audit.text
                    record["audit"]=json.loads(audit.text[audit.text.find("{"):audit.text.rfind("}")+1])
                    if any(type(record["audit"].get(k)) is not bool for k in ["facts_supported","conditions_complete","unanswerable_fabricated"]) or any(not isinstance(record["audit"].get(k),list) for k in ["unsupported_claims","citation_errors"]):
                        raise ValueError("invalid_answer_audit_contract")
                    record["status"]="measured"
                except Exception as ex:
                    record.update(status="failed",reason=KnowledgeBase._reason(ex))
                write(path,record)
                print(f"answer {name} {qid} status={record['status']}",flush=True)
        if any(read(path).get("status")!="measured" for path in (self.output/"answers").glob("*.json")):
            raise RuntimeError("answer_or_audit_failed")

    async def production(self, backup_only=False):
        previous=read(self.output/"production.json")
        if previous and previous.get("status")!="measured":
            write(self.output/"failed_production"/f"{datetime.now().strftime('%H%M%S%f')}.json",previous)
        selection=read(self.output/"selection.json")
        if not selection:
            raise RuntimeError("selection_missing")
        self.prepare_mini()
        # 复用真实文档向量，但这20次查询实际调用Google并走Chroma/工具。
        original_embed=self.google.embed
        def production_embed(texts,task,titles=None):
            return self.google.cached_embed(texts,task,titles) if task=="RETRIEVAL_DOCUMENT" else original_embed(texts,task,titles)
        self.google.embed=production_embed
        path=self.output/"production-index"
        client=chromadb.PersistentClient(path=str(path),settings=chromadb.Settings(anonymized_telemetry=False))
        os.environ["RAG_CHUNK_TOKENS"]=str(selection["main_budget"])
        os.environ["RAG_OVERLAP_TOKENS"]=str(selection["main_overlap"])
        os.environ["RAG_MAIN_VECTOR_WEIGHT"]=str(selection["main_weight"])
        os.environ["RAG_BACKUP_VECTOR_WEIGHT"]=str(selection["backup_weight"])
        kb=KnowledgeBase(chroma_path=str(path),embedding=self.google,client=client,backup_function=self.mini,backup_counter=self.mini_count,
                         load_defaults=False,collection_prefix="v2production")
        manager=MCPToolManager("experiment",model=self.model or "missing",gateway=self.gateway)
        manager.register(Tool("knowledge_search","knowledge",kb.search_handler,{},cache_ttl=300))
        kb.on_change=manager.invalidate_cache
        rows=[]
        try:
            imported=await asyncio.to_thread(kb.import_documents,self.corpus)
            if imported["status"]!="success" and not backup_only:
                raise RuntimeError("production_import_incomplete")
            # 文档批量限速留给下一批的等待在在线测量之前结清。
            delay=max(0,self.google._next_request-time.monotonic())
            if delay:
                await asyncio.sleep(delay)
                self.google.usage["throttle_sleep_ms"]+=delay*1000
            # 本机预热后的轻量端点观测；硬性并发正确性由同步事件测试证明。
            from api import main as api
            previous = (api._knowledge_base, api._monitor, api._orchestrator)
            api._knowledge_base = kb
            api._monitor = SimpleNamespace(summary=lambda: {"knowledge":kb.stats()})
            api._orchestrator = SimpleNamespace(get_stats=lambda: {})
            health_ms, heartbeats = [], []
            async def heartbeat():
                last=time.perf_counter()
                for _ in range(40):
                    await asyncio.sleep(.005)
                    now=time.perf_counter()
                    heartbeats.append((now-last)*1000)
                    last=now
            try:
                beat=asyncio.create_task(heartbeat())
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url="http://test") as web:
                    for _ in range(20):
                        t=time.perf_counter()
                        response=await web.get("/health")
                        assert response.status_code==200
                        health_ms.append((time.perf_counter()-t)*1000)
                        await asyncio.gather(web.get("/monitor"),web.get("/knowledge/stats"))
                await beat
                write(self.output/"endpoint_observations.json",{"status":"measured","n":20,"health_p50_ms":statistics.median(health_ms),
                    "health_p95_ms":sorted(health_ms)[18],"max_heartbeat_interval_ms":max(heartbeats),"health_rows_ms":health_ms,
                    "scope":"真实知识库/Chroma统计、ASGI端点，本机预热轻量观测；编排器统计为固定就绪替身，不是全部署负载测试"})
            finally:
                api._knowledge_base,api._monitor,api._orchestrator=previous
            questions=[q for q in self.rag["questions"] if q["split"]=="test"][:20]
            for q in questions:
                started=time.perf_counter()
                result=await manager.call("knowledge_search",{"query":q["question"],"top_k":5})
                if not result.success:
                    raise RuntimeError("production_tool_failed")
                row=self.evaluate(q,result.data)
                row.update(route=result.data[0]["index_route"] if result.data else "empty",wall_ms=(time.perf_counter()-started)*1000,cached=result.cached)
                rows.append(row)
                write(self.output/("backup_production_progress.json" if backup_only else "production.json"),{"status":"running","rows":rows})
            if backup_only:
                changed={"title":"本地恢复验证","source_id":"v2-recovery","content":"测试规则：七天受理。","format":"md"}
                first=await asyncio.to_thread(kb.import_documents,[changed])
                changed["content"]="测试新规则：九天受理，须提供订单号。"
                updated=await asyncio.to_thread(kb.import_documents,[changed])
                pending=await manager.call("knowledge_search",{"query":"本地恢复验证九天订单号"},use_cache=False)
                pending_before=kb.stats()["pending_main"]
                kb.close()
                kb=KnowledgeBase(chroma_path=str(path),embedding=self.google,client=client,backup_function=self.mini,backup_counter=self.mini_count,load_defaults=False,collection_prefix="v2production",on_change=manager.invalidate_cache)
                manager._tools["knowledge_search"].handler=kb.search_handler
                pending_after=kb.stats()["pending_main"]
                await asyncio.to_thread(kb.delete_document,"v2-recovery")
                removed=await manager.call("knowledge_search",{"query":"本地恢复验证九天订单号"},use_cache=False)
                checks={"real_backup_20":len(rows)==20 and all(r["route"]=="backup" for r in rows),
                    "google_degraded_import":first["status"]=="degraded" and updated["status"]=="degraded", "pending_stays_backup":pending.success and pending.degraded,
                    "new_content":any(i["source_id"]=="v2-recovery" and "九天" in i["content"] for i in pending.data),
                    "old_absent":not any("七天受理" in i["content"] for i in pending.data),"restart_pending":pending_before==pending_after and pending_after>0,
                    "deleted_backup":not any(i["source_id"]=="v2-recovery" for i in removed.data),
                    "deleted_main_storage":not kb._main.get(where={"source_id":"v2-recovery"},include=[])["ids"]}
                # 单文档主索引由本轮已取得的真实Gemini向量建立；延迟注入触发真实MiniLM。
                fault_path=self.output/"delay-fault-index"
                fault_client=chromadb.PersistentClient(path=str(fault_path),settings=chromadb.Settings(anonymized_telemetry=False))
                fault=KnowledgeBase(chroma_path=str(fault_path),embedding=self.google,client=fault_client,backup_function=self.mini,backup_counter=self.mini_count,load_defaults=False,collection_prefix="v2fault")
                accepted=await asyncio.to_thread(fault.import_documents,[self.corpus[0]])
                def delayed(*args,**kwargs):
                    with self.google.query_budget(.1):
                        time.sleep(.15)
                        self.google._remaining()
                self.google.embed=delayed
                fault_manager=MCPToolManager("experiment",model=self.model or "missing",gateway=self.gateway)
                fault_manager.register(Tool("knowledge_search","knowledge",fault.search_handler,{},timeout_s=1))
                started=time.perf_counter()
                result=await fault_manager.call("knowledge_search",{"query":"星河相机质量退款E610"})
                fault_wall=(time.perf_counter()-started)*1000
                checks["delay_fault_real_minilm"]=accepted["status"]=="success" and result.success and result.degraded and any("E610" in i["content"] for i in result.data) and fault_wall<1000
                self.google.embed=production_embed
                fault.close()
                rewrite=[]
                if self.gateway:
                    for q in questions[:2]:
                        value=await manager.search_with_rewrite("knowledge_search",q["question"],top_k=5)
                        rewrite.append({"id":q["id"],"success":value.success,"results":value.data,"degraded":value.degraded})
                write(self.output/"backup_production.json",{"status":"measured" if all(checks.values()) else "failed","rows":rows,"checks":checks,"stats":kb.stats(),
                    "delay_fault_wall_ms":fault_wall,"delay_fault_results":result.data,"fault":"injected_synchronous_delay; cached_real_Gemini_documents_and_real_MiniLM_query",
                    "rewrite":rewrite,"main_repair_status":"blocked_by_google_daily_quota","main_recovery_status":"not_measured","deployment_boundaries":"Persistent Chroma/工具验证；HTTP Chroma/Redis未覆盖"})
                if not all(checks.values()):
                    raise RuntimeError("backup_production_checks_failed")
                return
            def failure(*args,**kwargs):
                raise EmbeddingError("injected_google_failure")
            self.google.embed=failure
            query=questions[0]["question"]
            fallback=await manager.call("knowledge_search",{"query":query},use_cache=False)
            self.google.embed=production_embed
            recovery=await manager.call("knowledge_search",{"query":query},use_cache=False)
            self.google.embed=failure
            changed={"title":"故障恢复测试","source_id":"v2-recovery","content":"仅用于集成验收：旧规则七天受理。","format":"md"}
            first=await asyncio.to_thread(kb.import_documents,[changed])
            changed["content"]="仅用于集成验收：新规则九天受理，必须提供订单号。"
            updated=await asyncio.to_thread(kb.import_documents,[changed])
            self.google.embed=production_embed
            pending=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            kb.close()
            kb=KnowledgeBase(chroma_path=str(path),embedding=self.google,client=client,backup_function=self.mini,backup_counter=self.mini_count,load_defaults=False,collection_prefix="v2production",on_change=manager.invalidate_cache)
            manager._tools["knowledge_search"].handler=kb.search_handler
            restart_pending=kb.stats()["pending_main"]
            repaired=await asyncio.to_thread(kb.repair_main)
            new_main=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            await asyncio.to_thread(kb.delete_document,"v2-recovery")
            deleted=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            self.google.embed=failure
            deleted_backup=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            self.google.embed=production_embed
            checks={"fallback_real_body":fallback.success and fallback.degraded and bool(fallback.data),"recovered_main":recovery.success and all(i["index_route"]=="main" for i in recovery.data),
                    "degraded_import":first["status"]=="degraded" and updated["status"]=="degraded","pending_stays_backup":pending.success and pending.degraded,
                    "update_visible":any("九天" in i["content"] for i in pending.data),"old_absent":not any("七天受理" in i["content"] for i in pending.data),
                    "restart_pending":restart_pending==1,"repair_complete":repaired["main_complete"],"new_main":any(i["source_id"]=="v2-recovery" and "九天" in i["content"] and i["index_route"]=="main" for i in new_main.data),
                    "deleted_both":not any(i["source_id"]=="v2-recovery" for i in deleted.data+deleted_backup.data)}
            # 真实MiniLM + 延迟注入，预算检查独立标记。
            def delayed(*args,**kwargs):
                with self.google.query_budget(.1):
                    time.sleep(.15)
                    self.google._remaining()
            self.google.embed=delayed
            started=time.perf_counter()
            slow=await manager.call("knowledge_search",{"query":query},use_cache=False)
            slow_ms=(time.perf_counter()-started)*1000
            checks["injected_delay_returns_minilm"]=slow.success and slow.degraded and bool(slow.data) and slow_ms<30000
            self.google.embed=production_embed
            rewrite=[]
            if self.gateway:
                for q in questions[:2]:
                    value=await manager.search_with_rewrite("knowledge_search",q["question"],top_k=5)
                    rewrite.append({"id":q["id"],"success":value.success,"results":value.data,"degraded":value.degraded})
            write(self.output/"production.json",{"status":"measured" if all(checks.values()) else "failed","rows":rows,"checks":checks,"stats":kb.stats(),
                  "fault":"injected_google_failure_and_delay; real_minilm_and_chroma","slow_fault_wall_ms":slow_ms,"rewrite":rewrite,
                  "deployment_boundaries":"Persistent Chroma verified; HTTP Chroma/Redis未启动完整部署"})
            if not all(checks.values()):
                raise RuntimeError("production_checks_failed")
        finally:
            kb.close()
            self.google.embed=original_embed

    async def production_cache(self):
        """真实降级结果不进入工具缓存，避免缓存锁住备用路由。"""
        previous=read(self.output/"production_cache.json")
        if previous:
            write(self.output/"failed_production_cache"/f"{datetime.now().strftime('%H%M%S%f')}.json",previous)
        self.prepare_mini()
        selection=read(self.output/"selection.json")
        os.environ["RAG_BACKUP_VECTOR_WEIGHT"]=str(selection["backup_weight"])
        path=self.output/"production-index"
        client=chromadb.PersistentClient(path=str(path),settings=chromadb.Settings(anonymized_telemetry=False))
        kb=KnowledgeBase(chroma_path=str(path),embedding=self.google,client=client,backup_function=self.mini,
                         backup_counter=self.mini_count,load_defaults=False,collection_prefix="v2production")
        manager=MCPToolManager("experiment",model=self.model or "missing",gateway=self.gateway)
        manager.register(Tool("knowledge_search","knowledge",kb.search_handler,{},cache_ttl=300))
        query=next(q["question"] for q in self.rag["questions"] if q["split"]=="test")
        rows=[]
        try:
            for step in ["first","repeat","invalidated"]:
                if step=="invalidated":
                    manager.invalidate_cache()
                started=time.perf_counter()
                result=await manager.call("knowledge_search",{"query":query,"top_k":5})
                rows.append({"step":step,"cached":result.cached,"success":result.success,"degraded":result.degraded,
                             "wall_ms":(time.perf_counter()-started)*1000,"results":result.data})
            passed=all(r["success"] and r["degraded"] and not r["cached"] for r in rows)
            passed=passed and rows[0]["results"]==rows[1]["results"]==rows[2]["results"]
            write(self.output/"production_cache.json",{"status":"measured" if passed else "failed","rows":rows,
                                                       "scope":"实际Persistent Chroma降级重复调用不缓存；显式失效，主路径缓存命中待Google恢复"})
            if not passed:
                raise RuntimeError("production_cache_checks_failed")
        finally:
            kb.close()

    async def execute(self,stage):
        if stage=="validate":
            self.validate()
        elif stage=="smoke":
            self.retrieval("gemini","structure",512,0,"dev",["vector","bm25","0.5"],3)
            self.retrieval("minilm","structure",240,0,"dev",["vector","bm25","0.5"],3)
        elif stage=="chunking":
            for kind in ["recursive","structure"]:
                for budget in [256,512,768]:
                    self.retrieval("gemini",kind,budget,0,"dev",["vector"])
        elif stage=="overlap":
            budget=self.choose_chunk()
            for overlap in [32,64]:
                self.retrieval("gemini","structure",budget,overlap,"dev",["vector"])
        elif stage=="fusion":
            budget=self.choose_chunk()
            overlap=self.choose_overlap(budget)
            self.retrieval("gemini","structure",512,0,"dev",["vector","bm25","0.5"])
            self.retrieval("gemini","structure",budget,overlap,"dev",["vector","bm25","0.25","0.5","0.75"])
            if overlap:
                self.retrieval("gemini","structure",budget,0,"dev",["0.25","0.5","0.75"])
            self.retrieval("minilm","structure",240,0,"dev",["vector","bm25","0.1","0.25","0.5"])
        elif stage=="intent_dev":
            await self.intent("dev")
        elif stage=="select":
            self.select()
        elif stage=="test":
            selection=read(self.output/"selection.json")
            if not selection:
                raise RuntimeError("selection_missing")
            self.retrieval("gemini","structure",512,0,"test",["0.5"])
            self.retrieval("gemini","structure",selection["main_budget"],selection["main_overlap"],"test",["vector","bm25",str(selection["main_weight"])])
            if selection["main_overlap"]:
                self.retrieval("gemini","structure",selection["main_budget"],0,"test",[str(selection["main_weight"])])
            self.retrieval("gemini","recursive",selection["recursive_control_budget"],0,"test",["vector"])
            self.retrieval("minilm","structure",240,0,"test",["vector","bm25","0.5",str(selection["backup_weight"])])
        elif stage=="intent_test":
            await self.intent("test")
            await self.intent_latency()
        elif stage=="answers":
            await self.answers()
        elif stage=="production":
            await self.production()
        elif stage=="backup_dev":
            self.retrieval("minilm","structure",240,0,"dev",["vector","bm25","0.1","0.25","0.5"])
            self.retrieval("gemini","structure",512,0,"dev",["bm25"])
        elif stage.startswith("intent_llm_"):
            await self.intent_llm(stage.rsplit("_",1)[1])
        elif stage=="local_select":
            self.select_local()
        elif stage=="local_test":
            selection=read(self.output/"selection.json")
            self.retrieval("minilm","structure",240,0,"test",list(dict.fromkeys(["vector","bm25","0.5",str(selection["backup_weight"])])))
            self.retrieval("gemini","structure",512,0,"test",["bm25"])
        elif stage=="backup_answers":
            await self.answers(backup_only=True)
        elif stage=="backup_production":
            await self.production(backup_only=True)
        elif stage=="production_cache":
            await self.production_cache()

    async def run(self):
        stages=STAGES if self.args.stage=="all" else LOCAL_STAGES if self.args.stage=="local" else [self.args.stage]
        for stage in stages:
            if self.args.resume and self.manifest["stages"].get(stage,{}).get("status")=="measured":
                continue
            started=time.perf_counter()
            previous=self.manifest["stages"].get(stage,{})
            result={"status":"running","attempt":previous.get("attempt",0)+1}
            self.manifest["stages"][stage]=result
            write(self.output/"manifest.json",self.manifest)
            try:
                print(f"stage={stage} started",flush=True)
                dependencies = {"smoke":["validate"], "chunking":["smoke"], "overlap":["chunking"],
                    "fusion":["overlap"], "intent_dev":["validate"], "select":["fusion","intent_dev"],
                    "test":["select"], "intent_test":["select"], "answers":["test"], "production":["select","test"]}
                dependencies.update({"backup_dev":["validate"],"intent_llm_dev":["validate"],"local_select":["backup_dev"],
                    "local_test":["local_select"],"intent_llm_test":["local_select","intent_llm_dev"],"backup_answers":["local_test"],"backup_production":["local_test"],"production_cache":["backup_production"]})
                if any(self.manifest["stages"].get(dep,{}).get("status")!="measured" for dep in dependencies.get(stage,[])):
                    raise RuntimeError("required_stage_not_complete")
                await self.execute(stage)
                result["status"]="measured"
            except Exception as ex:
                reason=KnowledgeBase._reason(ex)
                if isinstance(ex,RuntimeError):
                    reason=str(ex)
                result.update(status="blocked" if "missing" in reason or "not_complete" in reason or "budget_exhausted" in reason or "PerDay" in reason else "failed",reason=reason)
                print(f"stage={stage} {result['status']} reason={reason}",flush=True)
            result["wall_ms"]=(time.perf_counter()-started)*1000
            result["api_usage"]=dict(self.google.ledger)
            write(self.output/"attempts"/f"{stage}-{result['attempt']}.json",result)
            write(self.output/"manifest.json",self.manifest)
            self.google.flush()
            print(f"stage={stage} finished status={result['status']} google_http={self.google.ledger['google_http']} google_texts={self.google.ledger['google_encoded_texts']} llm={self.google.ledger['llm_calls']}",flush=True)
        self.google.close()
        if self.gateway and hasattr(self.gateway.gateway,"close"):
            await self.gateway.gateway.close()
        return all(self.manifest["stages"].get(stage,{}).get("status")=="measured" for stage in stages)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=STAGES+LOCAL_STAGES+["all","local"],default="all")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rag-data",type=Path,default=ROOT/"data/eval/rag_optimization_v2.json")
    parser.add_argument("--intent-data",type=Path,default=ROOT/"data/eval/intent_optimization_v2.json")
    parser.add_argument("--api-interval",type=float,default=1)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--retry-google",action="store_true",help="额度恢复后显式解除已保存的Google停止状态；仍沿用同一累计预算")
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(Runner(args).run()) else 1)


if __name__=="__main__":
    main()
