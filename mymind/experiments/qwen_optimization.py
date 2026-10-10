"""Qwen 第二轮：冻结语料、顺序选择、真实编码和累计费用断点。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sqlite3
import statistics
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import chromadb
import httpx
import numpy as np
from dotenv import load_dotenv

from core.document_chunking import BudgetChunker, parse_sections
from core.embedding import EmbeddingError, EmbeddingSettings, QwenEmbedding
from core.intent_recognizer import IntentRecognizer, IntentCategory
from core.llm_gateway import LLMRequest
from core.retrieval import BM25Index, weighted_rrf, dedupe_items, RRF_K
from experiments.gemini_rag import chat_gateway, normalize, duplicate_fraction, macro_f1
from experiments.retrieval_scoring import evaluate_retrieval, summarize_retrieval, rescore_retrieval_group
from experiments.optimization_v2 import read, write
from mcp.indexed_knowledge_base import KnowledgeBase, prepare_minilm
from mcp.tool_manager import MCPToolManager, Tool

ROOT = Path(__file__).resolve().parents[1]
STAGES = ['validate', 'smoke', 'chunking', 'overlap', 'fusion', 'intent_dev', 'select', 'test', 'intent_test', 'answers', 'production', 'application']


class BudgetQwen(QwenEmbedding):
    def __init__(self, settings, output, interval):
        super().__init__(settings, min_interval=interval)
        self.output = output
        self.lock = threading.RLock()
        self.db = sqlite3.connect(output/'encoding_cache.sqlite3', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS counts (text TEXT PRIMARY KEY, tokens INTEGER, vector TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector TEXT)')
        self.ledger = read(output/'api_usage.json', {'http':0, 'encoded_texts':0, 'count_requests':0, 'reported_tokens':0, 'llm_calls':0, 'throttle_sleep_ms':0})
        self.reuse = Counter(read(output/'cache_reuse.json', {}))
        self._base_wait = self.ledger['throttle_sleep_ms']

    def flush(self):
        with self.lock:
            self.ledger['throttle_sleep_ms'] = self._base_wait + self.usage['throttle_sleep_ms']
            write(self.output/'api_usage.json', self.ledger)
            write(self.output/'cache_reuse.json', dict(self.reuse))

    def _send(self, method, url, headers, payload=None, units=1):
        with self.lock:
            # 历史 Gemini + 已完成 Qwen 接入 smoke，均扣除授权总预算。
            if self.ledger['http']+685 >= 20000 or self.ledger['encoded_texts']+264+units > 10000:
                raise EmbeddingError('authorized_budget_exhausted')
            self.ledger['http'] += 1
            self.ledger['encoded_texts'] += units
            self.flush()
        try:
            return super()._send(method,url,headers,payload,units)
        except Exception as ex:
            with self.lock:
                failures=read(self.output/'api_failures.json',[])
                failures.append({'operation':'embedding', 'reason':KnowledgeBase._reason(ex), 'http_attempt':self.ledger['http']})
                write(self.output/'api_failures.json',failures)
            raise

    def _encode_batch(self, batch, task):
        vectors,tokens=super()._encode_batch(batch,task)
        with self.lock:
            self.ledger['reported_tokens'] += tokens or 0
            self.ledger[f'{task}:texts'] = self.ledger.get(f'{task}:texts',0)+len(batch)
            self.flush()
        return vectors,tokens

    def count_tokens(self,text):
        if not text:
            return 0
        with self.lock:
            row=self.db.execute('SELECT tokens,vector FROM counts WHERE text=?',(text,)).fetchone()
        if row:
            self.reuse['token_count'] += 1
            self._token_cache[text]=(row[0],json.loads(row[1]))
            return row[0]
        with self.lock:
            self.ledger['count_requests'] += 1
        count=super().count_tokens(text)
        with self.lock:
            self.db.execute('INSERT OR REPLACE INTO counts VALUES (?,?,?)',(text,count,json.dumps(self._token_cache[text][1])))
            self.db.commit()
        return count

    def cached_embed(self,texts,task,titles=None):
        keys=[json.dumps([self.identity,task,titles[i] if titles else '',text],ensure_ascii=False) for i,text in enumerate(texts)]
        values={}
        pending=[]
        for key in dict.fromkeys(keys):
            with self.lock:
                row=self.db.execute('SELECT vector FROM vectors WHERE key=?',(key,)).fetchone()
            if row:
                values[key]=json.loads(row[0])
                self.reuse[task] += 1
            else:
                pending.append(key)
        for start in range(0,len(pending),20):
            batch=pending[start:start+20]
            decoded=[json.loads(k) for k in batch]
            if task=='RETRIEVAL_DOCUMENT':
                for _,_,title,text in decoded:
                    canonical=f'{title}\n{text}' if title else text
                    with self.lock:
                        row=self.db.execute('SELECT tokens,vector FROM counts WHERE text=?',(canonical,)).fetchone()
                    if row:
                        self._token_cache[canonical]=(row[0],json.loads(row[1]))
            encoded=super().embed([d[3] for d in decoded],task,[d[2] for d in decoded])
            with self.lock:
                self.db.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?)',[(k,json.dumps(v)) for k,v in zip(batch,encoded)])
                self.db.commit()
            values.update(zip(batch,encoded))
        self.flush()
        return [values[k] for k in keys]

    def has_vector(self,text,task):
        key=json.dumps([self.identity,task,'',text],ensure_ascii=False)
        return self.db.execute('SELECT 1 FROM vectors WHERE key=?',(key,)).fetchone() is not None


class Runner:
    def __init__(self,args):
        self.args,self.output=args,args.output.resolve()
        self.output.mkdir(parents=True,exist_ok=True)
        load_dotenv(ROOT/'.env')
        settings=EmbeddingSettings.from_env()
        if settings.model!='qwen3.7-text-embedding-flash' or settings.dimensions!=768:
            raise ValueError('本轮固定 Qwen Flash 768 维')
        self.embedding=BudgetQwen(settings,self.output,args.api_interval)
        self.rag=read(args.rag_data)
        self.intent_data=read(args.intent_data)
        self.corpus=self.rag['corpus']
        self.by_source={d['source_id']:d for d in self.corpus}
        self.mini=self.mini_count=None
        self.mini_vectors=read(self.output/'mini_vectors.json',read(args.previous/'mini_vectors.json',{}))
        for directory in ['chunks','rankings','retrieval']:
            for source in (args.previous/directory).glob('minilm-*.json'):
                target=self.output/directory/source.name
                if not target.exists():
                    value=read(source)
                    value['reused_from']=str(source.resolve())
                    write(target,value)
        self.gateway,self.model=chat_gateway()
        old_manifest=read(args.previous/'manifest.json')
        if old_manifest['models']['chat']!=self.model:
            raise ValueError('历史意图输出的聊天模型与当前模型不同，不可复用')
        for split in ['dev','test']:
            source=read(args.previous/f'intent_llm_raw_{split}.json')
            cases=[c for c in self.intent_data['cases'] if c['split']==split]+(self.intent_data['diagnostics'] if split=='test' else [])
            if any(source[c['id']]['case']!=c or source[c['id']]['status']!='measured' for c in cases):
                raise ValueError('历史真实 LLM 输出不匹配冻结题目')
            write(self.output/f'intent_llm_raw_{split}.json',source)
        if self.gateway:
            original=self.gateway.complete
            async def measured(request):
                with self.embedding.lock:
                    if self.embedding.ledger['llm_calls']+343 >= 600:
                        raise RuntimeError('authorized_llm_budget_exhausted')
                    self.embedding.ledger['llm_calls'] += 1
                    self.embedding.flush()
                try:
                    result=await original(request)
                except Exception as ex:
                    failures=read(self.output/'llm_failures.json',[])
                    failures.append({'model':request.model,'type':type(ex).__name__,'http_status':getattr(ex,'status_code',None)})
                    write(self.output/'llm_failures.json',failures)
                    raise
                saved=read(self.output/'llm_usage.json',[])
                saved.append({**self.gateway.usage[-1],'response_retry':result.metadata.get('response_retry'),
                              'empty_response_retry':result.metadata.get('empty_response_retry')})
                write(self.output/'llm_usage.json',saved)
                return result
            self.gateway.complete=measured
        self.manifest=read(self.output/'manifest.json',None)
        if self.manifest and self.manifest['models']!={'embedding':settings.model,'dimensions':768,'endpoint':settings.base_url,'backup':'all-MiniLM-L6-v2','chat':self.model}:
            raise ValueError('运行目录模型配置已改变，不能复用向量')
        if not self.manifest:
            self.manifest={'date':'2026-10-10','head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'initial_git_status':subprocess.check_output(['git','status','--short'],text=True),'python':platform.python_version(),
                'environment':'learn_claude','chroma':chromadb.__version__,'data_version':self.rag['version'],
                'models':{'embedding':settings.model,'dimensions':768,'endpoint':settings.base_url,'backup':'all-MiniLM-L6-v2','chat':self.model},
                'counting_mode':self.embedding.counting_mode,'rag_ids':[q['id'] for q in self.rag['questions']],
                'intent_ids':[c['id'] for c in self.intent_data['cases']],'answer_ids':self.rag['answer_ids'],
                'historical_usage':{'http':685,'encoded_texts':264,'llm_calls':343},'limits':{'http':20000,'encoded_texts':10000,'llm_calls':600},
                'previous_run':str(args.previous.resolve()),'reused_llm_outputs':248,'estimate':{'encoded_texts':'约 3000–7000，含真实计数编码','llm_calls':'约 100–200，复用原备用回答'},'stages':{}}
            write(self.output/'manifest.json',self.manifest)

    def reuse_backup_answer(self,path,encoder):
        if encoder!='minilm' or path.exists():
            return
        source=self.args.previous/'answers'/path.name
        saved=read(source)
        if not saved or saved.get('status')!='measured' or saved['model']!=self.model:
            return
        name=path.name.split('-rag-v2-')[0]
        result=read(self.output/'retrieval'/f'{name}.json')
        qid=saved['question']['id']
        row=next(r for r in result['rows'] if r['id']==qid)
        if saved['retrieved']!=row['retrieved'] or saved['context_budget_chars']!=6000 or saved['rerank_input_budget_chars']!=12000:
            raise ValueError('备用回答的上下文契约已改变，不可复用')
        saved['reused_from']=str(source.resolve())
        write(path,saved)

    async def application(self):
        from experiments.qwen_application_acceptance import run
        await run(self)

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
        counter = (lambda text:len(text.encode('utf-8'))) if encoder=='qwen_byte' else self.embedding.count_tokens if encoder == "qwen" else self.mini_count
        space='qwen' if encoder=='qwen_byte' else encoder
        chunker = BudgetChunker(counter, budget, overlap, space, reserve=16 if encoder.startswith('qwen') else 0)
        items = []
        for doc in self.corpus:
            items += [{**r.metadata, "content": r.text} for r in chunker.chunk_document(doc["title"], doc["content"], doc["source_id"], structural=kind=="structure")]
            print(f"chunks {name} source={doc['source_id']} total_chunks={len(items)}",flush=True)
        counts = [counter(BudgetChunker.embedding_title(i["title"], i["section_path"])+"\n"+i["content"]) for i in items]
        assert max(counts) + chunker.reserve <= budget
        assert len(items) >= 50
        write(path, {"data_version": self.rag["version"], "corpus": self.corpus, "items": items, "input_counts": counts,
                     "config": chunker.config, "counting_mode": 'utf8_byte_upper_bound' if encoder=='qwen_byte' else self.embedding.counting_mode if encoder=="qwen" else "minilm_tokenizer"})
        self.embedding.flush()
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
            for mode, value in completed.items():
                completed[mode] = rescore_retrieval_group(value, {q["id"]: q for q in questions})
                write(self.output/"retrieval"/f"{name}-{mode}.json", completed[mode])
            print(f"retrieval {name} reused_completed_groups={','.join(modes)}",flush=True)
            return {mode:value["rows"] for mode,value in completed.items()}
        items=self.chunks(encoder,kind,budget,overlap)
        cached=read(path, {"rows":{}})
        need_vector=any(mode!="bm25" for mode in modes)
        before=time.perf_counter()
        titles=[BudgetChunker.embedding_title(i["title"],i["section_path"]) for i in items]
        if need_vector:
            values=self.embedding.cached_embed([i["content"] for i in items],"RETRIEVAL_DOCUMENT",titles) if encoder.startswith('qwen') else self.mini_embed([f"{t}\n{i['content']}" for t,i in zip(titles,items)])
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
                query_reused=self.embedding.has_vector(q["question"],"RETRIEVAL_QUERY") if encoder.startswith('qwen') else q["question"] in self.mini_vectors
                qv=self.embedding.cached_embed([q["question"]],"RETRIEVAL_QUERY")[0] if encoder.startswith('qwen') else self.mini_embed([q["question"]])[0]
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
        self.embedding.flush()
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
            value=max(self.embedding.count_tokens(BudgetChunker.embedding_title(doc["title"],s.path)+"\n"+s.text) for s in parse_sections(doc["content"]))
            long.append({"source_id":doc["source_id"],"max_section_input_tokens":value})
        assert all(d["max_section_input_tokens"]>768 for d in long)
        assert any(d["max_section_input_tokens"]>1536 for d in long)
        result={"status":"measured","data_frozen":True,"long_sections":long,"families":90,"intent_families":114,
                "answer_ids":self.rag["answer_ids"],"review":"代理语义复核：版本/渠道明确限定，常规办理与故障例外分别处理；非人工审核。合成规则共享知识库问题泛化。"}
        write(self.output/"data_validation.json",result)
        self.embedding.flush()


    def choose_chunk(self):
        values=[read(self.output/"retrieval"/f"qwen-structure-{b}-0-dev-vector.json") for b in [256,512,768]]
        if any(v is None for v in values):
            raise RuntimeError("chunking_not_complete")
        best=max(values,key=lambda v:(v["recall_at_5"],v["precision_at_3"]))
        current=values[1]
        return int(best["config"].split("-")[2]) if best["recall_at_5"]>=current["recall_at_5"]+.02 and best["precision_at_3"]>=current["precision_at_3"] else 512


    def choose_overlap(self,budget):
        values=[read(self.output/"retrieval"/f"qwen-structure-{budget}-{o}-dev-vector.json") for o in [0,32,64]]
        if any(v is None for v in values):
            raise RuntimeError("overlap_not_complete")
        best=max(values,key=lambda v:(v["recall_at_5"],v["precision_at_3"],-int(v["config"].split("-")[3])))
        current=values[0]
        return int(best["config"].split("-")[3]) if best["recall_at_5"]>=current["recall_at_5"]+.02 and best["precision_at_3"]>=current["precision_at_3"] else 0


    async def intent(self,split):
        if self.gateway is None:
            raise RuntimeError("chat_credentials_missing")
        recognizer=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.embedding,
            template_cache_path=str(self.output/"intent_templates.json"),template_wait_s=180)
        cold_started=time.perf_counter()
        before=self.embedding.ledger["encoded_texts"]
        await recognizer._load_template_embeddings()
        template_time=(time.perf_counter()-cold_started)*1000
        template_texts=self.embedding.ledger["encoded_texts"]-before
        # 重启读取同一真实模板缓存，单独记录耗时和编码数。
        restarted=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.embedding,
            template_cache_path=str(self.output/"intent_templates.json"),template_wait_s=180)
        started=time.perf_counter()
        before=self.embedding.ledger["encoded_texts"]
        await restarted._load_template_embeddings()
        startup={"template_prepare_wall_ms":template_time,"template_encoded_texts":template_texts,
                 "cache_hit_startup_ms":(time.perf_counter()-started)*1000,"cache_hit_template_encoded_texts":self.embedding.ledger["encoded_texts"]-before}
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
                    vector=(await asyncio.to_thread(self.embedding.cached_embed,[case["message"]],"CLASSIFICATION"))[0]
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
        for encoder,b,ov,options in [("qwen",budget,overlap,[.25,.5,.75]),("minilm",240,0,[.1,.25,.5])]:
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
            "main_weight":weights["qwen"],"backup_weight":weights["minilm"],"intent_threshold":threshold,"recursive_control_budget":512,
            "answer_ids":self.rag["answer_ids"],"rationale":"仅开发集判断；差异不足2个百分点保留起点，overlap没有清楚收益选0；纯BM25仅对照。正式测试不再搜索参数。"})
        selected=read(self.output/'selection.json')
        selected['frozen_at']=datetime.now().isoformat()
        selected['template_selection']='保留原54条模板；仅开发集选择阈值'
        if overlap:
            with_overlap=read(self.output/'retrieval'/f'qwen-structure-{budget}-{overlap}-dev-{selected["main_weight"]}.json')
            no_overlap=read(self.output/'retrieval'/f'qwen-structure-{budget}-0-dev-{selected["main_weight"]}.json')
            if with_overlap['recall_at_5'] < no_overlap['recall_at_5']+.02 or with_overlap['precision_at_3'] < no_overlap['precision_at_3']:
                selected['main_overlap']=0
                selected['overlap_confirmation']='最终混合检索无清楚收益，选择0'
        write(self.output/'selection.json',selected)


    async def intent_latency(self):
        """另测真实recognize路径，准备成本、缓存启动和热请求分别保留。"""
        path=self.output/"latency_templates.json"
        selection=read(self.output/"selection.json")
        threshold=selection["intent_threshold"]
        cases=[c for c in self.intent_data["cases"] if c["split"]=="test"][:4]
        cold=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.embedding,
            confidence_threshold=threshold,template_cache_path=str(path),template_wait_s=3)
        before=dict(self.embedding.ledger)
        started=time.perf_counter()
        value=await cold.recognize(cases[0]["message"])
        cold_request_ms=(time.perf_counter()-started)*1000
        prep_started=started
        await cold._template_task
        preparation_ms=(time.perf_counter()-prep_started)*1000
        cold_usage={k:self.embedding.ledger[k]-before[k] for k in ["http","encoded_texts","llm_calls"]}
        restarted=IntentRecognizer("experiment",model=self.model,gateway=self.gateway,embedding=self.embedding,
            confidence_threshold=threshold,template_cache_path=str(path),template_wait_s=3)
        before=dict(self.embedding.ledger)
        started=time.perf_counter()
        await restarted._load_template_embeddings()
        hit_start_ms=(time.perf_counter()-started)*1000
        hit_template_texts=self.embedding.ledger["encoded_texts"]-before["encoded_texts"]
        rows=[]
        for case in cases[1:]:
            before=dict(self.embedding.ledger)
            value=await restarted.recognize(case["message"])
            rows.append({"id":case["id"],"wall_ms":value.latency_ms,"intent":value.intent.value,"scores":value.source_scores,
                         "usage":{k:self.embedding.ledger[k]-before[k] for k in ["http","encoded_texts","llm_calls"]}})
        write(self.output/"intent_latency.json",{"status":"measured","cold_first_request_ms":cold_request_ms,
            "cold_preparation_until_ready_ms":preparation_ms,"cold_total_usage":cold_usage,"cache_hit_startup_ms":hit_start_ms,
            "cache_hit_template_texts":hit_template_texts,"hot_requests":rows,"latency_includes_experiment_throttle":True,
            "note":"首次请求有界等待后可按85/15降级；后台模板准备成本另列，没有消失。"})
        await cold.close()
        await restarted.close()


    async def answers(self, backup_only=False, byte_only=False):
        if self.gateway is None:
            raise RuntimeError('chat_credentials_missing')
        selection=read(self.output/'selection.json')
        specs=[('qwen',512,0,.5),('qwen',selection['main_budget'],selection['main_overlap'],selection['main_weight']),('minilm',240,0,selection['backup_weight'])]
        if backup_only:
            specs=specs[-1:]
        if byte_only:
            specs=[('qwen_byte',512,0,.5)]
        os.environ['RAG_RERANK_MAX_CHARS']='12000'
        questions={q['id']:q for q in self.rag['questions']}
        semaphore=asyncio.Semaphore(3)
        async def measure(name,encoder,qid,items):
            async with semaphore:
                path=self.output/'answers'/f'{name}-{qid}.json'
                self.reuse_backup_answer(path,encoder)
                previous=read(path,{})
                if previous.get('status')=='measured':
                    return
                if previous:
                    write(self.output/'failed_answers'/f"{path.stem}-{datetime.now().strftime('%H%M%S%f')}.json",previous)
                q=questions[qid]
                record=previous if previous.get('answer_status')=='measured' else {}
                try:
                    if not record:
                        manager=MCPToolManager('experiment',model=self.model,gateway=self.gateway)
                        ranked=await manager._rerank(q['question'],items,5)
                        chosen=[]
                        remaining=6000
                        for item in ranked:
                            block=f"[{len(chosen)+1}] {item['title']} / {item['section_path']}\n{item['content']}"
                            if len(block)<=remaining:
                                chosen.append(block)
                                remaining-=len(block)+2
                        context='\n\n'.join(chosen)
                        record={'question':q,'retrieved':items,'reranked':ranked,'context':context,'context_budget_chars':6000,
                            'rerank_input_budget_chars':12000,'rerank_status':'measured' if manager.rerank_stats['llm_success'] else 'failed_fallback',
                            'evaluation':'real_llm_audit_not_human','model':self.model}
                        answer=await self.gateway.complete(LLMRequest(self.model,'仅依据所给知识回答并引用编号，完整说明版本渠道、条件和例外；没有标准时明确材料未规定，不凭常识填数。',
                            messages=[{'role':'user','content':f"问题：{q['question']}\n证据：\n{context}"}],max_tokens=4096,temperature=0))
                        record.update(answer=answer.text,answer_status='measured')
                        write(path,record)
                    audit=await self.gateway.complete(LLMRequest(self.model,'审核回答是否由原文支持。仅输出JSON。不得把检索非空算作无答案编造。',
                        messages=[{'role':'user','content':json.dumps({'question':q,'answer':record['answer'],'context':record['context'],
                            'output':{'facts_supported':'布尔','conditions_complete':'布尔','unanswerable_fabricated':'布尔','unsupported_claims':[], 'citation_errors':[], 'explanation':'依据'}},ensure_ascii=False)}],max_tokens=4096,temperature=0))
                    record['audit_raw']=audit.text
                    record['audit']=json.loads(audit.text[audit.text.find('{'):audit.text.rfind('}')+1])
                    if any(type(record['audit'].get(k)) is not bool for k in ['facts_supported','conditions_complete','unanswerable_fabricated']) or any(not isinstance(record['audit'].get(k),list) for k in ['unsupported_claims','citation_errors']):
                        raise ValueError('invalid_answer_audit_contract')
                    record['status']='measured'
                except Exception as ex:
                    record.update(status='failed',reason=KnowledgeBase._reason(ex))
                    if 'authorized_llm_budget_exhausted' in str(ex):
                        record.update(status='blocked',reason='authorized_llm_budget_exhausted')
                        write(path,record)
                        raise
                write(path,record)
                print(f"answer {name} {qid} status={record['status']}",flush=True)
        for encoder,budget,overlap,weight in dict.fromkeys(specs):
            name=f'{encoder}-structure-{budget}-{overlap}-test-{weight}'
            result=read(self.output/'retrieval'/f'{name}.json')
            if not result:
                raise RuntimeError('formal_retrieval_missing')
            rows={r['id']:r for r in result['rows']}
            await asyncio.gather(*(measure(name,encoder,qid,rows[qid]['retrieved']) for qid in self.rag['answer_ids']))
        if any(read(p).get('status')!='measured' for p in (self.output/'answers').glob('*.json')):
            raise RuntimeError('answer_or_audit_failed')
    async def production(self):
        previous=read(self.output/"production.json")
        if previous and previous.get("status")!="measured":
            write(self.output/"failed_production"/f"{datetime.now().strftime('%H%M%S%f')}.json",previous)
        selection=read(self.output/"selection.json")
        if not selection:
            raise RuntimeError("selection_missing")
        self.prepare_mini()
        # 复用真实文档向量，但这20次查询实际调用Qwen并走Chroma/工具。
        original_embed=self.embedding.embed
        def production_embed(texts,task,titles=None):
            return self.embedding.cached_embed(texts,task,titles) if task=="RETRIEVAL_DOCUMENT" else original_embed(texts,task,titles)
        self.embedding.embed=production_embed
        path=self.output/"production-index"
        client=chromadb.PersistentClient(path=str(path),settings=chromadb.Settings(anonymized_telemetry=False))
        os.environ["RAG_CHUNK_TOKENS"]=str(selection["main_budget"])
        os.environ["RAG_OVERLAP_TOKENS"]=str(selection["main_overlap"])
        os.environ["RAG_MAIN_VECTOR_WEIGHT"]=str(selection["main_weight"])
        os.environ["RAG_BACKUP_VECTOR_WEIGHT"]=str(selection["backup_weight"])
        kb=KnowledgeBase(chroma_path=str(path),embedding=self.embedding,client=client,backup_function=self.mini,backup_counter=self.mini_count,
                         load_defaults=False,collection_prefix="v2production")
        manager=MCPToolManager("experiment",model=self.model or "missing",gateway=self.gateway)
        manager.register(Tool("knowledge_search","knowledge",kb.search_handler,{},cache_ttl=300))
        kb.on_change=manager.invalidate_cache
        rows=[]
        try:
            imported=await asyncio.to_thread(kb.import_documents,self.corpus)
            if imported["status"]!="success":
                raise RuntimeError("production_import_incomplete")
            # 文档批量限速留给下一批的等待在在线测量之前结清。
            delay=max(0,self.embedding._next_request-time.monotonic())
            if delay:
                await asyncio.sleep(delay)
                self.embedding.usage["throttle_sleep_ms"]+=delay*1000
            questions=[q for q in self.rag["questions"] if q["split"]=="test"][:20]
            for q in questions:
                started=time.perf_counter()
                result=await manager.call("knowledge_search",{"query":q["question"],"top_k":5})
                if not result.success:
                    raise RuntimeError("production_tool_failed")
                row=self.evaluate(q,result.data)
                row.update(route=result.data[0]["index_route"] if result.data else "empty",wall_ms=(time.perf_counter()-started)*1000,cached=result.cached)
                rows.append(row)
                write(self.output/"production.json",{"status":"running","rows":rows})
            def failure(*args,**kwargs):
                raise EmbeddingError("injected_provider_failure")
            self.embedding.embed=failure
            query=questions[0]["question"]
            fallback=await manager.call("knowledge_search",{"query":query},use_cache=False)
            self.embedding.embed=production_embed
            recovery=await manager.call("knowledge_search",{"query":query},use_cache=False)
            self.embedding.embed=failure
            changed={"title":"故障恢复测试","source_id":"v2-recovery","content":"仅用于集成验收：旧规则七天受理。","format":"md"}
            first=await asyncio.to_thread(kb.import_documents,[changed])
            changed["content"]="仅用于集成验收：新规则九天受理，必须提供订单号。"
            updated=await asyncio.to_thread(kb.import_documents,[changed])
            self.embedding.embed=production_embed
            pending=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            kb.close()
            kb=KnowledgeBase(chroma_path=str(path),embedding=self.embedding,client=client,backup_function=self.mini,backup_counter=self.mini_count,load_defaults=False,collection_prefix="v2production",on_change=manager.invalidate_cache)
            manager._tools["knowledge_search"].handler=kb.search_handler
            restart_pending=kb.stats()["pending_main"]
            repaired=await asyncio.to_thread(kb.repair_main)
            new_main=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            await asyncio.to_thread(kb.delete_document,"v2-recovery")
            deleted=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            self.embedding.embed=failure
            deleted_backup=await manager.call("knowledge_search",{"query":"故障恢复测试九天订单号"},use_cache=False)
            self.embedding.embed=production_embed
            checks={"fallback_real_body":fallback.success and fallback.degraded and bool(fallback.data),"recovered_main":recovery.success and all(i["index_route"]=="main" for i in recovery.data),
                    "degraded_import":first["status"]=="degraded" and updated["status"]=="degraded","pending_stays_backup":pending.success and pending.degraded,
                    "update_visible":any("九天" in i["content"] for i in pending.data),"old_absent":not any("七天受理" in i["content"] for i in pending.data),
                    "restart_pending":restart_pending==1,"repair_complete":repaired["main_complete"],"new_main":any(i["source_id"]=="v2-recovery" and "九天" in i["content"] and i["index_route"]=="main" for i in new_main.data),
                    "deleted_both":not any(i["source_id"]=="v2-recovery" for i in deleted.data+deleted_backup.data)}
            # 真实MiniLM + 延迟注入，预算检查独立标记。
            def delayed(*args,**kwargs):
                with self.embedding.query_budget(.1):
                    time.sleep(.15)
                    self.embedding._remaining()
            self.embedding.embed=delayed
            started=time.perf_counter()
            slow=await manager.call("knowledge_search",{"query":query},use_cache=False)
            slow_ms=(time.perf_counter()-started)*1000
            checks["injected_delay_returns_minilm"]=slow.success and slow.degraded and bool(slow.data) and slow_ms<30000
            self.embedding.embed=production_embed
            rewrite=[]
            if self.gateway:
                for q in questions[:2]:
                    value=await manager.search_with_rewrite("knowledge_search",q["question"],top_k=5)
                    rewrite.append({"id":q["id"],"success":value.success,"results":value.data,"degraded":value.degraded})
            write(self.output/"production.json",{"status":"measured" if all(checks.values()) else "failed","rows":rows,"checks":checks,"stats":kb.stats(),
                  "fault":"injected_provider_failure_and_delay; real_minilm_and_chroma","slow_fault_wall_ms":slow_ms,"rewrite":rewrite,
                  "deployment_boundaries":"Persistent Chroma verified; HTTP Chroma/Redis未启动完整部署"})
            if not all(checks.values()):
                raise RuntimeError("production_checks_failed")
        finally:
            kb.close()
            self.embedding.embed=original_embed


    async def execute(self,stage):
        if stage=="validate":
            self.validate()
        elif stage=="smoke":
            self.retrieval("qwen","structure",512,0,"dev",["vector","bm25","0.5"],3)
            self.retrieval("minilm","structure",240,0,"dev",["vector","bm25","0.5"],3)
        elif stage=="chunking":
            for kind in ["recursive","structure"]:
                for budget in [256,512,768]:
                    self.retrieval("qwen",kind,budget,0,"dev",["vector"])
        elif stage=="overlap":
            budget=self.choose_chunk()
            for overlap in [32,64]:
                self.retrieval("qwen","structure",budget,overlap,"dev",["vector"])
        elif stage=="fusion":
            budget=self.choose_chunk()
            overlap=self.choose_overlap(budget)
            self.retrieval("qwen","structure",512,0,"dev",["vector","bm25","0.5"])
            self.retrieval("qwen","structure",budget,overlap,"dev",["vector","bm25","0.25","0.5","0.75"])
            if overlap:
                self.retrieval("qwen","structure",budget,0,"dev",["0.25","0.5","0.75"])
            self.retrieval("minilm","structure",240,0,"dev",["vector","bm25","0.1","0.25","0.5"])
        elif stage=="intent_dev":
            await self.intent("dev")
        elif stage=="select":
            self.select()
        elif stage=="test":
            selection=read(self.output/"selection.json")
            if not selection:
                raise RuntimeError("selection_missing")
            self.retrieval("qwen","structure",512,0,"test",["0.5"])
            self.retrieval("qwen","structure",selection["main_budget"],selection["main_overlap"],"test",["vector","bm25",str(selection["main_weight"])])
            if selection["main_overlap"]:
                self.retrieval("qwen","structure",selection["main_budget"],0,"test",[str(selection["main_weight"])])
            self.retrieval("qwen","recursive",selection["recursive_control_budget"],0,"test",["vector"])
            self.retrieval("minilm","structure",240,0,"test",["vector","bm25","0.5",str(selection["backup_weight"])])
        elif stage=="intent_test":
            await self.intent("test")
            await self.intent_latency()
        elif stage=="answers":
            await self.answers()
        elif stage=="production":
            await self.production()
        elif stage=="application":
            await self.application()
        elif stage=='byte_baseline':
            self.retrieval('qwen_byte','structure',512,0,'test',['vector','bm25','0.5'])
            await self.answers(byte_only=True)


    async def run(self):
        stages=STAGES if self.args.stage=="all" else [self.args.stage]
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
                    "test":["select"], "intent_test":["select"], "answers":["test"], "production":["select","test"], "application":["production"], 'byte_baseline':['select','test']}
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
            result["api_usage"]=dict(self.embedding.ledger)
            write(self.output/"attempts"/f"{stage}-{result['attempt']}.json",result)
            write(self.output/"manifest.json",self.manifest)
            self.embedding.flush()
            print(f"stage={stage} finished status={result['status']} http={self.embedding.ledger['http']} encoded_texts={self.embedding.ledger['encoded_texts']} llm={self.embedding.ledger['llm_calls']}",flush=True)
        self.embedding.close()
        self.embedding.db.close()
        if self.gateway and hasattr(self.gateway.gateway,"close"):
            await self.gateway.gateway.close()
        return all(self.manifest["stages"].get(stage,{}).get("status")=="measured" for stage in stages)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=STAGES+['all','byte_baseline'],default='all')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=ROOT/'artifacts/rag-optimization-v2/20261009-v2-01')
    parser.add_argument('--rag-data',type=Path,default=ROOT/'data/eval/rag_optimization_v2.json')
    parser.add_argument('--intent-data',type=Path,default=ROOT/'data/eval/intent_optimization_v2.json')
    parser.add_argument('--api-interval',type=float,default=.05)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    raise SystemExit(0 if asyncio.run(Runner(args).run()) else 1)


if __name__=='__main__':
    main()
