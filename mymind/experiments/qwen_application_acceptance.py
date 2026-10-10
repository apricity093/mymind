"""真实 FastAPI lifespan、业务默认语料、Agent 工具、HTTP Chroma 与 Redis 验收。"""
import asyncio
import os
import statistics
import threading
import time
import uuid
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit

import chromadb
import httpx
import redis

from core.embedding import EmbeddingError
from experiments.optimization_v2 import read, write
from mcp.knowledge_base import KnowledgeBase as Baseline


async def run(runner):
    from core import embedding as embedding_module, llm_gateway
    from api import main as api

    output=runner.output/'application.json'
    previous=read(output)
    if previous and previous.get('status')!='measured':
        write(runner.output/'failed_application'/f'application-{time.time_ns()}.json',previous)
    client=chromadb.HttpClient(host=os.getenv('CHROMA_HOST','localhost'),port=int(os.getenv('CHROMA_PORT','8001')),
                             settings=chromadb.Settings(anonymized_telemetry=False))
    client.heartbeat()
    initial={c.name:c.count() for c in client.list_collections()}
    defaults=[]
    Baseline._load_default_docs(SimpleNamespace(add_documents=lambda docs:defaults.extend(docs)))
    if 'knowledge_base' in initial:
        legacy=client.get_collection('knowledge_base').get(include=['documents','metadatas'])
        allowed={(d['title'],d['content']) for d in defaults}
        if any(((meta or {}).get('title'),text) not in allowed for text,meta in zip(legacy['documents'],legacy['metadatas'])):
            write(output,{'status':'blocked','reason':'实际业务库包含非内置语料，未经授权不能外发私人数据'})
            raise RuntimeError('private_business_corpus_not_authorized')

    prefix=f'qwen_acceptance_{uuid.uuid4().hex[:8]}'
    parts=urlsplit(os.getenv('REDIS_URL','redis://:mymind123@localhost:6379/0'))
    redis_url=urlunsplit(parts._replace(path='/15'))
    cache_client=redis.from_url(redis_url,decode_responses=True)
    cache_client.ping()
    overrides={'REDIS_URL':redis_url,'CACHE_PREFIX':prefix+':cache','TRACE_KEY_PREFIX':prefix+':trace',
               'EPISODIC_COLLECTION':prefix+'_episodic','PROFILE_COLLECTION':prefix+'_profile',
               'INTENT_TEMPLATE_CACHE_PATH':str(runner.output/'application_templates.json'),
               'ALERT_WEBHOOK_URL':'','PROMETHEUS_PORT':'0','MONITOR_INTERVAL':'60'}
    previous_env={key:os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    original_factory=embedding_module.build_embedding
    original_gateway=llm_gateway.build_gateway
    original_embed=runner.embedding.embed
    embedding_module.build_embedding=lambda *args,**kwargs:runner.embedding
    llm_gateway.build_gateway=lambda *args,**kwargs:runner.gateway
    result={'status':'running','scope':'实际FastAPI lifespan及Agent，HTTP Chroma/Redis；内置公开业务语料；非外部部署压测',
            'initial_collections':initial,'checks':{},'api_results':[],'chat_results':[]}
    checks=result['checks']
    prior_failures=len(read(runner.output/'llm_failures.json',[]))
    created_sources=[]
    try:
        async with api.app.router.lifespan_context(api.app):
            await api._orchestrator._intent_recognizer._template_task
            kb=api._knowledge_base
            result['tool_payloads']=[]
            original_search=api._tool_manager._tools['knowledge_search'].handler
            async def capture_search(params,context):
                items=await original_search(params,context)
                result['tool_payloads'].append({'params':params,'results':items})
                return items
            api._tool_manager._tools['knowledge_search'].handler=capture_search
            # 主计数规则改变后必须显式补齐；默认库只包含上述公开内置文档。
            repaired=await asyncio.to_thread(kb.repair_main)
            checks['business_main_complete']=repaired['main_complete'] and kb.stats()['documents']==6
            checks['http_chroma']=kb._client.get_settings().chroma_api_impl=='chromadb.api.fastapi.FastAPI' and kb._client.heartbeat()>0
            result['business_stats']=kb.stats()
            health=[]
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://acceptance',timeout=120) as web:
                for path in ['/health','/monitor','/knowledge/stats']:
                    started=time.perf_counter()
                    response=await web.get(path)
                    result['api_results'].append({'path':path,'status':response.status_code,'body':response.json()})
                    health.append((time.perf_counter()-started)*1000)
                    checks[path]=response.status_code==200

                docs=[{'source_id':prefix+'-add','title':'公开并发验收A','content':'公开退款规则：九天受理，需要订单号。'}]
                upload=[{'source_id':prefix+'-upload','title':'公开并发验收B','content':'公开发票规则：需要抬头和税号。'},
                        {'source_id':prefix+'-upload2','title':'公开并发验收C','content':'公开配送规则：下午六点停止发货。'}]
                import json
                added,uploaded=await asyncio.gather(web.post('/knowledge/add',json={'documents':docs}),
                    web.post('/knowledge/upload',files={'file':('public.json',json.dumps(upload,ensure_ascii=False).encode('utf-8'),'application/json')}))
                result['concurrent_import']={'add':added.json(),'upload':uploaded.json()}
                created_sources=[d['source_id'] for d in docs+upload]
                checks['concurrent_request_outcomes']=added.status_code==uploaded.status_code==200 and added.json()['processed_chunks']==1 and uploaded.json()['processed_chunks']==2 and {d['source_id'] for d in added.json()['documents']}=={d['source_id'] for d in docs} and {d['source_id'] for d in uploaded.json()['documents']}=={d['source_id'] for d in upload}
                for source in created_sources:
                    response=await web.delete('/knowledge/documents/'+source)
                    checks['delete_'+source]=response.status_code==200
                created_sources=[]

                # 在实际远程计数开始前暂停导入线程，确认统计和事件循环仍响应。
                entered,released=threading.Event(),threading.Event()
                original_encode=runner.embedding._encode_batch
                def paused(batch,task):
                    if task=='TOKEN_COUNT' and any('监控慢导入验收' in text for text in batch):
                        entered.set()
                        if not released.wait(10):
                            raise RuntimeError('controlled_import_wait_expired')
                    return original_encode(batch,task)
                runner.embedding._encode_batch=paused
                slow_source=prefix+'-slow'
                pending=asyncio.create_task(web.post('/knowledge/add',json={'documents':[{'source_id':slow_source,'title':'监控慢导入验收','content':'公开验收规则：七天处理。批次编号：'+prefix}]}))
                created_sources.append(slow_source)
                try:
                    if not await asyncio.to_thread(entered.wait,5):
                        raise RuntimeError('controlled_remote_pause_not_reached')
                    started=time.perf_counter()
                    values=await asyncio.wait_for(asyncio.gather(web.get('/health'),web.get('/monitor'),web.get('/knowledge/stats')),3)
                    result['stats_during_paused_import_ms']=(time.perf_counter()-started)*1000
                    checks['stats_during_paused_remote_import']=all(v.status_code==200 for v in values)
                finally:
                    released.set()
                    await pending
                    runner.embedding._encode_batch=original_encode
                await web.delete('/knowledge/documents/'+slow_source)
                created_sources=[]

                query='请先检索知识库，告诉我购买后超过7天但不足30天，商品有质量问题，退款需要什么证据？'
                response=await web.post('/chat',json={'message':query,'user_id':prefix,'conv_id':prefix+'-healthy'})
                body=response.json()
                result['chat_results'].append({'fault':'none','status':response.status_code,'body':body})
                checks['agent_real_main_tool']=response.status_code==200 and body.get('knowledge_used') and body.get('knowledge_status')=='used'
                def delayed(*args,**kwargs):
                    with runner.embedding.query_budget(.1):
                        time.sleep(.15)
                        runner.embedding._remaining()
                runner.embedding.embed=delayed
                before_fault=len(result['tool_payloads'])
                response=await web.post('/chat',json={'message':'请先搜索知识库：出现500服务器错误，按知识库应该如何处理？','user_id':prefix,'conv_id':prefix+'-fault'})
                body=response.json()
                result['chat_results'].append({'fault':'injected_budget_delay_real_minilm','status':response.status_code,'body':body})
                checks['agent_delay_real_backup_tool']=response.status_code==200 and body.get('knowledge_used') and body.get('knowledge_status')=='degraded'
                checks['agent_delay_real_backup_body']=any(item.get('index_route')=='backup' and '500 服务器错误' in item['content']
                    for call in result['tool_payloads'][before_fault:] for item in call['results'])
                runner.embedding.embed=original_embed
                result['endpoint_wall_ms']=health
                result['final_business_stats']=kb.stats()
                checks['business_restored_six_docs']=kb.stats()['documents']==6 and kb.stats()['main_complete']
            if api._background_tasks:
                await asyncio.gather(*tuple(api._background_tasks),return_exceptions=True)
            for key in cache_client.scan_iter(match=f'*{prefix}*'):
                cache_client.delete(key)
            checks['isolated_redis_cleaned']=not list(cache_client.scan_iter(match=f'*{prefix}*'))
            checks['no_new_llm_provider_failures']=len(read(runner.output/'llm_failures.json',[]))==prior_failures
            checks['original_collections_preserved']=all(client.get_collection(name).count()==count for name,count in initial.items())
            result['status']='measured' if all(checks.values()) else 'failed'
            write(output,result)
            if not all(checks.values()):
                raise RuntimeError('application_acceptance_checks_failed')
    except Exception as ex:
        result['status']='blocked' if 'authorized' in str(ex) else 'failed'
        result['reason']=str(ex) if isinstance(ex,RuntimeError) else type(ex).__name__
        write(output,result)
        raise
    finally:
        for source in created_sources:
            if api._knowledge_base is not None:
                await asyncio.to_thread(api._knowledge_base.delete_document,source)
        runner.embedding.embed=original_embed
        embedding_module.build_embedding=original_factory
        llm_gateway.build_gateway=original_gateway
        for key,value in previous_env.items():
            if value is None:
                os.environ.pop(key,None)
            else:
                os.environ[key]=value
        for name in [prefix+'_episodic',prefix+'_profile']:
            if name in {c.name for c in client.list_collections()}:
                client.delete_collection(name)
        for key in cache_client.scan_iter(match=f'*{prefix}*'):
            cache_client.delete(key)
        cache_client.close()
