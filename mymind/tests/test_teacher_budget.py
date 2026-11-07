"""Usage结算、金额预留与教师长度单位的离线契约。"""
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
import asyncio
import json

import httpx
import pytest

from rl import teacher_budget as module
from rl.teacher_budget import CollectionBudget, CollectionStopped, price_period
from agents.customer_policy.generation import DeepSeekGeneration, Generation, TOOLS
from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.runner import run_task
from agents.customer_policy.teacher_length import input_token_upper_bound
from tests.test_teacher_collection import CONFIG, FROZEN, FakeGeneration, FakeRetrieval, read
from rl.tasks import load_public_tasks


def stamp(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def budget_at(tmp_path, monkeypatch, when='2026-11-05T02:00:00'):
    now=[stamp(when)]
    monkeypatch.setattr(module,'time',SimpleNamespace(time=lambda:now[0]))
    return CollectionBudget(tmp_path,deepcopy(CONFIG['collection_limits']),'pilot'), now


def payload(service='teacher'):
    return {'model':'deepseek-flash','messages':[{'role':'user','content':'政策材料'*1000}], 'max_tokens':3072 if service=='teacher' else 1024}


def response(hit=500,miss=500,out=10):
    return {'model':'deepseek-flash','usage':{'prompt_tokens':hit+miss,'prompt_cache_hit_tokens':hit,
        'prompt_cache_miss_tokens':miss,'completion_tokens':out,'total_tokens':hit+miss+out}}


@pytest.mark.parametrize('when,multiplier',[('2026-11-05T02:00:00',1),('2026-11-05T12:00:00',.5)])
def test_cache_and_short_output_settle_replace_reservation(tmp_path,monkeypatch,when,multiplier):
    b,now=budget_at(tmp_path,monkeypatch,when)
    for hit in (0,500,800):
        i=b.reserve('teacher',payload())
        reserved=b.state['requests'][i]['reservation']['usd']
        b.finish(i,200,response(hit,1000-hit,10))
        e=b.state['requests'][i]
        expected=(hit*.006+(1000-hit)*.3+10*1.2)*multiplier/1e6
        assert e['estimated_cost']['usd']==pytest.approx(expected) and expected<reserved
        before=deepcopy(b.totals())
        b.finish(i,200,response(0,1000,3072))
        assert b.totals()==before
    totals=b.totals()
    assert totals['in_flight']['usd']==0 and totals['unknown']['usd']==0
    assert totals['occupied']['usd']==totals['settled']['usd']
    restored=CollectionBudget(tmp_path,CONFIG['collection_limits'],'pilot')
    assert restored.totals()==totals and restored.totals()['http']['teacher']==3


def test_inflight_unknown_failure_and_remaining(tmp_path,monkeypatch):
    b,now=budget_at(tmp_path,monkeypatch)
    i=b.reserve('teacher',payload()); j=b.reserve('teacher',payload())
    b.finish(i,200,{})
    assert b.totals()['unknown']['usd']>0 and b.totals()['in_flight']['usd']>0
    reserved=b.totals()['occupied']['usd']
    b=CollectionBudget(tmp_path,CONFIG['collection_limits'],'pilot')
    assert b.totals()['occupied']['usd']==reserved and b.totals()['in_flight']['usd']==0
    # 有可信usage的失败响应同样结算；HTTP次数不返还。
    b.finish(j,500,response())
    assert b.state['requests'][j]['accounting_state']=='settled'
    assert b.totals()['http']['teacher']==2
    assert b.totals()['authorization_remaining']['usd']==pytest.approx(2-b.totals()['occupied']['usd'])


def test_missing_cache_details_and_invalid_usage(tmp_path,monkeypatch):
    b,now=budget_at(tmp_path,monkeypatch)
    i=b.reserve('teacher',payload())
    b.finish(i,200,{'usage':{'prompt_tokens':1000,'completion_tokens':10}})
    e=b.state['requests'][i]
    assert e['billing_basis']['cache_basis']=='cache_details_missing_all_input_priced_as_miss'
    assert e['estimated_cost']['usd']==pytest.approx((1000*.3+10*1.2)/1e6)
    j=b.reserve('teacher',payload()); body=response();body['usage']['completion_tokens']=None
    b.finish(j,200,body)
    assert b.state['requests'][j]['accounting_state']=='unknown'


def test_no_dispatch_when_money_or_http_exhausted(tmp_path,monkeypatch):
    b,now=budget_at(tmp_path,monkeypatch)
    b.limits['pilot']['usd']=0
    calls=[]
    async def run():
        async with httpx.AsyncClient(event_hooks=b.hooks('teacher'), transport=httpx.MockTransport(
            lambda req: calls.append(req) or httpx.Response(200,json=response()))) as client:
            with pytest.raises(CollectionStopped): await client.post('https://offline.invalid',json=payload())
    asyncio.run(run())
    assert not calls and not b.state['requests']
    other=tmp_path/'http';other.mkdir()
    b=CollectionBudget(other,deepcopy(CONFIG['collection_limits']),'pilot')
    b.limits['pilot']['http']['teacher']=1
    i=b.reserve('teacher',payload());b.finish(i,200,response())
    with pytest.raises(CollectionStopped): b.reserve('teacher',payload())
    assert b.totals()['http']['teacher']==1


def test_price_boundary_and_weekend():
    period,basis=price_period(stamp('2026-11-05T00:59:59'),stamp('2026-11-05T01:00:01'))
    assert period=='peak' and basis=='boundary_peak_conservative'
    assert price_period(stamp('2026-11-07T02:00:00'),stamp('2026-11-07T02:01:00'))[0]=='off_peak'


def test_embedding_reserves_submitted_query_and_settles(tmp_path,monkeypatch):
    from core.embedding import QwenEmbedding, EmbeddingSettings
    b,now=budget_at(tmp_path,monkeypatch)
    requests=[]
    def respond(req):
        requests.append(json.loads(req.content))
        return httpx.Response(200,json={'data':[{'index':0,'embedding':[1.,0.]}],
            'usage':{'prompt_tokens':7,'total_tokens':7}})
    embedding=QwenEmbedding(EmbeddingSettings('qwen3.7-text-embedding-flash',2,'mock-only'),transport=httpx.MockTransport(respond))
    embedding.collection_budget=b
    embedding.embed(['公开退款政策'],'RETRIEVAL_QUERY')
    assert len(requests)==1 and requests[0]['input']==['公开退款政策']
    assert read(tmp_path/'http-request-00001.json') == {'service':'embedding','payload':requests[0]}
    e=b.state['requests'][0]
    assert e['input_upper']==len('公开退款政策'.encode('utf-8'))+16
    assert e['estimated_cost']['cny']==pytest.approx(7*.125/1e6)
    assert embedding.usage['count_requests']==0
    embedding.close()


def test_train_public_roundtrips_preserve_policy_and_output_budget(tmp_path):
    # 360公开train只查长度/往返；不重审私有标准、test或评分器。
    tasks=load_public_tasks(FROZEN/'public/train.json')
    references={r['task_id']:r for r in read(FROZEN/'acceptance_only/references/train.json')}
    class OutputUsageGeneration(FakeGeneration):
        async def generate(self,*args,**kwargs):
            generated=await super().generate(*args,**kwargs)
            generated.diagnostics['raw_response']={'usage':{'completion_tokens':30}}
            return generated
    async def run():
        stats=[]
        for tid,task in tasks.items():
            ref=references[tid]; g=OutputUsageGeneration(ref)
            row=await run_task(task,g,PolicyEnvironment(FakeRetrieval(ref)),budget=CONFIG['budget'])
            assert row['runtime_error'] is None
            assert len(row['generations'])==2 and row['trajectory']['searches'][0]['observation']['snippets']
            assert not row['tool_diagnostics'][0].get('observation_budget_applied')
            assert sum(x['output_token_budget_used'] for x in row['generations'])==60
            observed=[g.input_token_upper_bound(messages) for messages in g.inputs]
            assert max(observed)+3072 <= CONFIG['budget']['context_tokens']
            stats.append({'task_id':tid,'input_upper':observed,'output_usage_test_value':60})
        (tmp_path/'length_roundtrips.json').write_text(json.dumps({'simulation':True,'actual_calls':0,'rows':stats}),encoding='utf-8')
    asyncio.run(run())

def test_network_failure_becomes_unknown_without_returning_http(tmp_path,monkeypatch):
    b,now=budget_at(tmp_path,monkeypatch)
    async def run():
        def timeout(request): raise httpx.ReadTimeout('offline timeout',request=request)
        async with httpx.AsyncClient(event_hooks=b.hooks('teacher'),transport=httpx.MockTransport(timeout)) as client:
            with pytest.raises(httpx.ReadTimeout): await client.post('https://offline.invalid',json=payload())
    asyncio.run(run())
    b.mark_inflight_unknown('candidate_generation_failure')
    assert b.state['requests'][0]['accounting_state']=='unknown'
    assert b.totals()['unknown']['usd']==b.state['requests'][0]['reservation']['usd']
    assert b.totals()['http']['teacher']==1


def test_rerank_billing_and_reservation_ignores_wire_escaping(tmp_path,monkeypatch):
    b,now=budget_at(tmp_path,monkeypatch)
    data=payload('rerank');data['messages'][0]['content']='政策\n正文'*100
    i=b.reserve('rerank',data)
    b.finish(i,200,response(800,200,12))
    assert b.state['requests'][i]['estimated_cost']['usd']==pytest.approx((800*.006+200*.3+12*1.2)/1e6)
    assert b.state['requests'][i]['input_upper']==input_token_upper_bound(data['messages'])
    assert b.totals()['http']['rerank']==1
