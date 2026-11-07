"""Offline collection integration checks; references are test fixtures only."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path

import httpx
import pytest

from agents.customer_policy.generation import Generation, DeepSeekGeneration
from agents.customer_policy.protocol import SearchObservation
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support
from rl.teacher_budget import CollectionBudget, CollectionStopped
from rl.teacher_collection import collect, read

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'runs/balanced-policy-v1/teacher-preparation'
if not PREP.exists():
    pytest.skip('需要单独保留的 teacher-preparation 实验原件', allow_module_level=True)
CONFIG = read(PREP/'collection.json')
FROZEN = ROOT/CONFIG['frozen_dir']


class FakeGeneration:
    input_token_upper_bound = DeepSeekGeneration.input_token_upper_bound

    def __init__(self, reference, fail=False, network=False):
        self.config = {}
        self.reference, self.fail, self.network = reference, fail, network
        self.inputs = []

    async def generate(self, messages, tools_enabled=True, max_new_tokens=None):
        self.inputs.append(deepcopy(messages))
        assert max_new_tokens <= 3072
        if self.network:
            raise ConnectionError('simulated network failure; actual HTTP=0')
        if not any(m['role']=='tool' for m in messages):
            calls=[dict(id='fake',type='function',function=dict(name='search',arguments={'query':'公开客服政策'}))]
            return Generation('',calls,{'role':'assistant','content':'','tool_calls':calls},{'simulation':True})
        text = '{}' if self.fail else json.dumps(self.reference['result'],ensure_ascii=False)
        return Generation(text,[],{'role':'assistant','content':text},{'simulation':True,'test_tokens':10,'actual_fee':0})


class FakeRetrieval:
    def __init__(self, reference): self.reference=reference
    async def search(self, query, number):
        assert query=='公开客服政策'
        return SearchObservation.model_validate(self.reference['trajectory']['searches'][0]['observation']), {'simulation':True}


def context(tmp_path):
    c=deepcopy(CONFIG)
    tasks=load_public_tasks(FROZEN/'public/train.json')
    standards=load_private_standards(FROZEN/'private/train.json')
    clauses=load_policy_support(FROZEN/'policy/policy_support.json')
    tid=c['pilot_task_ids'][0]
    c.update(task_ids=[tid],pilot_task_ids=[tid],collection_stage='pilot')
    ref=next(r for r in read(FROZEN/'acceptance_only/references/train.json') if r['task_id']==tid)
    b=CollectionBudget(tmp_path,c['collection_limits'],'pilot')
    return c,tasks,standards,clauses,ref,b


def test_collection_raw_selection_and_resume(tmp_path, monkeypatch):
    c,t,s,clauses,ref,b=context(tmp_path)
    from rl import teacher_collection as module
    original=module.run_task
    calls=[]
    async def alternating(task,generation,environment,**kwargs):
        generation.fail=not calls
        calls.append(task.task_id)
        return await original(task,generation,environment,**kwargs)
    monkeypatch.setattr(module,'run_task',alternating)
    g=FakeGeneration(ref)
    summary=asyncio.run(collect(c,tmp_path,t,s,clauses,g,FakeRetrieval(ref),b,read(FROZEN/'metadata/task_manifest.json')))
    assert len(calls)==2 and summary['accepted'][0]['candidate']==2
    first=(tmp_path/f'{ref["task_id"]}-candidate-1.json').read_bytes()
    assert not read(tmp_path/f'{ref["task_id"]}-candidate-1.json')['grade']['complete_success']
    assert summary['missing_by_stratum']['A01']==14 and len(summary['missing_task_ids'])==359
    assert g.inputs[0]==t[ref['task_id']].messages()
    for messages in g.inputs:
        assert not any(word in json.dumps(messages) for word in ['expectations','severe_items','expected_disposition','acceptance_only','subtype_id'])
    # No private metadata is accepted by the generation or retrieval interface.
    again=asyncio.run(collect(c,tmp_path,t,s,clauses,g,FakeRetrieval(ref),b,read(FROZEN/'metadata/task_manifest.json')))
    assert len(calls)==2 and again['accepted']==summary['accepted']
    assert first==(tmp_path/f'{ref["task_id"]}-candidate-1.json').read_bytes()


def test_first_success_and_network_stop(tmp_path):
    c,t,s,clauses,ref,b=context(tmp_path)
    g=FakeGeneration(ref)
    summary=asyncio.run(collect(c,tmp_path,t,s,clauses,g,FakeRetrieval(ref),b,[]))
    assert len(summary['attempts'])==1 and summary['accepted'][0]['candidate']==1
    other=tmp_path/'network'; other.mkdir()
    b=CollectionBudget(other,c['collection_limits'],'pilot')
    summary=asyncio.run(collect(c,other,t,s,clauses,FakeGeneration(ref,network=True),FakeRetrieval(ref),b,[]))
    assert len(summary['attempts'])==1 and not summary['accepted'] and summary['stopped']
    assert not (other/f'{ref["task_id"]}-candidate-2.json').exists()


def test_interrupted_candidate_consumed(tmp_path):
    c,t,s,clauses,ref,b=context(tmp_path)
    g=FakeGeneration(ref)
    async def interrupt(*args,**kwargs): raise KeyboardInterrupt()
    g.generate=interrupt
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(collect(c,tmp_path,t,s,clauses,g,FakeRetrieval(ref),b,[]))
    partial=tmp_path/f'{ref["task_id"]}-candidate-1.partial.json'
    before=partial.read_bytes()
    summary=asyncio.run(collect(c,tmp_path,t,s,clauses,FakeGeneration(ref),FakeRetrieval(ref),b,[]))
    assert summary['attempts'][0]['runtime_error']['kind']=='interrupted'
    assert partial.read_bytes()==before and len(summary['attempts'])==1


@pytest.mark.parametrize('limit', ['http','input_upper','output_cap','usd','cny','wall_seconds','request'])
def test_budget_limits(tmp_path,limit):
    caps=deepcopy(CONFIG['collection_limits'])
    if limit=='http': caps['pilot']['http']['teacher']=0
    elif limit=='request': caps['request']['teacher']['input']=1
    else: caps['pilot'][limit]=0
    b=CollectionBudget(tmp_path,caps,'pilot')
    with pytest.raises(CollectionStopped):
        b.reserve('embedding', {'input': ['测试值']}) if limit == 'cny' else b.reserve('teacher',{'max_tokens':20})
    assert b.state['requests']==[] and b.state['stop_reason']


def test_http_hooks_and_reserved_resume(tmp_path):
    b=CollectionBudget(tmp_path,CONFIG['collection_limits'],'pilot')
    async def exercise():
        for service in ['teacher','rerank']:
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200,json={'usage':{'prompt_tokens':12,'completion_tokens':5,'total_tokens':17}})),event_hooks=b.hooks(service)) as client:
                await client.post('https://offline.invalid',json={'max_tokens':20})
    asyncio.run(exercise())
    assert all(r['accounting_state']=='settled' for r in b.state['requests'])
    b.reserve('embedding',{'input':['测试值']})  # 模拟中断；没有实际HTTP。
    recovered=CollectionBudget(tmp_path,CONFIG['collection_limits'],'remainder')
    assert len(recovered.state['requests'])==3 and recovered.state['requests'][2]['accounting_state']=='unknown'
    assert (tmp_path/'http-00002.json').exists()
    assert read(tmp_path/'http-request-00003.json') == {'service':'embedding','payload':{'input':['测试值']}}





def test_unresolved_embedding_stops_before_clients(tmp_path,monkeypatch):
    from rl import teacher_collection as module
    c,t,s,clauses,ref,b=context(tmp_path)
    c['embedding']['base_url'] = None
    monkeypatch.setattr(module,'DeepSeekGeneration',lambda *a,**k: pytest.fail('must not instantiate client'))
    with pytest.raises(ValueError,match='Embedding'):
        asyncio.run(module.collect_real(ROOT,c,tmp_path,t,s,clauses))
