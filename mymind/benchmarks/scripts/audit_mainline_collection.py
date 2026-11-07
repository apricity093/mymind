"""Verify saved mainline originals and accounting without service calls."""
from collections import Counter
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    prep=root/'runs/success-pool-v1/mainline-candidates-v1'
    out=root/'runs/success-pool-v1/teacher-mainline-v1'
    sys.path.insert(0,str(out/'source/mymind'))
    from agents.customer_policy.protocol import VisibleTrajectory,parse_result
    from agents.customer_policy.teacher_length import input_token_upper_bound
    from rl.tasks import load_public_tasks,load_private_standards,load_policy_support
    from rl.grading import grade_result
    from rl.teacher_budget import CollectionBudget
    cfg=read(out/'configuration.json'); summary=read(out/'summary.json'); budget=read(out/'budget.json')
    assert cfg==read(prep/'configuration.json')
    names=(prep/'source_files.txt').read_text().splitlines()
    for name in names: assert (out/'source'/name).read_bytes()==(prep/'source_snapshot'/name).read_bytes(),name
    for key in ('tasks','standards','policy_support'):
        assert (out/'inputs'/key/Path(cfg[key]).name).read_bytes()==(root/cfg[key]).read_bytes(),key
    tasks=load_public_tasks(out/'inputs/tasks/candidates.json')
    standards=load_private_standards(out/'inputs/standards/standards.json')
    clauses=load_policy_support(out/'inputs/policy_support/policy_support.json')
    responses={}; requests={}; unknown=[]
    for i,e in enumerate(budget['requests'],1):
        request=read(out/f'http-request-{i:05d}.json'); requests[i]=request['payload']
        assert request['service']==e['service']
        path=out/f'http-{i:05d}.json'
        if not path.exists():
            assert e['accounting_state'] in {'unknown','in_flight'};unknown.append(i);continue
        response=read(path); responses[i]=response['response']
        assert response['service']==e['service'] and response['status']==e['http_status']
        assert response['response'].get('usage')==e.get('usage')
        cost,basis=CollectionBudget.usage_cost(SimpleNamespace(state={'pricing':budget['pricing']}),e,response['response'])
        assert cost==e['estimated_cost'] and basis==e['billing_basis']
        if e['service']=='teacher':
            p=requests[i]; assert p['model']=='deepseek-flash' and p['thinking']=={'type':'enabled'}
            assert p['reasoning_effort']=='high' and 'temperature' not in p
            assert p['max_tokens']==32768 and p['top_p']==1
    teacher_ids={body['id']:i for i,body in responses.items() if budget['requests'][i-1]['service']=='teacher' and 'id' in body}
    checked=[];used=[]
    stop_message=dict(role='system',content='本轮不再执行检索，请依据现有观察提交终止 JSON。')
    for attempt in summary['attempts']:
        row=read(out/attempt['file']);tid=row['task_id'];trajectory=VisibleTrajectory.model_validate(row['trajectory'])
        assert grade_result(standards[tid],row['raw_result'],trajectory,clauses).to_dict()==row['grade']==attempt['grade']
        expected=tasks[tid].messages();expected[0]['content']+='\n'+cfg['public_teacher_guidance'];search_index=0
        for g in row['generations']:
            if g['input_messages']!=expected:
                assert g['input_messages']==expected+[stop_message];expected.append(stop_message)
            i=teacher_ids[g['raw_response']['id']];used.append(i);payload=requests[i];e=budget['requests'][i-1]
            assert e['task_id']==tid and payload['messages']==g['input_messages']==expected
            message=responses[i]['choices'][0]['message']
            assert g['raw_text']==(message.get('content') or '')
            assert g['assistant_message'].get('reasoning_content')==message.get('reasoning_content')
            assert input_token_upper_bound(payload['messages'],payload.get('tools'))==e['input_upper']
            assert g['output_token_budget_used']==responses[i]['usage']['completion_tokens']
            expected.append(g['assistant_message'])
            if responses[i]['choices'][0]['finish_reason']=='length':break
            for call in g['assistant_message'].get('tool_calls') or []:
                if search_index>=len(trajectory.searches):break
                turn=trajectory.searches[search_index];args=call['function']['arguments']
                if isinstance(args,str):args=json.loads(args)
                assert args==turn.action.model_dump()
                expected.append(dict(role='tool',tool_call_id=call['id'],name='search',content=turn.observation.tool_content()));search_index+=1
        assert expected==row['messages']
        if row['generations']:
            text=row['generations'][-1]['raw_text']
            try: assert parse_result(text).model_dump()==row['raw_result']
            except ValueError: assert text==row['raw_result']
        checked.append(dict(task_id=tid,original_score_reproduced=True,raw_reply_unchanged=True,exact_message_chain=True,
                            searches=len(trajectory.searches),score_pass=row['score_pass'],process_errors=row['process_errors']))
    assert summary['attempted_task_ids']==cfg['task_ids'][:len(summary['attempted_task_ids'])]
    assert len(summary['attempted_task_ids'])==len(set(summary['attempted_task_ids']))
    assert sorted(used)==sorted(teacher_ids.values())
    assert all(e['accounting_state']=='settled' for e in budget['requests'])
    caps=cfg['collection_limits']['main']
    assert all(budget['totals']['http'][k]<=v for k,v in caps['http'].items())
    assert all(budget['totals']['occupied'][k]<=caps[k] for k in ('usd','cny'))
    assert sum(e['input_upper'] for e in budget['requests'])<=caps['input_upper']
    assert sum(e['output_cap'] for e in budget['requests'])<=caps['output_cap']
    assert sum(c['searches'] for c in checked)<=108
    token_counts={service:{key:sum((e.get('usage') or {}).get(key,0) for e in budget['requests'] if e['service']==service) for key in ['prompt_tokens','completion_tokens','total_tokens']} for service in ['teacher','rerank','embedding']}
    audit=dict(configuration_and_inputs_equal=True,source_snapshot_files=len(names),requests=len(requests),responses=len(responses),unknown_responses=unknown,
               http=dict(Counter(e['service'] for e in budget['requests'])),tokens=token_counts,
               input_upper=sum(e['input_upper'] for e in budget['requests']),output_cap=sum(e['output_cap'] for e in budget['requests']),
               totals=budget['totals'],searches=sum(c['searches'] for c in checked),attempt_checks=checked,
               stopped=summary['stopped'],content_review='pending',real_calls_during_audit=0)
    (out/'evidence_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in audit.items() if k!='attempt_checks'},ensure_ascii=False))

if __name__=='__main__':main()
