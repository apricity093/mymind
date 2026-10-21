"""故障补测单列，首次评分及正常失败保留。"""
import asyncio
from copy import deepcopy
import json

import pytest

from experiments.agentic_rl import formal_evaluate as evaluation
from experiments.agentic_rl.runtime import write_json
from experiments.agentic_rl.stage7 import (check_inputs, processed_infrastructure_faults,
                                          select_checkpoint)
from tests.test_stage7_sft_recovery import job, saved, tasks, read, PREP

FAULT = "M6-P07-003"


@pytest.fixture
def resolved_job(job):
    root, cfg, output, tasks, standards, clauses, generator = job
    rows = [json.loads(line) for line in (PREP / 'formal/sft-validation-40/first_records.jsonl').read_text(encoding='utf-8').splitlines()][:43]
    assert len(rows) == 43 and rows[-1]['task_id'] == FAULT and rows[-1]['infrastructure_failure']
    (output / 'first_records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    frozen = {k:v for k,v in cfg.items() if k != 'resume_partial_task'}
    write_json(output / 'configuration.json', frozen)
    directory = 'diagnostic-40'
    target = root / directory
    target.mkdir()
    diagnostic = {**frozen,'run_kind':'diagnostic','task_ids':[FAULT], 'output_dir':directory,
        'timeout_s':900,'diagnostic_of':frozen['output_dir']}
    write_json(target / 'configuration.json',diagnostic)
    healthy = deepcopy(rows[-1]); healthy['infrastructure_failure'] = False
    healthy['grade']['reward'] = 1
    healthy['grade']['complete_success'] = True
    (target / 'first_records.jsonl').write_text(json.dumps(healthy,ensure_ascii=False)+'\n',encoding='utf-8')
    write_json(target / 'metrics.json',{'all_tasks_finished':True,'infrastructure_failures':0,
        'table_kind':'infrastructure_diagnostic','diagnostic_of':frozen['output_dir']})
    write_json(target / 'execution.json',{'exit_code':0})
    write_json(output / 'infrastructure_resolution.json',{'diagnostics':{FAULT:directory}})
    write_json(root / cfg['limits_file'],{'limits':{'student_trajectories':1000,'api_request_allowance':12000},
        'consumed':{'student_trajectories':123,'api_request_allowance':1476},'events':[]})
    return job, target, rows


def test_processed_fault_allows_only_37_unstarted_and_keeps_fault_score(resolved_job):
    (root,cfg,output,tasks,standards,clauses,generator),target,old = resolved_job
    prefix = (output / 'first_records.jsonl').read_bytes()
    asyncio.run(evaluation.evaluate(root,cfg,output,tasks,standards,clauses))
    assert len(generator.inputs) == 37
    assert (output / 'first_records.jsonl').read_bytes().startswith(prefix)
    rows = [json.loads(line) for line in (output / 'first_records.jsonl').read_text(encoding='utf-8').splitlines()]
    assert rows[42] == old[42] and rows[42]['infrastructure_failure'] and rows[42]['grade']['reward'] == 0
    metrics = read(output / 'metrics.json')
    assert metrics['all_tasks_finished'] and metrics['infrastructure_failures'] == 1
    assert metrics['unresolved_infrastructure_failures'] == 0 and metrics['processed_infrastructure_tasks'] == [FAULT]
    assert read(root / cfg['limits_file'])['consumed'] == {'student_trajectories':160,'api_request_allowance':1920}
    assert read(target / 'configuration.json')['timeout_s'] == 900
    assert read(output / 'configuration.json')['timeout_s'] == 3600


@pytest.mark.parametrize('change',['adapter','generation','task','fault','exit','source'])
def test_wrong_or_unfinished_diagnostic_does_not_release_fault(resolved_job,change):
    (root,cfg,output,*_),target,rows = resolved_job
    config = read(target / 'configuration.json')
    if change == 'adapter': config['adapter_path'] = 'another-adapter'
    if change == 'generation': config['generation']['max_new_tokens'] = 4096
    if change == 'task': config['task_ids'] = ['another-task']
    if change == 'source': config['diagnostic_of'] = 'another-first-table'
    write_json(target / 'configuration.json',config)
    if change == 'fault':
        data = json.loads((target / 'first_records.jsonl').read_text(encoding='utf-8'))
        data['infrastructure_failure'] = True
        (target / 'first_records.jsonl').write_text(json.dumps(data)+'\n',encoding='utf-8')
    if change == 'exit': write_json(target / 'execution.json',{'exit_code':1})
    with pytest.raises(ValueError): processed_infrastructure_faults(root,output,rows)


def test_new_fault_still_stops_after_a_processed_fault(resolved_job):
    (root,cfg,output,tasks,standards,clauses,generator),_,_ = resolved_job
    generator.failure = True
    with pytest.raises(RuntimeError,match='基础设施故障'):
        asyncio.run(evaluation.evaluate(root,cfg,output,tasks,standards,clauses))
    assert read(output / 'metrics.json')['finished_tasks'] == 44
    assert read(output / 'metrics.json')['infrastructure_failures'] == 2
    assert read(output / 'metrics.json')['unresolved_infrastructure_failures'] == 1


def test_base_counterpart_needs_real_same_task_trigger(resolved_job):
    (root,cfg,output,tasks,standards,*_),_,_ = resolved_job
    base = read(PREP / 'configs/formal-base.json')
    source = root / base['output_dir']; source.mkdir(parents=True)
    write_json(source / 'configuration.json',base)
    (source / 'first_records.jsonl').write_text(json.dumps({'task_id':FAULT,'infrastructure_failure':False})+'\n')
    diagnostic = {**base,'run_kind':'diagnostic','task_ids':[FAULT],'output_dir':'base-diagnostic',
        'timeout_s':900,'diagnostic_of':base['output_dir'],'diagnostic_trigger':cfg['output_dir']}
    check_inputs(root,diagnostic,tasks,standards,'evaluate')
    with pytest.raises(ValueError,match='正常低分'):
        check_inputs(root,{k:v for k,v in diagnostic.items() if k != 'diagnostic_trigger'},tasks,standards,'evaluate')


def test_selection_keeps_first_metrics_with_processed_fault(resolved_job):
    (root,cfg,output,*_),_,_ = resolved_job
    training = root / 'train'; training.mkdir()
    write_json(training / 'configuration.json',{'run_kind':'formal','run_purpose':'sft','baseline':cfg['baseline']})
    (training / 'updates-rank0.jsonl').write_text(json.dumps({'step':1,'parameter_delta_l1':1})+'\n')
    adapter = training / 'checkpoint-40'; adapter.mkdir()
    write_json(adapter / 'trainer_state.json',{'global_step':40})
    # 使用当前候选的真实adapter相对路径，诊断也必须匹配。
    original = read(output / 'configuration.json'); original['adapter_path'] = 'train/checkpoint-40'
    write_json(output / 'configuration.json',original)
    diagnostic_path = root / 'diagnostic-40/configuration.json'
    diagnostic = read(diagnostic_path); diagnostic['adapter_path'] = original['adapter_path']
    write_json(diagnostic_path,diagnostic)
    write_json(output / 'metrics.json',{'all_tasks_finished':True,'infrastructure_failures':1,
        'combination_equal_success_rate':.1,'combination_equal_severe_error_rate':.4})
    write_json(root / 'select.json',{'phase':'sft','training_output':'train','nodes':[{'step':40,'evaluation':cfg['output_dir']}],
                                  'selection_output':'selection.json'})
    result = select_checkpoint(root,root / 'select.json')
    assert result['selected_step'] == 40 and result['candidates'][0]['success'] == .1
    assert result['candidates'][0]['original_infrastructure_failures'] == 1
    metrics = read(output / 'metrics.json'); metrics['all_tasks_finished'] = False
    write_json(output / 'metrics.json',metrics)
    with pytest.raises(ValueError,match='完整'):
        select_checkpoint(root,root / 'select.json')
    metrics['all_tasks_finished'] = True; write_json(output / 'metrics.json',metrics)
    (output / 'infrastructure_resolution.json').unlink()
    with pytest.raises(ValueError,match='未处理'): select_checkpoint(root,root / 'select.json')
