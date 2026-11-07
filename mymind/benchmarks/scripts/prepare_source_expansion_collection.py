"""冻结已验收扩充清单、源码和后台命令；不连接服务器。"""
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.scripts.prepare_source_expansion import REL
from benchmarks.scripts.prepare_mainline_candidates import read,write


def main():
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve();out=root/REL
    assert read(out/'validation.json')['status']=='passed'
    if (out/'configuration.json').exists():raise ValueError('不覆盖固定配置')
    base=root/'runs/success-pool-v1';template=base/'expression-pair-v1'
    cfg=read(base/'teacher-mainline-v1/configuration.json')
    ids=[m['task_id'] for m in read(out/'metadata/task_manifest.json')]
    cfg.update(dataset_mode='source_expansion',frozen_dir=REL,tasks=REL+'/public/candidates.json',
        standards=REL+'/private/standards.json',policy_support=REL+'/policy/policy_support.json',
        output_dir='runs/success-pool-v1/teacher-source-expansion-v1',task_ids=ids,pilot_task_ids=ids,
        schedule=[dict(sequence=i+1,task_id=t) for i,t in enumerate(ids)],second_pass_order=ids,
        standards_revision='success-pool-evidence-v5',candidates_per_task=2,
        authorization_status='仅具体提案；尚未授权部署或真实采集；历史授权和余额不适用',
        baseline_configuration=REL+'/baseline_configuration.json',source_manifest=REL+'/source_files.txt')
    n=len(ids)*2
    caps=dict(http=dict(teacher=4*n,rerank=3*n,embedding=3*n),input_upper=2721152*n,
              output_cap=134144*n,usd=3,cny=.2,wall_seconds=21600)
    cfg['collection_limits']={k:v for k,v in cfg['collection_limits'].items() if k in {'request','pricing'}}
    cfg['collection_limits']['main']=caps
    cfg['collection_limits']['total']={k:v for k,v in caps.items() if k!='wall_seconds'}
    write(out/'configuration.json',cfg)
    shutil.copyfile(base/'teacher-mainline-v1/configuration.json',out/'baseline_configuration.json')
    names=[s for s in (template/'source_files.txt').read_text(encoding='utf-8').splitlines() if s!='mymind/rl/teacher_expression_pair.py']
    names.append('mymind/rl/teacher_source_expansion.py')
    (out/'source_files.txt').write_text('\n'.join(names)+'\n',encoding='utf-8')
    for name in names:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
    for name in ('prepare_source_expansion.py','verify_source_expansion.py','prepare_source_expansion_collection.py'):
        dest=out/'construction_source'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(Path(__file__).parent/name,dest)
    for name in ('run.sh','preflight.sh','start.sh','supervise.py','deploy.py'):
        text=(template/name).read_text(encoding='utf-8')
        text=text.replace('teacher-expression-pair-v1','teacher-source-expansion-v1').replace('expression-pair-v1','source-expansion-v1')
        text=text.replace('teacher_expression_pair','teacher_source_expansion')
        if name=='supervise.py':text=text.replace('7200','21600')
        (out/name).write_bytes(text.encode('utf-8'))
    resources=['configuration.json','baseline_configuration.json','source_files.txt','public/candidates.json',
        'private/standards.json','metadata/task_manifest.json','metadata/source_families.json','metadata/source_eligibility.json',
        'metadata/new_events.json','policy/knowledge.md','policy/policy_support.json','run.sh','preflight.sh','start.sh','supervise.py','validation.json']
    write(out/'deployment_manifest.json',
        [dict(source=REL+'/source_snapshot/'+s,destination='src/source-expansion-v1/'+s) for s in names]+
        [dict(source=REL+'/source_snapshot/'+s,destination=REL+'/source_snapshot/'+s) for s in names]+
        [dict(source=REL+'/'+s,destination=REL+'/'+s) for s in resources])
    from rl.teacher_source_expansion import preflight
    preflight(root,cfg)
    print(f'{len(ids)} tasks, {n} maximum attempts, {len(names)} runtime files, {len(read(out/"deployment_manifest.json"))} deployment entries; no deployment')


if __name__=='__main__':main()
