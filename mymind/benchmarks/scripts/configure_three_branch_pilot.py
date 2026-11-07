"""Freeze the executable pilot configuration and deployment file list."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import shutil

REL=Path('runs/success-pool-v1/three-branch-pilot-v1')
def posix(p): return p.as_posix()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve(); out=root/REL
    baseline=read(root/'mymind/rl/configs/training.json')
    source=read(root/'runs/success-pool-v1/source-expansion-v1/configuration.json')
    manifest=read(out/'manifest.json')
    ids={s:[r['task_id'] for r in manifest if r['split']==s] for s in ['train','validation','test']}
    base={k:deepcopy(baseline[k]) for k in ['model_id','model_revision','model_dir','lora','grpo']}
    base.update(dataset_mode='three_branch_pilot',context_tokens=16384,turn_tokens=2048,seed=42,
        grading_revision='v17-policy-constant-equivalence',standards_revision='success-pool-evidence-v5',
        retrieval=source['retrieval'],policy_support=posix(REL/'policy/policy_support.json'),
        budget=dict(context_tokens=16384,completion_tokens=8192,turn_tokens=2048),
        source_manifest=posix(REL/'source_files.txt'))
    base['generation']=dict(model_id=base['model_id'],model_revision=base['model_revision'],
        temperature=0,top_p=1,max_new_tokens=2048,context_tokens=16384)
    def cfg(name,split):
        return {**deepcopy(base),'tasks':posix(REL/f'public/{split}.json'),
            'standards':posix(REL/f'private/{split}.json'),'task_ids':ids[split],
            'output_dir':posix(REL/f'output/{name}')}
    training=dict(max_steps=-1,num_train_epochs=3,per_device_train_batch_size=1,
        gradient_accumulation_steps=4,learning_rate=5e-6,lr_scheduler_type='constant',warmup_steps=0,
        weight_decay=0.,max_grad_norm=1.,save_steps=999,seed=42,disable_tqdm=True,dataloader_drop_last=False)
    sft=cfg('sft','train'); sft.update(training=training,checkpoint_steps=[3,6,9],expected_updates=9,
        student_data=posix(REL/'train/student_data.json'))
    write(out/'configs/sft.json',sft)
    pre=deepcopy(sft); pre['output_dir']=posix(REL/'output/gpu-preflight')
    pre['training'].update(max_steps=1,num_train_epochs=1,gradient_accumulation_steps=1)
    pre['checkpoint_steps']=[1]
    pre['expected_updates']=1
    write(out/'configs/gpu-preflight.json',pre)
    def limits(searches,usd,cny):
        request=dict(teacher=dict(input=0,output=0),rerank=dict(input=131072,output=1024),embedding=dict(input=8192,output=0))
        cap=dict(http=dict(teacher=0,rerank=searches,embedding=searches),input_upper=searches*(131072+8192),
            output_cap=searches*1024,usd=usd,cny=cny,wall_seconds=28800)
        return dict(request=request,pricing=source['collection_limits']['pricing'],main=cap,total=deepcopy(cap))
    rl=cfg('rl','train')
    rng=random.Random(42); buckets=[]
    for b in ['A01','B01','R01']:
        bucket=sorted(r['task_id'] for r in manifest if r['split']=='train' and r['branch']==b)
        rng.shuffle(bucket); buckets.append(bucket)
    rl['task_ids']=[bucket[i] for i in range(5) for bucket in buckets if i<len(bucket)]
    rl.update(training={**training,'max_steps':22,'num_train_epochs':1,'gradient_accumulation_steps':1},
        checkpoint_steps=[8,16,22],expected_updates=22,group_attempts=1,post_update_task_ids=[],
        sft_adapter=None,collection_limits=limits(66,3,.08))
    rl['grpo'].update(generation_batch_size=4,steps_per_generation=2,num_iterations=1)
    write(out/'configs/rl.json',rl)
    for name in ['base-val','sft-val-3','sft-val-6','sft-val-9','rl-val-8','rl-val-16','rl-val-22','base-test','sft-test','rl-test']:
        evaluation=cfg(name,'test' if name.endswith('test') else 'validation')
        evaluation['collection_limits']=limits(18,.9,.024)
        write(out/f'configs/{name}.json',evaluation)
    files=[p for p in (root/'scripts/agentic_sync_files.txt').read_text(encoding='utf-8').splitlines()
           if p.startswith('mymind/') and (root/p).is_file() and not p.startswith('mymind/tests/')]
    files += ['mymind/rl/three_branch_pilot.py','mymind/rl/teacher_budget.py','mymind/agents/customer_policy/teacher_length.py']
    for file in ['mymind/rl/grading.py','mymind/agents/customer_policy/protocol.py']:
        assert (root/file).read_bytes()==(root/'runs/success-pool-v1/fact-sources-v5/source_snapshot'/file).read_bytes()
    files=sorted(set(files))
    (out/'source_files.txt').write_text('\n'.join(files)+'\n',encoding='utf-8')
    for file in files:
        dest=out/'source_snapshot'/file; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(root/file,dest)
    # Validation/test teacher references remain local and are not deployed.
    resources=[p for folder in ['public','private','policy','train','configs'] for p in (out/folder).glob('*.json')]
    resources += [out/'manifest.json',out/'split_check.json',out/'mask_audit.json',out/'source_files.txt',out/'allocation_method.txt']
    resources += [out/name for name in ['supervise.py','run.sh','start.sh']]
    deployment=[dict(source=posix((out/'source_snapshot'/f).relative_to(root)),destination=posix(Path('src')/'three-branch-pilot-v1'/f)) for f in files]
    deployment += [dict(source=posix(p.relative_to(root)),destination=posix(p.relative_to(root))) for p in resources]
    write(out/'deployment_manifest.json',deployment)
    print('frozen',len(files),'source files;',len(deployment),'deployment files')

if __name__=='__main__': main()
