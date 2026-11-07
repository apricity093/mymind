"""Export the accepted three-branch pilot without changing historical pools."""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rl.sft_data import student_messages
from rl.tasks import load_public_tasks, load_private_standards, load_policy_support, validate_task_pairs

VERSION = 'three-branch-pilot-v1'
METHOD = '''Seed 42, Python random.Random. Process A01 then B01/R01 shared families.
Within each block sort family_id, separate reused_old_train and new_synthetic_event.
Allocate old-source quotas by largest remainder of (3,2,2)*old_count/7;
ties use train,validation,test order. New quotas fill remaining (3,2,2).
Shuffle old then new lists with one Random(42) stream and slice train/validation/test.
R01-only families go to train. No student scores, seed search or redraw.
Source type comes from collection metadata, never task performance.'''

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    root = Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    base = root/'runs/success-pool-v1'
    out = base/VERSION
    out.mkdir()  # An existing version is never overwritten.
    (out/'allocation_method.txt').write_text(METHOD, encoding='utf-8')
    pool = read(base/'unified-pool-v5-expansion-v1/success_pool.json')['tasks']
    eligibility = read(base/'source-expansion-v1/metadata/source_eligibility.json')
    blocked = set(eligibility['excluded_dev_families']) | {x['family_id'] for x in eligibility['sealed_families']}
    metadata = {}
    for path in ['mainline-candidates-v1/metadata/task_manifest.json', 'source-expansion-v1/metadata/task_manifest.json']:
        metadata.update({x['task_id']: x for x in read(base/path)})
    kept = [deepcopy(x) for x in pool if x['branch'] in ('A01','B01','R01')]
    families = defaultdict(list)
    for row in kept:
        assert row['source_eligible'] and row['content_accepted'] and row['process_ok']
        assert row['family_id'] not in blocked
        row['source_kind'] = metadata[row['task_id']]['source_kind']
        families[row['family_id']].append(row)
    assert len(kept) == 23 and len(families) == 16
    blocks = [[f for f, rows in families.items() if {r['branch'] for r in rows} == branches]
              for branches in ({'A01'}, {'B01','R01'}, {'R01'})]
    assert list(map(len, blocks)) == [7,7,2]
    rng = random.Random(42)
    assignment = {}
    splits = ['train','validation','test']
    for block in blocks[:2]:
        old = sorted(f for f in block if families[f][0]['source_kind'] == 'reused_old_train')
        new = sorted(set(block)-set(old))
        assert all(len({r['source_kind'] for r in families[f]}) == 1 for f in block)
        quotas = [3*len(old)//7, 2*len(old)//7, 2*len(old)//7]
        ranked = sorted(range(3), key=lambda i: (-([3,2,2][i]*len(old)%7), i))
        for i in ranked[:len(old)-sum(quotas)]:
            quotas[i] += 1
        for items, counts in [(old, quotas),(new,[n-q for n,q in zip([3,2,2],quotas)])]:
            rng.shuffle(items)
            start = 0
            for split, n in zip(splits, counts):
                for f in items[start:start+n]: assignment[f] = split
                start += n
    assignment.update({f:'train' for f in blocks[2]})
    clauses = None
    exported = {s:[] for s in splits}
    standards = {s:[] for s in splits}
    trajectories = {s:[] for s in splits}
    manifest = []
    for row in sorted(kept, key=lambda x:(x['branch'],x['family_id'],x['task_id'])):
        original = root/row['original']
        run = original.parent
        cfg = read(run/'configuration.json')
        public_path = run/'inputs/tasks/candidates.json'
        standard_path = root/row['standard'] if row.get('standard') else root/cfg['standards']
        task = load_public_tasks(public_path)[row['task_id']]
        standard = load_private_standards(standard_path)[row['task_id']]
        policy_path = run/'inputs/policy_support/policy_support.json'
        policy = read(policy_path)
        if clauses is None: clauses = policy
        assert clauses == policy
        split = assignment[row['family_id']]
        task = task.model_copy(update={'split':split})
        standard = standard.model_copy(update={'split':split})
        validate_task_pairs({task.task_id:task},{task.task_id:standard},load_policy_support(policy_path))
        raw = read(original)
        messages = student_messages(raw)  # drops reasoning_content, keeps assistant actions/final
        # Historical generation used its recorded public protocol and guidance.
        # Preserve that original supervision; new student runs use frozen v5.
        assert messages[1] == task.messages()[1]
        assert messages[:2] == raw['generations'][0]['input_messages']
        assert not any('reasoning_content' in m for m in messages)
        exported[split].append(task.model_dump())
        standards[split].append(standard.model_dump())
        trajectories[split].append(dict(task_id=task.task_id, messages=messages, original=row['original']))
        manifest.append({**row,'split':split,'public_source':str(public_path.relative_to(root)),
                         'standard_source':str(standard_path.relative_to(root))})
    expected = {'train':{'A01':3,'B01':3,'R01':5},'validation':{'A01':2,'B01':2,'R01':2},'test':{'A01':2,'B01':2,'R01':2}}
    actual = {s:dict(Counter(r['branch'] for r in manifest if r['split']==s)) for s in splits}
    assert actual == expected
    for s in splits:
        write(out/f'public/{s}.json',exported[s])
        write(out/f'private/{s}.json',standards[s])
        write(out/('train/trajectories.json' if s=='train' else f'sealed/{s}_teacher.json'),trajectories[s])
    write(out/'policy/policy_support.json',clauses)
    write(out/'manifest.json',manifest)
    write(out/'split_check.json',dict(counts=actual,families=assignment,
        source_counts={s:dict(Counter(r['source_kind'] for r in manifest if r['split']==s)) for s in splits},
        excluded_branches=[dict(task_id=r['task_id'],family_id=r['family_id'],original=r['original']) for r in pool if r['branch'] not in ('A01','B01','R01')],
        source_eligibility='source-expansion-v1/metadata/source_eligibility.json',
        mutually_exclusive_families=True,model_calls=0,teacher_reasoning_exported=False))
    print(json.dumps(actual))

if __name__ == '__main__': main()
