"""Deliver the complete event candidate and the limits of its local review."""
import json
import os
from pathlib import Path
import sys

BACKEND=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BACKEND))
from benchmarks.scripts.balanced_policy_content import load, dump, SPLITS


def main():
    root=Path(os.environ['MYMIND_WORK_ROOT']).resolve()
    data=root/'data/agentic_rl/balanced-policy-v1'
    logs=root/'runs/balanced-policy-v1/data-build'
    summary=load(data/'metadata/acceptance_summary.json')
    meta=load(data/'metadata/task_manifest.json')
    public={t['task_id']:t for split in SPLITS for t in load(data/f'public/{split}.json')}
    private={t['task_id']:t for split in SPLITS for t in load(data/f'private/{split}.json')}
    reviews=load(data/'metadata/content_reviews.json')
    checks={r['task_id']:r for r in load(data/'metadata/semantic_review.json')}
    auth_ids=[f'balanced-{sid}-{n:02}' for sid in ('A01','A02') for n in (1,2,5)]
    reps=auth_ids+['balanced-P01-08','balanced-P01-16','balanced-M03-04','balanced-P06-03','balanced-F03-24','balanced-R04-19','balanced-F02-06']
    dump(data/'metadata/representative_review.json',[
        dict(task_id=tid,public_task=public[tid],private_standard=private[tid],
             evidence_dependencies=next(m['evidence_dependencies'] for m in meta if m['task_id']==tid),
             review=next(r['task_notes'][tid] for r in reviews if tid in r['task_notes']),
             source='同一执行助手AI；旧认证重复失败记录保留在runs/balanced-policy-v1/data-build/r2_content_evidence/')
        for tid in reps])
    version=load(data/'metadata/version.json');version['status']=summary['status']
    dump(data/'metadata/version.json',version)
    lines=[
        'MyMind balanced-policy-v1 完整事件候选交付，2026-11-05',
        '结论：'+summary['status']+'；不是冻结批准，test未封存，不启动教师采集。',
        '唯一干预为实质事件材料与证据关系；政策、公开协议、v14-scoped-status、模型和训练算法保持。',
        '标准：'+summary['standards_revision'],
        f"目标600，构造{summary['constructed']}，自动通过{summary['passed']}，同一AI内容通过{summary['content_reviewed_nonduplicate']}，待修{len(summary['content_revision_required'])}，未审阅{summary['content_unreviewed']}，未构造{summary['pending']}。",
        '保留原72题，修复其中6份认证；新增44族528题，50族每族12题。原seed42整族划分360/120/120，每层15/5/5，billing/technical各半。',
        '', '集合：构造/自动通过/内容通过/未构造缺口',
    ]
    for split,target in [('train',360),('validation',120),('test',120)]:
        r=summary['split_report'][split]
        lines.append(f"{split}: {r['constructed']}/{r['passed']}/{r['content_passed']}/{target-r['constructed']}")
    lines+=['','24层：train | validation | test，每项构造/自动通过/内容通过/目标缺口']
    for sid in summary['distribution']:
        cells=[]
        for split,target in [('train',15),('validation',5),('test',5)]:
            rows=[m for m in meta if m['split']==split and m['stratum_id']==sid]
            n=len(rows)
            passed=sum(checks[m['task_id']]['status']=='passed' for m in rows)
            content=sum(checks[m['task_id']]['content_status']=='passed' for m in rows)
            cells.append(f'{n}/{passed}/{content}/{target-content}')
        lines.append(sid+' | '+' | '.join(cells))
    lines+=['','56子型：train | validation | test，每项构造/自动通过/内容通过/目标缺口']
    for row in summary['subtype_coverage']:
        cells=[]
        for split in SPLITS:
            c=row['splits'][split]
            cells.append(f"{c['constructed']}/{c['passed']}/{c['content_passed']}/{c['target']-c['content_passed']}")
        lines.append(row['stratum_id']+'.'+row['subtype_id']+' | '+' | '.join(cells))
    lines+=['','50族实际完成及相容性：']
    for r in reviews:
        lines.append(r['family_id']+'：12/12，'+r['status']+'；'+r['basis'])
    lines+=['',
        '审阅方式：content_reviews.json保存明确的50份逐族结论、600题实际条件及依据、例外表和本次审阅的公开材料。当前例外为空。自动验收读取这些记录；缺记录或公开材料已修改即未审阅，不按A标签、非A数量或字符串不同授予通过。新增题不继承旧66题结论。',
        '6份认证修复：A01/A02各01、02、05。01区分日历调度缓存与手工新凭据；02区分蓝实例回放槽与绿入口授权；05区分重投序号2新增消费者与首投。脱敏响应自含本次调用，A01明确token_expired，A02不提供原因且有效性/过期明确未知。标准从当前响应解码；直接响应证据足够，无需强制背景，不索取完整凭据或已给摘要。详见representative_review.json。',
        '新增关系代表：M03-04按成员自付份额分别定位团长/参与人，R1升级、R2核对未知并存；P06-03合并后项目书签不能沿用旧分公司功能单；F03-24按恢复后批次核对应有40/实际36，不由快照可读推出完整；R04-19实际到账时已超过上限而当前已到账；F02-06已给请求号、时间、摘要不再索取。',
        '技术01与14均涉及定时任务：01决定性定位经过共享/专用出口路由，14按日历周期及机器人账户定位；共享任务主题不等于同一来源摘录。17修复验收与24快照恢复分别核对修复批次和快照后当前业务范围。来源独立性是同一AI判断，非外部实证。',
    ]
    for split in ('validation','test'):
        d=summary['duplication'][split]
        lines.append(f"{split}对train：整份对象正文相同{len(d['identical_object_bodies'])}题；归一化状态匹配{len(d['normalized_state_matches'])}题，完整匹配清单见acceptance_summary.json。")
    lines+=['状态重复保留：认证结果、退款边界及故障条件复用同一政策推理是预期现象。独立性依据实际事件、对象、业务摘录及关系；正文不相同不是独立性证明，不要求参考答案唯一。',
        '自动验收：文件及ID对应、最终材料解码、对象唯一定位、来源状态、标准事实/待补/建议/参数/处置/严重项、引用依赖、全部参考固定评分、真实条件归层/子型配额、整族边界及公开信息隔离。原四项关系探针继续通过；新机制增加当前认证响应改变过期状态、实际到账计时一致性两项，未扩验稳定v14。',
        'r3构造失败/修复：当时首次build成功；首次validate在分布字段合并处TypeError退出，命令与stderr完整保留；改字典合并后全量自动600通过，但内容未审阅600，未宣称交付完成。随后保存明确内容结论并补入新机制探针，最终全量通过。语义修订及旧认证重复失败分别保留r3_content_repairs.json与r2_content_evidence/，旧r1候选及失败证据未回填新版。',
        '边界：600份参考与标准同过程构造，600份内容结论均为同一执行助手AI，非教师轨迹、SFT示范、独立复核或人工金标。test参与构造审阅，未封存。独立审阅、模型表现、教师可用率、真实检索及训练收益未知；评分早退后的missing=[]不证明完整。',
        '产物：public/、private/、acceptance/三集合；metadata/task_manifest.json、source_families.json、content_reviews.json、semantic_review.json、acceptance_summary.json、pending_slots.json（空）、representative_review.json、version.json。原设计50族600槽位不改。',
        '命令与真实退出状态：runs/balanced-policy-v1/data-build/commands.jsonl；最终validation_编号.json；四项原探针及两项新探针见r3_relation_probes.json。',
        'PowerShell复核：$env:MYMIND_WORK_ROOT=(Get-Location).Path；在项目根目录使用当前agentic_rl python运行 mymind/benchmarks/scripts/balanced_policy_build.py validate。build仅重建数据，不生成审阅通过结论。',
        '需整合事实：r4-selection完整600、50族、24层56子型缺口均0、同一AI内容审阅600、认证6题已修复、v14固定、seed42原配额、test未封存。未重写agentic_plan.md，未改HISTORY.md。',
        '教师0，程序真实模型0，GPU0，外部API0，真实检索0，索引重建0，下载升级0，费用0。无服务器操作、Git提交推送、正式资产替换或服务切换。',
    ]
    if (logs/'selection_r4_probes.json').exists():
        lines+=['',
            'r4-selection 定向一致性修复：原P01-08固定选生成作业、原P01-16固定选审计导出，但表单/标准却是预览基础能力。旧公开材料、标准、参考、原分1.0和通过审阅保留before-selection-r4/，原通过不作为语义正确证据。',
            '实际影响仅balanced-P01-08、balanced-P01-16，均train；同机制其余候选未发现冲突。50族600题定向核对依据见selection_r4_scope_review.json。598份公开任务、标准/参考负载及逐题审阅记录不变，原审阅修订标识保留以记录实际来源；两题按最终材料重新审阅。',
            'P01-08：生成作业/夜批生成作业单/生成作业调用 → 预览作业/夜批预览作业单/预览作业调用；原作业清单和两种用途保留。P01-16：审计任务/审计任务选择单/审计任务调用 → 日常任务/日常任务选择单/日常任务调用；审计导出及日常预览用途分工保留。叙述、关系说明、当前选择、路径、业务关联和表单归属同步；导出及其他变体保持原选择。',
            '两题重推标准仍basic_query、platform_enablement_required=false，整体answer，白名单建议参数及user:2+user:3必要支持不变；标准materials和离线trajectory更新，参考答案负载不变。无需改层、子型、政策或评分器。',
            '针对性验证：原两冲突均被拒绝；修复预览、保留导出、直接认证响应与异项目/异环境对照通过；构造预览选择配生成路径的反例被拒绝。结构化用途直接检查纳入semantic_audit，自然语言仍是同一AI语义审阅。',
            '本次首次定向检查退出1：直接认证响应不使用记录归属，被误报4题；检查改按本次认证记录.实际调用定位，未修改认证材料。修复后selection命令退出0，再build及相关全量validate；命令见commands.jsonl，首次工具输出补记见selection_r4_first_failure.json。',
            '复核：balanced_policy_build.py selection 运行定向反例；balanced_policy_build.py validate 运行全量验收，均使用现有agentic_rl环境及MYMIND_WORK_ROOT。test未封存，费用及外部调用全0；未改agentic_plan、HISTORY或提交推送。',
        ]
    (data/'metadata/delivery.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(summary['status'])


if __name__=='__main__': main()
