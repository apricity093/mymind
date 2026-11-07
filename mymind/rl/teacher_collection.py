"""教师采集共用客户端、预算及原始轨迹记录。"""
import asyncio
import json
import os
from collections import Counter

from agents.customer_policy.environment import PolicyEnvironment
from agents.customer_policy.generation import DeepSeekGeneration
from agents.customer_policy.protocol import VisibleTrajectory
from agents.customer_policy.runner import run_task
from rl.grading import grade_result
from rl.runtime import write_json
from rl.teacher_budget import CollectionBudget, CollectionStopped


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


async def collect(config, output, tasks, standards, clauses, generation, retrieval, budget, metadata):
    from rl.evaluate import process_errors
    stage = config['collection_stage']
    ids = config['pilot_task_ids'] if stage == 'pilot' else config['task_ids']
    rows, accepted, stopped = [], [], None
    # 读取全部train原件，恢复和首批/余量共用同一候选编号与额度。
    for task_id in config['task_ids']:
        succeeded = False
        for candidate in range(1, config['candidates_per_task'] + 1):
            path = output / f'{task_id}-candidate-{candidate}.json'
            partial = output / f'{task_id}-candidate-{candidate}.partial.json'
            if path.exists():
                row = read(path)
            elif partial.exists():
                row = read(partial)
                row.update(candidate=candidate, runtime_error={'kind': 'interrupted'},
                           infrastructure_failure=True, elapsed_s=None)
                grade = grade_result(standards[task_id], row['raw_result'],
                                     VisibleTrajectory.model_validate(row['trajectory']), clauses)
                row.update(grade=grade.to_dict(), process_errors=process_errors(row),
                           grading_revision=config['grading_revision'], standards_revision=config['standards_revision'])
                write_json(path, row)
                budget.stop('interrupted_candidate_requires_review')
            elif succeeded or task_id not in ids or stopped:
                break
            else:
                try:
                    budget.check()
                except CollectionStopped as ex:
                    stopped = str(ex)
                    break
                try:
                    row = await asyncio.wait_for(run_task(tasks[task_id], generation, PolicyEnvironment(retrieval),
                        budget=config['budget'], on_progress=lambda state: write_json(partial, state),
                        public_guidance=config.get('public_teacher_guidance')),
                        timeout=budget.remaining_seconds())
                except asyncio.TimeoutError:
                    budget.mark_inflight_unknown('candidate_wall_clock_timeout')
                    budget.stop('wall_clock_limit')
                    stopped = 'wall_clock_limit'
                    break  # partial原件保留，恢复按已耗候选处理。
                row.update(candidate=candidate, grading_revision=config['grading_revision'],
                           standards_revision=config['standards_revision'])
                budget.mark_inflight_unknown('candidate_finished_without_response_usage')
                grade = grade_result(standards[task_id], row['raw_result'],
                                     VisibleTrajectory.model_validate(row['trajectory']), clauses)
                row.update(grade=grade.to_dict(), process_errors=process_errors(row))
                write_json(path, row)
                if row['infrastructure_failure'] or (row['runtime_error'] or {}).get('kind') == 'generation_failure':
                    budget.stop('infrastructure_failure_requires_review')
            rows.append({'task_id': task_id, 'candidate': candidate, 'file': path.name,
                         'grade': row['grade'], 'runtime_error': row.get('runtime_error'),
                         'infrastructure_failure': row['infrastructure_failure'], 'process_errors': row['process_errors']})
            if (not succeeded and row['grade']['complete_success'] and not row['infrastructure_failure']
                    and not row.get('runtime_error') and not row['process_errors']):
                accepted.append({'task_id': task_id, 'candidate': candidate, 'file': path.name})
                succeeded = True
    selected = {r['task_id'] for r in accepted}
    gaps = [r for r in metadata if r['split'] == 'train' and r['task_id'] not in selected]
    summary = dict(stage=stage, attempts=rows, accepted=accepted, accepted_count=len(accepted),
        budget_totals=budget.totals(),
        missing_task_ids=[r['task_id'] for r in gaps],
        missing_by_stratum=dict(Counter(r['stratum_id'] for r in gaps)),
        missing_by_subtype=dict(Counter(r['stratum_id']+'.'+r['subtype_id'] for r in gaps)),
        unstarted_task_ids=[t for t in config['task_ids'] if not any(r['task_id']==t for r in rows)
                            and not list(output.glob(f'{t}-candidate-*.partial.json'))],
        interrupted_files=[p.name for p in output.glob('*.partial.json') if not (output / p.name.replace('.partial','')).exists()],
        stopped=stopped or budget.state['stop_reason'], grading_revision=config['grading_revision'],
        standards_revision=config['standards_revision'])
    write_json(output / f'summary-{stage}.json', summary)
    # 导出只引用未经编辑的原件；失败与partial均不删除。
    write_json(output / 'accepted_trajectories.json', [read(output/r['file']) for r in accepted])
    return summary


async def collect_real(root, config, output, tasks, standards, clauses, collector=collect):
    import httpx
    from agents.customer_policy.configuration import rerank_gateway
    from agents.customer_policy.retrieval import build_retrieval
    embedding_config = config['embedding']
    if not embedding_config['base_url'] or not embedding_config['dimensions']:
        raise ValueError('须先确认既有索引的Embedding端点、维度及适用价格；当前建议配置不能直接调用')
    stage = config['collection_stage']
    if any(config['collection_limits'][stage][c] is None for c in ('usd', 'cny')):
        raise ValueError('剩余采集金额尚未制定；仅准备首批提案')
    if stage == 'remainder' and not (output / 'summary-pilot.json').exists():
        raise ValueError('须先完成首批验收并另获剩余采集授权')
    budget = CollectionBudget(output, config['collection_limits'], stage)
    budget.check()
    gateway = rerank_gateway(config['retrieval'])
    generation = DeepSeekGeneration(os.environ['DEEPSEEK_API_KEY'], config['generation'])
    retrieval = None
    try:
        # SDK重试关闭；每个真正派发的HTTP都先预留并持久记录。
        from openai import AsyncOpenAI
        await gateway.client.close()
        gateway.client = AsyncOpenAI(api_key=gateway.api_key, base_url=gateway.base_url, max_retries=0,
            timeout=config['retrieval']['rerank_timeout_s'],
            http_client=httpx.AsyncClient(event_hooks=budget.hooks('rerank')))
        await generation.client.close()
        generation.client = AsyncOpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url=config['generation']['base_url'],
            max_retries=0, timeout=config['generation']['timeout_s'],
            http_client=httpx.AsyncClient(event_hooks=budget.hooks('teacher')))
        retrieval = build_retrieval(root, config['retrieval'], gateway, prepared_only=True)
        embedding = retrieval.knowledge.embedding
        if (embedding.settings.model != embedding_config['model'] or
                embedding.settings.dimensions != embedding_config['dimensions'] or
                embedding.base_url != embedding_config['base_url']):
            raise ValueError('实际Embedding环境与采集配置不一致')
        retrieval.knowledge.embedding.collection_budget = budget
        if not retrieval.knowledge._main_complete or not retrieval.knowledge._main.count():
            raise RuntimeError('既有主索引未就绪；停止，不重建')
        text = (root/config['frozen_dir']/'policy/knowledge.md').read_text(encoding='utf-8')
        split = text.index('### 接口认证')
        expected = {'billing-policy': text[:split], 'technical-policy': text[split:]}
        actual = {r['source_id']: r['content'] for r in retrieval.knowledge._rows()}
        if actual != expected:
            raise ValueError('既有索引来源正文与冻结政策不一致；停止，不重建')
        metadata = read(root/config['frozen_dir']/'metadata/task_manifest.json')
        return await collector(config, output, tasks, standards, clauses, generation, retrieval, budget, metadata)
    finally:
        await generation.close()
        await gateway.client.close()
        if retrieval:
            retrieval.knowledge.close()
            retrieval.knowledge.embedding.close()
