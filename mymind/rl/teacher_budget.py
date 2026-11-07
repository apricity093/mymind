"""冻结train采集：请求预留、usage费用估算、未知占用及持久HTTP计数。"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import threading
import time

from agents.customer_policy.teacher_length import input_token_upper_bound, text_bytes
from rl.runtime import write_json


class CollectionStopped(RuntimeError):
    pass


def nonnegative_int(value):
    return type(value) is int and value >= 0


def price_period(start, end):
    # 供应商未说明跨边界按哪个时点计费；覆盖区间碰到高峰即按高峰。
    def peak(stamp):
        utc = datetime.fromtimestamp(stamp, timezone.utc)
        return utc.weekday() < 5 and (1 <= utc.hour < 4 or 6 <= utc.hour < 10)
    samples = [start, end] + list(range(int(start // 3600 + 1) * 3600, int(end) + 1, 3600))
    flags = [peak(stamp) for stamp in samples]
    if any(flags):
        reason = 'boundary_peak_conservative' if not all(flags) else 'weekday_peak_holiday_unverified'
        return 'peak', reason
    return 'off_peak', 'entire_request_outside_weekday_peak_windows'


class CollectionBudget:
    def __init__(self, output, limits, stage):
        self.output, self.limits, self.stage = output, limits, stage
        self.path = output / 'budget.json'
        self.lock = threading.Lock()
        self.request_context = {}
        self.state = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {
            'requests': [], 'stages': {}, 'stop_reason': None,
            'pricing': deepcopy(limits['pricing']),
            'accounting': 'usage-based local fee estimates; not supplier invoices'}
        if self.state['pricing'] != limits['pricing']:
            raise ValueError('恢复须沿用首次记录价格')
        for entry in self.state['requests']:
            if entry['accounting_state'] == 'in_flight':
                entry.update(accounting_state='unknown', accounting_reason='interrupted_without_response')
        self.state['stages'].setdefault(stage, time.time())
        self.save()

    def totals(self, entries=None):
        entries = self.state['requests'] if entries is None else entries
        result = {state: {'usd': 0., 'cny': 0.} for state in ['settled', 'in_flight', 'unknown']}
        for entry in entries:
            kind = entry['accounting_state']
            money = entry['estimated_cost'] if kind == 'settled' else entry['reservation']
            for currency in ('usd', 'cny'):
                result[kind][currency] += money[currency]
        result['occupied'] = {c: sum(result[s][c] for s in ['settled','in_flight','unknown']) for c in ('usd','cny')}
        result['authorization_remaining'] = {c: None if self.limits['total'][c] is None else
            self.limits['total'][c] - result['occupied'][c] for c in ('usd','cny')}
        result['http'] = {s: sum(e['service']==s for e in entries) for s in self.limits['request']}
        return result

    def save(self):
        self.state['totals'] = self.totals()
        write_json(self.path, self.state)

    def check(self):
        if self.state['stop_reason']:
            raise CollectionStopped(self.state['stop_reason'])
        if self.remaining_seconds() <= 0:
            self.stop('wall_clock_limit')
            raise CollectionStopped('wall_clock_limit')

    def remaining_seconds(self):
        return self.limits[self.stage]['wall_seconds'] - (time.time() - self.state['stages'][self.stage])

    def stop(self, reason):
        self.state['stop_reason'] = reason
        self.save()

    def mark_inflight_unknown(self, reason):
        with self.lock:
            for entry in self.state['requests']:
                if entry['accounting_state'] == 'in_flight':
                    entry.update(accounting_state='unknown', accounting_reason=reason)
            self.save()

    def exceeded(self, entries):
        for scope, rows in [('total', entries), (self.stage, [e for e in entries if e['stage']==self.stage])]:
            caps = self.limits[scope]
            money = self.totals(rows)['occupied']
            if (any(caps[c] is None or money[c] > caps[c] for c in ('usd','cny')) or
                sum(e['input_upper'] for e in rows) > caps['input_upper'] or
                sum(e['output_cap'] for e in rows) > caps['output_cap'] or
                any(sum(e['service']==s for e in rows) > n for s,n in caps['http'].items())):
                return scope
        return None

    def reserve(self, service, payload):
        with self.lock:
            self.check()
            if service == 'embedding':
                texts = payload['input']
                texts = [texts] if isinstance(texts, str) else texts
                inp = sum(text_bytes(text) + 16 for text in texts)
            else:
                inp = input_token_upper_bound(payload.get('messages', []), payload.get('tools'))
            out = payload.get('max_tokens', 0)
            ceiling = self.limits['request'][service]
            if inp > ceiling['input'] or out > ceiling['output']:
                self.stop('request_token_limit:' + service)
                raise CollectionStopped(self.state['stop_reason'])
            rates = self.state['pricing']
            reservation = {'usd': 0., 'cny': 0.}
            if service == 'embedding':
                reservation['cny'] = inp * rates['embedding_cny_per_million'] / 1e6
            else:
                peak = rates['deepseek']['peak']
                reservation['usd'] = (inp * peak['miss'] + out * peak['output']) / 1e6
            proposed = dict(service=service, stage=self.stage, requested_model=payload.get('model'),
                input_upper=inp, input_unit='utf8_based_token_upper_estimate', output_cap=out,
                reservation=reservation, accounting_state='in_flight', started_at=time.time())
            proposed.update(self.request_context)
            entries = self.state['requests'] + [proposed]
            exceeded = self.exceeded(entries)
            if exceeded:
                self.stop('global_limit:' + exceeded)
                raise CollectionStopped(self.state['stop_reason'])
            self.state['requests'].append(proposed)
            self.save()
            write_json(self.output / f'http-request-{len(entries):05d}.json',
                       {'service': service, 'payload': payload})
            return len(entries)-1

    def usage_cost(self, entry, body):
        usage = body.get('usage')
        if not isinstance(usage, dict):
            return None, {'reason': 'usage_missing'}
        pricing = self.state['pricing']
        if entry['service'] == 'embedding':
            tokens = usage.get('total_tokens', usage.get('prompt_tokens'))
            if not nonnegative_int(tokens):
                return None, {'reason': 'embedding_input_usage_missing'}
            return {'usd': 0., 'cny': tokens * pricing['embedding_cny_per_million']/1e6}, {
                'input_tokens': tokens, 'rate_cny_per_million': pricing['embedding_cny_per_million']}
        response_model = body.get('model')
        if response_model and response_model.lower() not in pricing['deepseek_response_models']:
            return None, {'reason': 'unpriced_response_model', 'model': response_model}
        prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
        hit, miss = usage.get('prompt_cache_hit_tokens'), usage.get('prompt_cache_miss_tokens')
        if prompt is None and nonnegative_int(hit) and nonnegative_int(miss):
            prompt = hit + miss
        if not nonnegative_int(prompt) or not nonnegative_int(completion):
            return None, {'reason': 'prompt_or_completion_usage_missing'}
        if 'total_tokens' in usage and usage['total_tokens'] != prompt + completion:
            return None, {'reason': 'inconsistent_total_usage'}
        if hit is None and miss is None:
            hit = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if nonnegative_int(hit) and miss is None and hit <= prompt:
            miss = prompt-hit
        if nonnegative_int(miss) and hit is None and miss <= prompt:
            hit = prompt-miss
        if hit is None and miss is None:
            hit, miss, cache_basis = 0, prompt, 'cache_details_missing_all_input_priced_as_miss'
        elif not nonnegative_int(hit) or not nonnegative_int(miss) or hit+miss != prompt:
            return None, {'reason': 'inconsistent_cache_usage'}
        else:
            cache_basis = 'reported_cache_counts'
        period, timing_basis = price_period(entry['started_at'], entry['completed_at'])
        rates = pricing['deepseek'][period]
        return {'usd': (hit*rates['hit']+miss*rates['miss']+completion*rates['output'])/1e6, 'cny': 0.}, {
            'prompt_cache_hit_tokens': hit, 'prompt_cache_miss_tokens': miss, 'completion_tokens': completion,
            'cache_basis': cache_basis, 'period': period, 'timing_basis': timing_basis, 'rates': rates,
            'model_basis': 'response_model' if response_model else 'requested_model_response_model_missing'}

    def finish(self, index, status, body):
        with self.lock:
            entry = self.state['requests'][index]
            if entry.get('completed_at') is not None:
                return  # 同一完成记录重复处理不重复结算、不覆盖首次响应。
            entry.update(http_status=status, usage=body.get('usage'), response_model=body.get('model'),
                         completed_at=time.time())
            cost, basis = self.usage_cost(entry, body)
            entry.update(accounting_state='settled' if cost is not None else 'unknown',
                         billing_basis=basis, estimated_cost=cost)
            if cost is None:
                self.state['stop_reason'] = 'usage_unsettled_requires_review'
            write_json(self.output / f'http-{index+1:05d}.json', {
                'service': entry['service'], 'status': status,
                'response': body})
            if status >= 400:
                self.state['stop_reason'] = f'service_http_{status}'
            if basis.get('reason') == 'unpriced_response_model':
                self.state['stop_reason'] = 'unpriced_response_model'
            if self.exceeded(self.state['requests']):
                self.state['stop_reason'] = 'usage_exceeds_authorized_limits'
            if cost and any(cost[c] > entry['reservation'][c] for c in ('usd','cny')):
                self.state['stop_reason'] = 'usage_exceeds_request_reservation'
            self.save()

    def hooks(self, service):
        async def before(request):
            request.extensions['collection_index'] = self.reserve(service, json.loads(request.content))
        async def after(response):
            await response.aread()
            try:
                body = response.json()
            except ValueError:
                body = {'response_text': response.text}
            self.finish(response.request.extensions['collection_index'], response.status_code, body)
            self.check()
        return {'request': [before], 'response': [after]}
