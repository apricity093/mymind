"""HTTP诊断记录不更改请求、重试或错误传播。"""
import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from core.llm_gateway import LLMRequest
from experiments.agentic_rl.formal_evaluate import install_http_diagnostics


@pytest.mark.parametrize('failed',[False,True])
def test_trace_preserves_wire_request_and_full_failure_chain(tmp_path,failed):
    path=tmp_path/'trace.jsonl'
    hooks={'request':[],'response':[]}
    request=httpx.Request('POST','https://api.deepseek.com/chat/completions',
        headers={'Authorization':'Bearer private-test-key','x-stainless-retry-count':'0'},
        json={'model':'deepseek-flash','messages':[{'role':'user','content':'公开检索输入'}]},
        extensions={'timeout':{'connect':45,'read':45,'write':45,'pool':45}})
    body=request.content
    result=SimpleNamespace(metadata={'response_model':'deepseek-flash'},usage=SimpleNamespace(input_tokens=10,output_tokens=3))
    error=RuntimeError('API timeout')
    cause=httpx.ReadTimeout('response body not received');error.__cause__=cause
    async def complete(value):
        for hook in hooks['request']:await hook(request)
        await request.extensions['trace']('connection.connect_tcp.complete',{'return_value':SimpleNamespace(get_extra_info=lambda key:['192.0.2.1',443])})
        if failed:
            await request.extensions['trace']('http11.receive_response_body.failed',{'exception':cause})
            raise error
        response=httpx.Response(200,headers={'x-request-id':'id1','set-cookie':'not-recorded'},request=request)
        for hook in hooks['response']:await hook(response)
        return result
    gateway=SimpleNamespace(complete=complete,client=SimpleNamespace(_client=SimpleNamespace(event_hooks=hooks)))
    install_http_diagnostics(gateway,path)
    value=LLMRequest(model='deepseek-flash',stable_prompt='公开输入',max_tokens=1024,temperature=0)
    if failed:
        with pytest.raises(RuntimeError) as caught:asyncio.run(gateway.complete(value))
        assert caught.value is error
    else:assert asyncio.run(gateway.complete(value)) is result
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    wire=next(row for row in rows if row['event']=='http.request')
    assert wire['body']==json.loads(body) and wire['timeout']['read']==45 and wire['retry_count']=='0'
    assert request.content==body and request.headers['Authorization']=='Bearer private-test-key'
    content=path.read_text(encoding='utf-8')
    assert 'private-test-key' not in content and 'set-cookie' not in content
    if failed:assert [x['type'] for x in rows[-1]['exception_chain']]==['RuntimeError','ReadTimeout']
    else:assert rows[-1]['event']=='llm.complete'
