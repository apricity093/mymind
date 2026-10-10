"""持久复用和授权额度的实验契约；不作为模型质量证据。"""
import httpx
import pytest
from types import SimpleNamespace

from core.embedding import EmbeddingError, EmbeddingSettings
from experiments.qwen_optimization import BudgetQwen, Runner
from core.document_chunking import BudgetChunker
import asyncio


def test_persistent_count_and_document_vector_reuse(tmp_path):
    settings=EmbeddingSettings('qwen3.7-text-embedding-flash',768,'test-key','https://example.test/v1')
    def response(request):
        return httpx.Response(200,json={'data':[{'index':0,'embedding':[1.0]+[0.0]*767}],'usage':{'total_tokens':7}})
    first=BudgetQwen(settings,tmp_path,0)
    first._client.close()
    first._client=httpx.Client(transport=httpx.MockTransport(response))
    assert first.count_tokens('标题\n正文')==7
    assert first.ledger['http']==1 and first.ledger['encoded_texts']==1
    first.close()
    first.db.close()
    restarted=BudgetQwen(settings,tmp_path,0)
    assert restarted.count_tokens('标题\n正文')==7
    assert restarted.cached_embed(['正文'],'RETRIEVAL_DOCUMENT',['标题'])[0][0]==1
    assert restarted.ledger['http']==1
    restarted.close()
    restarted.db.close()


def test_historical_usage_consumes_authorized_limits(tmp_path):
    component=BudgetQwen(EmbeddingSettings('qwen3.7-text-embedding-flash',768,'test-key'),tmp_path,0)
    component.ledger['encoded_texts']=9736
    with pytest.raises(EmbeddingError,match='authorized_budget_exhausted'):
        component.embed(['不能越过授权预算'],'RETRIEVAL_QUERY')
    assert component.ledger['http']==0
    component.ledger['encoded_texts']=0
    component.ledger['http']=19315
    with pytest.raises(EmbeddingError,match='authorized_budget_exhausted'):
        component.embed(['不能越过授权预算'],'RETRIEVAL_QUERY')
    assert component.ledger['http']==19315
    component.close()
    component.db.close()


def test_bm25_reuses_chunks_without_any_encoding(tmp_path):
    def forbidden(*args,**kwargs):
        raise AssertionError('BM25 单路不应编码文档或查询')
    runner=Runner.__new__(Runner)
    runner.output=tmp_path
    runner.embedding=SimpleNamespace(cached_embed=forbidden,flush=lambda:None)
    runner.by_source={'policy':{'content':'九天内受理退款，需要订单号。'}}
    runner.rag={'questions':[{'id':'q','family_id':'f','question':'退款订单号','split':'test','tags':['condition'],
        'answerable':True,'evidence_groups':[{'alternatives':[{'source_id':'policy','start':0,'end':2,'quote':'九天'}]}]}]}
    runner.chunks=lambda *args:[{'source_id':'policy','chunk_id':'qwen:policy:0','title':'退款','section_path':'','content':runner.by_source['policy']['content']}]
    result=runner.retrieval('qwen','structure',512,0,'test',['bm25'])
    assert result['bm25'][0]['recall_at_5']==1


def test_section_exceeding_counter_request_limit_is_split():
    def limited(text):
        if len(text)>80:
            raise EmbeddingError('input_budget_exceeded')
        return len(text)
    records=BudgetChunker(limited,32,reserve=0).chunk_document('公开','公开退款条件。'*30)
    assert len(records)>1
    assert all(limited('公开\n'+record.text)<=32 for record in records)
    def unavailable(text):
        if len(text)>10:
            raise EmbeddingError('HTTP_401')
        return len(text)
    with pytest.raises(EmbeddingError,match='HTTP_401'):
        BudgetChunker(unavailable,32,reserve=0).chunk_document('公开','公开退款条件。'*30)


def test_measured_gateway_preserves_native_tool_continuation():
    from experiments.gemini_rag import MeasuredGateway
    from core.llm_gateway import LLMResult, CacheUsage, ToolCall
    from agents.agent_orchestrator import TechnicalAgent, AgentFeatureConfig, Request
    class Native:
        provider='anthropic'
        tool_protocol='anthropic'
        def __init__(self):
            self.calls=0
        async def complete(self,request):
            self.calls+=1
            if self.calls==1:
                block={'type':'tool_use','id':'call-1','name':'lookup_error_code','input':{'error_code':'500'}}
                return LLMResult('',CacheUsage('anthropic',input_tokens=10),tool_calls=[ToolCall('call-1','lookup_error_code',{'error_code':'500'})],assistant_message={'role':'assistant','content':[block]})
            assert request.messages[-1]['role']=='user'
            assert request.messages[-1]['content'][0]['type']=='tool_result'
            assert request.messages[-1]['content'][0]['tool_use_id']=='call-1'
            return LLMResult('请稍后重试。',CacheUsage('anthropic',input_tokens=10))
    async def check():
        measured=MeasuredGateway(Native())
        agent=TechnicalAgent(SimpleNamespace(),'test',gateway=measured,features=AgentFeatureConfig.for_variant('E5'))
        result=await agent.handle(Request('500服务器错误','public','public'))
        assert result.success and measured.calls==2
    asyncio.run(check())
