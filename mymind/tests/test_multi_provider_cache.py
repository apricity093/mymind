import asyncio
from types import SimpleNamespace

from agents.agent_orchestrator import GeneralAgent, Request
from core.cache_metrics import CacheMetricsCollector, ObservedCacheStore
from core.llm_gateway import (
    CacheUsage, DeepSeekAnthropicGateway, LLMResult, LLMRequest, OpenAIGateway,
)


def test_provider_usage_formulas_are_normalized():
    anthropic = CacheUsage("anthropic", input_tokens=20, cache_read_tokens=100, cache_write_tokens=10)
    assert anthropic.total_input_tokens == 130
    openai = CacheUsage("openai", input_tokens=130, cache_read_tokens=100)
    assert openai.total_input_tokens == 130
    deepseek = CacheUsage("deepseek", input_tokens=None, cache_read_tokens=100, cache_miss_tokens=30)
    assert deepseek.total_input_tokens == 130


def test_metrics_separate_application_and_provider_rates():
    metrics = CacheMetricsCollector()
    metrics.record_application("knowledge", "hit")
    metrics.record_application("knowledge", "miss")
    metrics.record_provider("deepseek", "model", CacheUsage("deepseek", 100, 70, 0, 30, True, "hit"))
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["application.knowledge.hit"] == 1
    assert snapshot["counters"]["provider.deepseek.model.read_tokens"] == 70


def test_observed_cache_records_failures_without_changing_store_contract():
    class Store:
        def get(self, namespace, key):
            return {"ok": True}
        def set(self, namespace, key, value, ttl):
            return None
        def invalidate_namespace(self, namespace):
            return 1

    metrics = CacheMetricsCollector()
    observed = ObservedCacheStore(Store(), metrics)
    assert observed.get("knowledge", "q") == {"ok": True}
    assert metrics.snapshot()["counters"]["application.knowledge.hit"] == 1


def test_agent_response_keeps_request_scoped_cache_metadata():
    class Gateway:
        provider = "deepseek"

        async def complete(self, request: LLMRequest):
            await asyncio.sleep(0)
            usage = CacheUsage("deepseek", 10, 5, 0, 5, True, "hit")
            return LLMResult("已处理", usage, {"request": request.messages[-1]["content"]})

    async def run():
        agent = GeneralAgent(None, "model", gateway=Gateway())
        first, second = await asyncio.gather(
            agent.handle(Request("甲", "u1", "c1")),
            agent.handle(Request("乙", "u2", "c2")),
        )
        assert first.cache_metadata["request"] == "甲"
        assert second.cache_metadata["request"] == "乙"

    asyncio.run(run())


def test_openai_adapter_reads_cached_token_details_and_sends_cache_key():
    class Completions:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            message = SimpleNamespace(content="ok")
            usage = {
                "prompt_tokens": 1200,
                "prompt_tokens_details": {"cached_tokens": 1024, "cache_write_tokens": 0},
            }
            return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage,
                                   model_dump=lambda: {"usage": usage})

    async def run():
        gateway = OpenAIGateway("test", "gpt-test")
        completions = Completions()
        gateway.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        result = await gateway.complete(LLMRequest(
            model="gpt-test", stable_prompt="stable", messages=[{"role": "user", "content": "q"}],
            cache_identity="agent:v1",
        ))
        assert completions.kwargs["prompt_cache_key"] == "agent:v1"
        assert result.usage.cache_read_tokens == 1024
        assert result.usage.total_input_tokens == 1200

    asyncio.run(run())


def test_deepseek_anthropic_adapter_uses_automatic_cache_and_parses_usage():
    class Messages:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                content=[{"type": "text", "text": "ok"}],
                usage={"input_tokens": 200, "prompt_cache_hit_tokens": 150,
                       "prompt_cache_miss_tokens": 50},
            )

    async def run():
        gateway = DeepSeekAnthropicGateway("test", "deepseek", "https://api.deepseek.com/anthropic")
        messages = Messages()
        gateway.client = SimpleNamespace(messages=messages)
        result = await gateway.complete(LLMRequest(
            model="deepseek", stable_prompt="stable", dynamic_prompt="dynamic",
            messages=[{"role": "user", "content": "q"}],
        ))
        assert isinstance(messages.kwargs["system"], str)
        assert result.usage.cache_read_tokens == 150
        assert result.usage.cache_miss_tokens == 50
        assert result.usage.status == "hit"

    asyncio.run(run())
