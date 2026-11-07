"""DeepSeek 教师和本地 Qwen 使用同一公开消息及工具 schema。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
import time
import os
from pathlib import Path
from collections import Counter

from agents.customer_policy.protocol import SEARCH_SCHEMA


TOOLS = [{"type": "function", "function": SEARCH_SCHEMA}]


@dataclass
class Generation:
    text: str
    calls: list[dict]
    assistant: dict
    diagnostics: dict


def parse_qwen(text):
    blocks = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)
    if "<tool_call>" in text and not blocks:
        raise ValueError("工具 JSON 未完整生成")
    calls = []
    for i, block in enumerate(blocks):
        call = json.loads(block)
        arguments = call["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("工具参数须为对象")
        calls.append({"id": f"qwen-call-{i}", "type": "function",
                      "function": {"name": call["name"], "arguments": arguments}})
    assistant = {"role": "assistant", "content": text if not calls else text.split("<tool_call>")[0].strip()}
    if calls:
        assistant["tool_calls"] = calls
    return Generation(text, calls, assistant, {})


class DeepSeekGeneration:
    def __init__(self, api_key, config):
        if config["model"] != "deepseek-flash":
            raise ValueError("教师模型须为 deepseek-flash")
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key, base_url=config["base_url"],
                                  timeout=config["timeout_s"], max_retries=config["retries"])
        self.config = config

    def input_token_upper_bound(self, messages):
        from agents.customer_policy.teacher_length import input_token_upper_bound, calibrated_input_estimate
        if self.config.get('length_estimator') == 'retained-text-v1':
            return calibrated_input_estimate(messages, TOOLS)
        return input_token_upper_bound(messages, TOOLS)

    async def generate(self, messages, tools_enabled=True, max_new_tokens=None):
        start = time.monotonic()
        mode_options = ({'reasoning_effort': self.config['reasoning_effort']}
                        if self.config['thinking'] == 'enabled' else
                        {'temperature': self.config['temperature']})
        # 原始响应包含供应商工具历史及实际返回的模型名。
        response = await self.client.chat.completions.create(model=self.config["model"], messages=messages,
            tools=TOOLS if tools_enabled else None,
            top_p=self.config["top_p"],
            max_tokens=max_new_tokens or self.config["max_new_tokens"], response_format={"type": self.config["response_format"]},
            extra_body={"thinking": {"type": self.config["thinking"]}}, **mode_options)
        raw = response.model_dump()
        assistant = {"role": "assistant", **raw["choices"][0]["message"]}
        calls = assistant.get("tool_calls") or []
        return Generation(assistant.get("content") or "", calls, assistant,
            {"raw_response": raw, "latency_ms": (time.monotonic()-start)*1000})

    async def close(self):
        await self.client.close()


class QwenGeneration:
    def __init__(self, model_path, config, adapter_path=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.config = torch, config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
        if adapter_path is not None:
            from peft import PeftModel, get_peft_model_state_dict
            from safetensors.torch import load_file
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False,
                                                   autocast_adapter_dtype=False)
            saved = load_file(str(Path(adapter_path) / "adapter_model.safetensors"))
            loaded = get_peft_model_state_dict(self.model)
            matched = sum(torch.equal(loaded[name].detach().cpu(), value)
                          for name, value in saved.items() if name in loaded)
            if set(saved) != set(loaded) or matched != len(saved):
                raise ValueError("实际加载的 adapter 张量与推理产物不一致")
            adapter_evidence = {"saved_tensors": len(saved), "matched_tensors": matched,
                "nonzero_tensors": sum(bool(torch.count_nonzero(value)) for value in saved.values()),
                "parameter_dtypes": dict(Counter(str(value.dtype) for value in loaded.values()))}
        else:
            adapter_evidence = None
        self.model.requires_grad_(False)
        self.model.eval()
        self.model.config.use_cache = True
        provenance = Path(model_path) / "base_revision.json"
        base = json.loads(provenance.read_text(encoding="utf-8")) if provenance.exists() else {}
        if config.get("model_revision") and base.get("revision") != config["model_revision"]:
            raise ValueError("本地基座 revision 与启动配置不一致")
        if config.get("model_id") and base.get("repo_id") != config["model_id"]:
            raise ValueError("本地基座标识与启动配置不一致")
        self.load_info = {
            "model_path": str(Path(model_path).resolve()), "base_provenance": base,
            "adapter_path": str(Path(adapter_path).resolve()) if adapter_path else None,
            "active_adapters": self.model.active_adapters if adapter_path else [],
            "adapter_weights": adapter_evidence,
            "model_class": type(self.model).__name__, "merged": False,
            "parameter_dtypes": dict(Counter(str(p.dtype) for p in self.model.parameters())),
            "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "attention": self.model.config._attn_implementation, "training": self.model.training,
            "tokenizer_path": str(Path(model_path).resolve()),
            "chat_template": self.tokenizer.chat_template,
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
            "device": str(self.model.device), "generation_mode": "torch.inference_mode",
        }

    def token_count(self, messages):
        # 统一保留工具 schema 计数，关闭工具的最终轮也不回收其预算。
        prompt = self.tokenizer.apply_chat_template(messages, tools=TOOLS,
                                                    tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    async def generate(self, messages, tools_enabled=True, max_new_tokens=None):
        torch = self.torch
        prompt = self.tokenizer.apply_chat_template(messages, tools=TOOLS if tools_enabled else None,
                                                   tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_tokens = inputs.input_ids.shape[1]
        limit = max_new_tokens or self.config["max_new_tokens"]
        if input_tokens + limit > self.config["context_tokens"]:
            raise ValueError("公开上下文超过本次 token 预算")
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        options = dict(max_new_tokens=limit, do_sample=self.config["temperature"] > 0,
                       pad_token_id=self.tokenizer.eos_token_id)
        if options["do_sample"]:
            options["temperature"] = self.config["temperature"]
            options["top_p"] = self.config["top_p"]
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        ids = output[0, input_tokens:].tolist()
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        try:
            result = parse_qwen(text)
        except (ValueError, KeyError, TypeError) as ex:
            result = Generation(text, [], {"role": "assistant", "content": text},
                                {"protocol_parse_error": type(ex).__name__})
        parse_diagnostics = result.diagnostics
        result.diagnostics = dict(prompt=prompt, input_tokens=input_tokens, output_tokens=len(ids),
            output_token_ids=ids, latency_ms=(time.monotonic()-start)*1000,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            max_new_tokens=limit, tools_enabled=tools_enabled,
            reached_token_limit=len(ids) >= limit)
        result.diagnostics.update(parse_diagnostics)
        return result
