"""教师文本请求的本地保守长度估算；不执行供应商分词或计数。"""
import json
import math


def calibrated_input_estimate(messages, tools=None):
    """保留的文本/工具请求校准估算；不是供应商分词或数学上界。

    旧96请求最高usage/字节估算为0.259，固定0.30及512余量，
    另48请求作保留验证。财务预留仍使用input_token_upper_bound。
    """
    return math.ceil(input_token_upper_bound(messages, tools) * .30) + 512


def text_bytes(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return len(value.encode('utf-8'))


def input_token_upper_bound(messages, tools=None):
    # 文本按一个UTF-8字节至多一个token预留；每消息64、请求256留作模板估算。
    # 只计模型可见内容与工具结构，不计HTTP JSON转义或财务配置。
    total = 256
    for message in messages:
        total += 64 + sum(text_bytes(value) for key, value in message.items()
                          if value is not None and key not in {'role', 'tool_call_id'})
    if tools:
        total += text_bytes(tools)
    return total
