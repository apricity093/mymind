"""读取评测 JSON 与逐题 JSONL 记录。"""
import json
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
