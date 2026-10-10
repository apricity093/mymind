import sys
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def isolate_embedding_configuration(monkeypatch, tmp_path):
    """默认单元测试不调用本机配置的真实 embedding；配置契约测试可显式覆盖。"""
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    monkeypatch.setenv("INTENT_TEMPLATE_CACHE_PATH", str(tmp_path / "intent_templates.json"))
