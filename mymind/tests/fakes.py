import json
from types import SimpleNamespace


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.lists = {}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, ex=None, px=None, nx=False):
        if nx and key in self.values:
            return False
        self.values[key] = str(value)
        return True

    def setex(self, key, ttl, value):
        self.values[key] = str(value)

    def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = str(value)
        return value

    def delete(self, key):
        self.values.pop(key, None)
        self.lists.pop(key, None)
        return 1

    def eval(self, script, count, key, token):
        if self.values.get(key) == token:
            return self.delete(key)
        return 0

    def lpush(self, key, value):
        self.lists.setdefault(key, []).insert(0, value)
        return len(self.lists[key])

    def llen(self, key):
        return len(self.lists.get(key, []))

    def lrange(self, key, start, end):
        values = self.lists.get(key, [])
        return values[start:] if end == -1 else values[start : end + 1]

    def expire(self, key, ttl):
        return True


class FakeCollection:
    def __init__(self, metric="cosine"):
        self.metadata = {"hnsw:space": metric}
        self.items = {}
        self.query_result = {"documents": [[]], "distances": [[]]}

    def add(self, ids, documents, metadatas):
        for ident, document, metadata in zip(ids, documents, metadatas):
            self.items[ident] = (document, metadata)

    def upsert(self, ids, documents, metadatas):
        self.add(ids, documents, metadatas)

    def delete(self, ids):
        for ident in ids:
            self.items.pop(ident, None)

    def get(self, ids=None, where=None, limit=None):
        selected = []
        for ident, (document, metadata) in self.items.items():
            if ids is not None and ident not in ids:
                continue
            if where and any(metadata.get(key) != value for key, value in where.items()):
                continue
            selected.append((ident, document, metadata))
        if limit is not None:
            selected = selected[:limit]
        return {
            "ids": [item[0] for item in selected],
            "documents": [item[1] for item in selected],
            "metadatas": [item[2] for item in selected],
        }

    def query(self, **kwargs):
        return self.query_result


class FakeChroma:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, FakeCollection())


class FakeMessages:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0) if self.outputs else "summary"
        return SimpleNamespace(content=[{"type": "text", "text": output}], usage=SimpleNamespace())


class FakeLlm:
    def __init__(self, outputs):
        self.messages = FakeMessages(outputs)
