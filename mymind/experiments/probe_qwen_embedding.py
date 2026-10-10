"""用公开样例检查 embedding 地址，输出不含凭证的真实响应契约。"""
import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]


def probe(base_url, output):
    config = dotenv_values(ROOT / ".env")
    key = config.get("EMBEDDING_API_KEY") or config.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("embedding_key_missing")
    model = "qwen3.7-text-embedding-flash"
    result = {"date": datetime.now().isoformat(timespec="seconds"), "model": model,
              "dimensions": 768, "base_url": base_url, "input_scope": "public_synthetic_connectivity_samples",
              "status": "failed", "attempts": []}
    payload = {"model": model, "input": ["公开测试：如何查询订单状态？", "公开测试：退款需要订单号。"],
               "dimensions": 768, "encoding_format": "float"}
    with httpx.Client(timeout=20) as client:
        for suffix in ["/embeddings", "/v1/embeddings"]:
            url = base_url.rstrip("/") + suffix
            row = {"url": url, "protocol": "openai_embeddings", "input_texts": 2}
            started = time.perf_counter()
            try:
                response = client.post(url, headers={"Authorization": f"Bearer {key}"}, json=payload)
                row["http_status"] = response.status_code
                row["content_type"] = response.headers.get("content-type", "")
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                    row["response_type"] = "non_json"
                if response.is_success:
                    data = body.get("data", []) if isinstance(body, dict) else []
                    vectors = [item.get("embedding") for item in data if isinstance(item, dict)]
                    valid = len(vectors) == 2 and all(isinstance(v, list) and len(v) == 768 and
                        all(isinstance(x, (float, int)) and math.isfinite(x) for x in v) and
                        sum(x*x for x in v) > 0 for v in vectors)
                    row.update(valid_embeddings=valid, returned_model=body.get("model"),
                               vector_dimensions=[len(v) if isinstance(v, list) else None for v in vectors],
                               usage=body.get("usage"))
                    if valid:
                        result.update(status="available", endpoint=url)
                else:
                    error = body.get("error", {}) if isinstance(body, dict) else {}
                    if isinstance(error, dict):
                        row["error_code"] = str(error.get("code", ""))[:100].replace(key, "[redacted]")
                        row["error_type"] = str(error.get("type", ""))[:100].replace(key, "[redacted]")
                        row["message"] = str(error.get("message", ""))[:300].replace(key, "[redacted]")
                    if "response_type" in row:
                        row["message"] = "endpoint_returned_non_json_error"
            except httpx.HTTPError as ex:
                row["error_type"] = type(ex).__name__
            row["wall_ms"] = round((time.perf_counter()-started)*1000, 2)
            result["attempts"].append(row)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if result["status"] == "available":
                break
    return result["status"] == "available"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://maas.qianwenaiapi.com/compatible-mode/v1")
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/qwen-embedding-migration/availability.json")
    args = parser.parse_args()
    raise SystemExit(0 if probe(args.base_url, args.output) else 1)


if __name__ == "__main__":
    main()
