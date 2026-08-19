from __future__ import annotations

import argparse
from pathlib import Path

from experiments.offline import run_offline


def main():
    parser = argparse.ArgumentParser(description="Run Python cache/memory experiments")
    parser.add_argument("--layer", choices=("offline", "integration", "real", "rag"), default="offline")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--redis-url", default="redis://:mymind123@localhost:6379/15")
    parser.add_argument("--chroma-host", default="localhost")
    parser.add_argument("--chroma-port", type=int, default=8001)
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--provider", choices=("deepseek", "openai", "anthropic"))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--cache-scenario", choices=("stable-prefix", "identical", "invalidation"), default="stable-prefix")
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rewrite-failure-rate", type=float, default=0.02)
    parser.add_argument("--rerank-failure-rate", type=float, default=0.02)
    args = parser.parse_args()
    if args.layer == "offline":
        report = run_offline(args.output_dir)
    elif args.layer == "rag":
        from experiments.rag_retrieval import run_rag_offline
        report = run_rag_offline(
            args.output_dir,
            variants=args.variants,
            rewrite_failure_rate=args.rewrite_failure_rate,
            rerank_failure_rate=args.rerank_failure_rate,
            top_k=args.top_k,
        )
    elif args.layer == "integration":
        import asyncio
        from experiments.integration import run_integration
        report = asyncio.run(run_integration(
            args.redis_url, args.chroma_host, args.chroma_port, args.output_dir
        ))
    else:
        import asyncio
        from experiments.real_model import run_real
        report = asyncio.run(run_real(
            args.output_dir, args.confirm_cost, args.provider, args.repeat, args.cache_scenario
        ))
    print(report["artifacts"]["json"])
    raise SystemExit(0 if report["overall_passed"] else 1)


if __name__ == "__main__":
    main()
