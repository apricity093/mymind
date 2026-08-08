# Python Cache and Memory Offline Experiment

- captured_at: 2026-08-08T09:12:05.903415+00:00
- git_commit: aeb4dee6bb753f49d7b8cea1bbde28a21d1faa75
- git_dirty: True
- source_hash: a665a0fad058831db4e2a51b55bd998d93496491eab681696fe64b40942ec10d
- overall_passed: True

## Variants

### B0
- intent_wrong_cache_hits: 40
- knowledge_stale_hits: 30
- max_context_chars: 24409
- profile_llm_calls: 50

### C1
- intent_wrong_cache_hits: 0
- knowledge_stale_hits: 0
- profile_llm_calls: 10
- profile_call_reduction: 0.8

### C2
- precision_at_3: 1.0
- recall_at_3: 1.0
- irrelevant_injection_rate: 0.0
- duplicate_injection_count: 0
- max_context_chars: 7993
- current_request_preserved_rate: 1.0
- context_compression_ratio: 0.6725
- cache_hit_p95_ns: 20200
- cache_miss_p95_ns: 16123100
- cache_p95_reduction: 0.9987

### C3
- prompt_cache_status: not_applicable
- stable_prefix_chars: 27
- minimum_chars: 4096

## Failures

- none
