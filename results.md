# ContextFlow Baseline Results (Unoptimized)

| Test ID | Category | Input Tokens | Output Tokens | Total Tokens | Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `test_1_unseen_long_context` | long_context_qa | 1,575 | 2,368 | 3,943 | 5,601 ms |
| `test_2_exact_repeat` | exact_cache | 1,575 | 1,675 | 3,250 | 4,062 ms |
| `test_3_semantic_rephrase` | semantic_cache | 1,571 | 1,661 | 3,232 | 4,480 ms |
| `test_4_different_query_same_doc` | context_compression | 1,567 | 319 | 1,886 | 30,311 ms |