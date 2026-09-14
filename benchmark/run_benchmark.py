"""
benchmark/run_benchmark.py
Runs the ContextFlow benchmark suite and evaluates token savings, latency, and quality.
"""

import json
from contextflow.client import ContextFlow
from benchmark.dataset import BENCHMARK_DATASET


def run_full_benchmark():
    print("==================================================")
    print("          RUNNING CONTEXTFLOW BENCHMARK           ")
    print("==================================================\n")

    # Initialize ContextFlow with target 35% token compression ratio
    cf = ContextFlow(target_compression_ratio=0.35)

    results = []
    for test in BENCHMARK_DATASET:
        print(f"Running ContextFlow: [{test['id']}]...")

        messages = [
            {"role": "system", "content": test["context"]},
            {"role": "user", "content": test["query"]}
        ]

        res = cf.chat(messages=messages)
        meta = res["contextflow_meta"]

        print(f"  ├─ Cache Status : {meta['cache_layer']} (Hit: {meta['cache_hit']})")
        print(f"  ├─ Input Tokens : {meta['input_tokens']}")
        print(f"  ├─ Latency      : {meta['latency_ms']} ms")
        if meta.get("compression_stats", {}).get("compressed"):
            c_meta = meta["compression_stats"]
            print(f"  └─ L3 Compression: {c_meta['original_tokens']} -> {c_meta['compressed_tokens']} tokens (Ratio: {c_meta['compression_ratio']})")
        print()

        results.append({
            "id": test["id"],
            "cache_layer": meta["cache_layer"],
            "input_tokens": meta["input_tokens"],
            "latency_ms": meta["latency_ms"]
        })

    return results


if __name__ == "__main__":
    run_full_benchmark()