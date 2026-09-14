"""
benchmark/dataset.py
Test cases for evaluating ContextFlow compression, caching, and quality retention.
"""

# Sample long-context research document for compression tests (~1,500+ tokens when duplicated)
LONG_CONTEXT_DOC = """
ContextFlow is an intelligent developer infrastructure middleware designed to optimize long-context interactions with Large Language Models (LLMs).
As LLM applications scale, developers face exponential token costs, high latency, and redundant context processing.
ContextFlow addresses these challenges through a three-layer caching architecture combined with an adaptive context compression engine.

Layer 1 (L1) Exact Cache intercepts identical incoming API requests using cryptographic SHA-256 digests over the model name, temperature, and message history.
When an exact match occurs, L1 immediately returns the stored output with near-zero millisecond latency and zero downstream token consumption.

Layer 2 (L2) Semantic Cache handles queries that share identical intent but differ in phrasing or syntax.
By generating vector embeddings of incoming prompts using lightweight open-source models, L2 computes cosine similarity against historical queries.
If the similarity score exceeds a specified safety threshold (e.g., 0.95), the cached response is served directly.

Layer 3 (L3) Context Cache stores pre-processed representations of reusable background documents, such as PDFs, codebases, or database extracts.
Instead of sending 50,000+ tokens repeatedly across a multi-turn session, L3 preserves chunked vector indexes.

When a query results in an L1/L2 miss, the Adaptive Context Optimizer pipeline engages.
First, the input document is segmented into semantic paragraphs or sentences.
Second, a sentence-transformer model calculates relevance scores for each chunk relative to the user query.
Third, a deduplication step eliminates redundant information across top-scoring chunks.
Finally, budget allocation logic selects the highest-value content to fit within a targeted compression ratio (e.g., retaining 30% of original tokens).

Evaluation of ContextFlow uses the ContextFlowBench dataset across metrics including Token Savings Percentage, Cost Reduction, Latency Savings, and Semantic Quality Retention.
""" * 4  # Expanded for benchmark payload size

BENCHMARK_DATASET = [
    {
        "id": "test_1_unseen_long_context",
        "category": "long_context_qa",
        "description": "First run of long context QA (Expects Cache MISS, triggers Context Optimizer compression)",
        "context": LONG_CONTEXT_DOC,
        "query": "How does the Layer 2 (L2) Semantic Cache work in ContextFlow?"
    },
    {
        "id": "test_2_exact_repeat",
        "category": "exact_cache",
        "description": "Identical prompt resubmission (Expects L1 Cache HIT)",
        "context": LONG_CONTEXT_DOC,
        "query": "How does the Layer 2 (L2) Semantic Cache work in ContextFlow?"
    },
    {
        "id": "test_3_semantic_rephrase",
        "category": "semantic_cache",
        "description": "Rephrased query with identical intent (Expects L2 Cache HIT)",
        "context": LONG_CONTEXT_DOC,
        "query": "Can you explain how ContextFlow's L2 semantic caching operates?"
    },
    {
        "id": "test_4_different_query_same_doc",
        "category": "context_compression",
        "description": "Different question on same document (Expects Cache MISS, tests relevance scoring)",
        "context": LONG_CONTEXT_DOC,
        "query": "What metrics are measured in ContextFlowBench?"
    }
]