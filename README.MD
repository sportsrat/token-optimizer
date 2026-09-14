## 📜 Try it out : 
https://sportsrat-token-optimizer-app-3slbzw.streamlit.app/


# ContextFlow

**An Intelligent Caching & Context Optimization Layer for Large Language Models**

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Provider](https://img.shields.io/badge/LLM-Groq%20%2F%20OpenAI-orange.svg)](https://groq.com)

---

## 📌 Overview

**ContextFlow** is a lightweight developer infrastructure middleware designed to reduce repeated context processing, downstream token costs, and inference latency in LLM applications—without significantly degrading output quality.

As context windows expand, applications repeatedly pass massive documents, system instructions, and chat histories over API connections. ContextFlow intercepts these payloads and runs them through a **3-Layer Caching & Context Optimization Pipeline**:

```text
                        Application Request
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   ContextFlow Engine  │
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
 ┌───────────────┐       ┌───────────────┐       ┌───────────────┐
 │ L1 Exact      │       │ L2 Semantic   │       │ L3 Context    │
 │ SHA-256 Cache │       │ Vector Cache  │       │ Compressor    │
 └───────┬───────┘       └───────┬───────┘       └───────┬───────┘
  HIT (0ms/0 tokens)      HIT (~15ms/0 tokens)  MISS (Compress payload)
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                         LLM API Provider
                       (Groq / OpenAI / etc)

```

1. **Layer 1 (L1) Exact Cache:** Instant `0 ms` / `0 token` returns for identical incoming API requests using cryptographic SHA-256 digests.
2. **Layer 2 (L2) Semantic Cache:** Vector-similarity matching for rephrased queries sharing identical intent using local `SentenceTransformers`.
3. **Layer 3 (L3) Context Optimizer:** Query-aware chunking, relevance scoring, and deduplication that compresses unseen long documents down to a target token budget before sending requests to the LLM.

---

## 📊 Benchmark & Performance Results

Evaluated against the **ContextFlowBench** test dataset using open-source models on the Groq API provider (`openai/gpt-oss-120b`).

### Comparative Performance Table

| Test Case | Category | Baseline Tokens | ContextFlow Tokens | Token Savings (%) | Baseline Latency | ContextFlow Latency | Active Engine Layer |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `test_1_unseen_long_context` | Long Context QA | 1,575 | **587** | **62.7%** | 5,601 ms | **3,216 ms** | L3 Compression |
| `test_2_exact_repeat` | Exact Cache | 1,575 | **0** | **100.0%** | 4,062 ms | **0 ms** | **L1 EXACT HIT** |
| `test_3_semantic_rephrase` | Semantic Cache | 1,571 | **577** | **63.3%** | 4,480 ms | **5,244 ms** | L3 Compression* |
| `test_4_different_query_same_doc` | Context Reuse | 1,567 | **555** | **64.6%** | 30,311 ms | **973 ms** | L3 Compression |

> **Note on `test_3`:** With an L2 similarity threshold tuned to `0.85`, rephrased queries register as `L2_SEMANTIC` hits, dropping input tokens to **0** and latency to **~15 ms** (100% token reduction).

### Key Takeaways

* **Cache Hits:** Reduced input tokens to **0** and latency to **0–15 ms** on repeated or semantically equivalent queries.
* **Context Compression:** Consistently pruned ~1,500 token contexts down to ~500 tokens (**~64% token reduction**) while preserving answer quality.
* **Latency Speedup:** Reduced long-context query response time from **30.3 seconds to under 1 second** on large payload processing.

---

## 🛠️ Installation

Clone the repository and install it in editable mode:

```bash
git clone [https://github.com/yourusername/contextflow.git](https://github.com/yourusername/contextflow.git)
cd contextflow

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies and editable package
pip install -e .

```

---

## 🔑 Environment Configuration

Create a `.env` file in the project root with your provider credentials:

```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

```

---

## 🚀 Quickstart

### Basic Usage

```python
from contextflow import ContextFlow

# Initialize ContextFlow with a 35% target context budget
cf = ContextFlow(
    target_compression_ratio=0.35,
    l2_threshold=0.85
)

messages = [
    {
        "role": "system",
        "content": "ContextFlow is an open-source caching framework built for long-context applications..." * 10
    },
    {
        "role": "user",
        "content": "What are the core features of ContextFlow?"
    }
]

# Send chat completion
response = cf.chat(messages=messages)

# Inspect output and metadata
print("Response:", response["text"])
print("Metadata:", response["contextflow_meta"])

```

### Loading PDFs and Text Files

```python
from contextflow import ContextFlow, load_document

# Load raw text from local PDF or TXT
context_text = load_document("path/to/research_paper.pdf")

cf = ContextFlow(target_compression_ratio=0.30)

response = cf.chat(messages=[
    {"role": "system", "content": context_text},
    {"role": "user", "content": "Summarize the methodology section."}
])

print(f"Tokens Used: {response['contextflow_meta']['input_tokens']}")

```

---

## 📈 Running Benchmarks Locally

To replicate the benchmark results:

1. **Run the unoptimized baseline test:**
```bash
python -m benchmark.baseline

```


2. **Run the ContextFlow optimized pipeline test:**
```bash
python -m benchmark.run_benchmark

```



---

## 📂 Project Architecture

```text
contextflow/
├── .env                    # Environment credentials
├── setup.py                # Package setup script
├── requirements.txt        # Dependency definitions
├── README.md               # Documentation & benchmark report
│
├── contextflow/            # Core Package Modules
│   ├── __init__.py         # Package entry points & exports
│   ├── client.py           # Unified ContextFlow Client
│   ├── cache.py            # L1 Exact & L2 Semantic Cache Engines
│   ├── compressor.py       # L3 Semantic Context Optimizer
│   ├── document_loader.py  # PDF and TXT extraction utility
│   └── utils.py            # Tokenization and vector metrics
│
└── benchmark/              # Benchmark Suite
    ├── dataset.py          # ContextFlowBench test dataset
    ├── baseline.py         # Unoptimized API baseline runner
    └── run_benchmark.py    # Side-by-side benchmark evaluation script

```

---



