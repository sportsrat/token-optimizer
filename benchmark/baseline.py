# """
# benchmark/baseline.py
# Executes raw LLM calls against Groq API to record unoptimized baseline performance.
# """

# import os
# import time
# import tiktoken
# from dotenv import load_dotenv
# from groq import Groq
# from benchmark.dataset import BENCHMARK_DATASET

# # Load environment variables from .env file
# load_dotenv()

# # Initialize API client and tokenizer
# groq_api_key = os.getenv("GROQ_API_KEY")
# if not groq_api_key:
#     raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")

# client = Groq(api_key=groq_api_key)
# tokenizer = tiktoken.get_encoding("cl100k_base")


# def count_tokens(text: str) -> int:
#     """Utility function to count tokens using cl100k_base encoding."""
#     return len(tokenizer.encode(text))

# # Change default model parameter from "llama-3.1-8b-instant" to "llama3-8b-8192"
# #def run_baseline_query(context: str, query: str, model: str = "llama3-8b-8192") -> dict:

# def run_baseline_query(context: str, query: str, model: str = "llama-3.3-70b-versatile") -> dict:
#     """Sends raw payload to Groq API and records latency and token consumption."""
#     messages = [
#         {"role": "system", "content": context},
#         {"role": "user", "content": query}
#     ]

#     full_prompt_str = f"{context}\n{query}"
#     input_tokens_local = count_tokens(full_prompt_str)

#     start_time = time.time()
#     response = client.chat.completions.create(
#         model=model,
#         messages=messages,
#         temperature=0.2
#     )
#     latency_ms = int((time.time() - start_time) * 1000)

#     response_text = response.choices[0].message.content
#     output_tokens_local = count_tokens(response_text)

#     # Extract API usage if reported, otherwise fallback to local count
#     prompt_tokens_api = getattr(response.usage, "prompt_tokens", input_tokens_local)
#     completion_tokens_api = getattr(response.usage, "completion_tokens", output_tokens_local)

#     return {
#         "text": response_text,
#         "input_tokens": prompt_tokens_api,
#         "output_tokens": completion_tokens_api,
#         "total_tokens": prompt_tokens_api + completion_tokens_api,
#         "latency_ms": latency_ms
#     }


# def execute_all_baselines():
#     """Executes all test cases in BENCHMARK_DATASET and prints baseline stats."""
#     print("==================================================")
#     print("          RUNNING UNOPTIMIZED BASELINES           ")
#     print("==================================================\n")

#     results = []
#     for test in BENCHMARK_DATASET:
#         print(f"Running: [{test['id']}]...")
#         res = run_baseline_query(context=test["context"], query=test["query"])
        
#         test_result = {
#             "id": test["id"],
#             "category": test["category"],
#             "input_tokens": res["input_tokens"],
#             "output_tokens": res["output_tokens"],
#             "total_tokens": res["total_tokens"],
#             "latency_ms": res["latency_ms"],
#             "response_preview": res["text"][:100].replace("\n", " ") + "..."
#         }
#         results.append(test_result)
        
#         print(f"  └─ Tokens: Input={res['input_tokens']} | Output={res['output_tokens']} | Total={res['total_tokens']}")
#         print(f"  └─ Latency: {res['latency_ms']} ms\n")

#     return results


# if __name__ == "__main__":
#     execute_all_baselines()


"""
benchmark/baseline.py
Executes raw LLM calls against Groq API to record unoptimized baseline performance.
"""

import os
import time
import tiktoken
from dotenv import load_dotenv
from groq import Groq
from benchmark.dataset import BENCHMARK_DATASET

# Load environment variables from .env file
load_dotenv()

# Initialize API client and tokenizer
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment. Please set it in your .env file.")

client = Groq(api_key=groq_api_key)
tokenizer = tiktoken.get_encoding("cl100k_base")

# Model is configurable via env var so future Groq deprecations don't
# require a code change. llama-3.3-70b-versatile was shut down 08/16/2026;
# openai/gpt-oss-120b and qwen/qwen3.6-27b are the recommended replacements.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Free-tier rate limit is ~30 requests/minute. If BENCHMARK_DATASET has
# more entries than that, this delay + retry logic keeps the run from
# dying on a 429 partway through.
REQUEST_DELAY_SECONDS = float(os.getenv("GROQ_REQUEST_DELAY", "0.5"))
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


def count_tokens(text: str) -> int:
    """Utility function to count tokens using cl100k_base encoding."""
    return len(tokenizer.encode(text))


def run_baseline_query(context: str, query: str, model: str = DEFAULT_MODEL) -> dict:
    """Sends raw payload to Groq API and records latency and token consumption.

    Retries on 429 (rate limit) with a fixed backoff, since the free tier
    caps at 30 requests/minute.
    """
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": query}
    ]

    full_prompt_str = f"{context}\n{query}"
    input_tokens_local = count_tokens(full_prompt_str)

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            start_time = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2
            )
            latency_ms = int((time.time() - start_time) * 1000)

            response_text = response.choices[0].message.content
            output_tokens_local = count_tokens(response_text)

            # Extract API usage if reported, otherwise fallback to local count
            prompt_tokens_api = getattr(response.usage, "prompt_tokens", input_tokens_local)
            completion_tokens_api = getattr(response.usage, "completion_tokens", output_tokens_local)

            return {
                "text": response_text,
                "input_tokens": prompt_tokens_api,
                "output_tokens": completion_tokens_api,
                "total_tokens": prompt_tokens_api + completion_tokens_api,
                "latency_ms": latency_ms
            }

        except Exception as e:
            last_error = e
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < MAX_RETRIES:
                print(f"  └─ Rate limited (attempt {attempt}/{MAX_RETRIES}), "
                      f"waiting {RETRY_BACKOFF_SECONDS}s...")
                time.sleep(RETRY_BACKOFF_SECONDS)
                continue
            raise last_error

    raise last_error


def execute_all_baselines():
    """Executes all test cases in BENCHMARK_DATASET and prints baseline stats."""
    print("==================================================")
    print("          RUNNING UNOPTIMIZED BASELINES           ")
    print("==================================================\n")

    results = []
    for test in BENCHMARK_DATASET:
        print(f"Running: [{test['id']}]...")
        res = run_baseline_query(context=test["context"], query=test["query"])

        test_result = {
            "id": test["id"],
            "category": test["category"],
            "input_tokens": res["input_tokens"],
            "output_tokens": res["output_tokens"],
            "total_tokens": res["total_tokens"],
            "latency_ms": res["latency_ms"],
            "response_preview": res["text"][:100].replace("\n", " ") + "..."
        }
        results.append(test_result)

        print(f"  └─ Tokens: Input={res['input_tokens']} | Output={res['output_tokens']} | Total={res['total_tokens']}")
        print(f"  └─ Latency: {res['latency_ms']} ms\n")

        # Stay under the free-tier 30 req/min cap
        time.sleep(REQUEST_DELAY_SECONDS)

    return results


if __name__ == "__main__":
    execute_all_baselines()