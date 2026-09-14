"""
contextflow/client.py
Unified ContextFlow Client routing requests across L1, L2, and L3 subsystems.
"""

import os
import time
from dotenv import load_dotenv
from groq import Groq

from contextflow.cache import ExactCacheL1, SemanticCacheL2
from contextflow.compressor import ContextCompressorL3
from contextflow.utils import count_tokens

load_dotenv()

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5


class ContextFlow:
    def __init__(
        self,
        groq_api_key: str = None,
        model: str = DEFAULT_MODEL,
        enable_l1: bool = True,
        enable_l2: bool = True,
        enable_l3: bool = True,
        target_compression_ratio: float = 0.35,
        l2_threshold: float = 0.85
    ):
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is required.")

        self.client = Groq(api_key=api_key)
        self.model = model
        self.enable_l1 = enable_l1
        self.enable_l2 = enable_l2
        self.enable_l3 = enable_l3
        self.target_ratio = target_compression_ratio

        # Initialize engine modules
        self.l1_cache = ExactCacheL1() if enable_l1 else None
        self.l2_cache = SemanticCacheL2(similarity_threshold=l2_threshold) if enable_l2 else None
        self.l3_compressor = ContextCompressorL3() if enable_l3 else None

    def chat(self, messages: list[dict], temperature: float = 0.2) -> dict:
        start_time = time.time()

        # Update dynamic L2 threshold if modified
        if self.l2_cache and hasattr(self, 'l2_threshold'):
            self.l2_cache.similarity_threshold = self.l2_threshold

        # ----------------------------------------------------
        # 1. LAYER 1: EXACT CACHE CHECK (SHA-256)
        # ----------------------------------------------------
        if self.enable_l1:
            l1_hit = self.l1_cache.get(self.model, messages, temperature)
            if l1_hit:
                latency_ms = int((time.time() - start_time) * 1000)
                res = dict(l1_hit)
                res["contextflow_meta"] = {
                    "cache_hit": True,
                    "cache_layer": "L1_EXACT",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "latency_ms": latency_ms
                }
                return res

        # ----------------------------------------------------
        # 2. LAYER 2: SEMANTIC CACHE CHECK (Vector Similarity)
        # ----------------------------------------------------
        if self.enable_l2:
            l2_hit = self.l2_cache.get(messages)
            if l2_hit:
                latency_ms = int((time.time() - start_time) * 1000)
                l2_hit["contextflow_meta"]["latency_ms"] = latency_ms
                l2_hit["contextflow_meta"]["input_tokens"] = 0
                l2_hit["contextflow_meta"]["output_tokens"] = 0
                l2_hit["contextflow_meta"]["total_tokens"] = 0
                return l2_hit

        # Retrieve debug score for telemetry display
        l2_attempted_score = (
            self.l2_cache.last_query_score 
            if (self.enable_l2 and self.l2_cache.last_query_score is not None) 
            else "N/A"
        )

        # ----------------------------------------------------
        # 3. LAYER 3: CONTEXT COMPRESSION (If Cache Miss)
        # ----------------------------------------------------
        optimized_messages = list(messages)
        compression_meta = {"compressed": False, "compression_ratio": 1.0}

        if self.enable_l3 and len(messages) >= 2:
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            user_msg = next((m for m in messages if m["role"] == "user"), None)

            if system_msg and user_msg:
                compressed_context, compression_meta = self.l3_compressor.compress(
                    context=system_msg["content"],
                    query=user_msg["content"],
                    target_ratio=self.target_ratio
                )

                optimized_messages = [
                    {"role": "system", "content": compressed_context},
                    {"role": "user", "content": user_msg["content"]}
                ]

        # ----------------------------------------------------
        # 4. DOWNSTREAM LLM INFERENCE (GROQ API WITH RETRY)
        # ----------------------------------------------------
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=optimized_messages,
                    temperature=temperature
                )
                break
            except Exception as e:
                last_error = e
                is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
                if is_rate_limit and attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS)
                    continue
                raise last_error

        latency_ms = int((time.time() - start_time) * 1000)
        response_text = response.choices[0].message.content
        prompt_tokens = getattr(response.usage, "prompt_tokens", count_tokens(str(optimized_messages)))
        completion_tokens = getattr(response.usage, "completion_tokens", count_tokens(response_text))

        output_data = {
            "text": response_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            "contextflow_meta": {
                "cache_hit": False,
                "cache_layer": "NONE (LLM INFERENCE)",
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "latency_ms": latency_ms,
                "l2_attempted_score": l2_attempted_score,
                "compression_stats": compression_meta
            }
        }

        # ----------------------------------------------------
        # 5. STORE RESPONSE IN L1 & L2 CACHES
        # ----------------------------------------------------
        if self.enable_l1:
            self.l1_cache.set(self.model, messages, temperature, output_data)

        if self.enable_l2:
            self.l2_cache.set(messages, output_data)

        return output_data