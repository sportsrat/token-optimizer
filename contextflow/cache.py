"""
contextflow/cache.py
Layer 1 (Exact SHA-256) and Layer 2 (Semantic Embedding) Cache Systems.
"""

import time
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer
from contextflow.utils import generate_l1_hash, cosine_similarity


class ExactCacheL1:
    """In-memory key-value store using SHA-256 hashes of exact request payloads."""

    def __init__(self, ttl: int = 86400):
        self._store = {}
        self.ttl = ttl

    def get(self, model: str, messages: list[dict], temperature: float):
        key = generate_l1_hash(model, messages, temperature)
        if key in self._store:
            entry = self._store[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["response"]
            else:
                del self._store[key]
        return None

    def set(self, model: str, messages: list[dict], temperature: float, response: dict):
        key = generate_l1_hash(model, messages, temperature)
        self._store[key] = {
            "timestamp": time.time(),
            "response": response
        }


class SemanticCacheL2:
    """In-memory vector cache matching queries by embedding cosine similarity."""

    def __init__(self, similarity_threshold: float = 0.85, ttl: int = 86400):
        self.similarity_threshold = similarity_threshold
        self.ttl = ttl
        self.embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        self._cache = []
        self.last_query_score = None  # Telemetry property for UI inspection

    def _hash_text(self, text: str) -> str:
        """Normalize whitespace and compute deterministic SHA-256 string hash."""
        normalized_text = " ".join(text.split())
        return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

    def _extract_query_and_context(self, messages: list[dict]):
        user_query = ""
        context_str = ""
        for m in messages:
            if m["role"] == "user":
                user_query = m["content"]
            elif m["role"] in ["system", "assistant"]:
                context_str += m["content"] + "\n"
        return context_str.strip(), user_query.strip()

    def get(self, messages: list[dict]):
        self.last_query_score = None  # Reset debug score on each query
        context_str, user_query = self._extract_query_and_context(messages)
        if not user_query:
            return None

        query_vector = self.embedder.encode(user_query, convert_to_numpy=True)
        now = time.time()

        best_match = None
        highest_score = -1.0
        current_ctx_hash = self._hash_text(context_str)

        for entry in self._cache:
            if now - entry["timestamp"] > self.ttl:
                continue

            if entry["context_hash"] != current_ctx_hash:
                continue

            sim_score = cosine_similarity(query_vector, entry["vector"])
            if sim_score > highest_score:
                highest_score = sim_score
                best_match = entry

        # Store debug score for telemetry
        self.last_query_score = round(highest_score, 4) if highest_score > -1 else None

        # Clean cache contract: Return dict on HIT, None on MISS
        if highest_score >= self.similarity_threshold and best_match:
            cached_res = dict(best_match["response"])
            cached_res["contextflow_meta"] = {
                "cache_hit": True,
                "cache_layer": "L2_SEMANTIC",
                "similarity_score": self.last_query_score
            }
            return cached_res

        return None

    def set(self, messages: list[dict], response: dict):
        context_str, user_query = self._extract_query_and_context(messages)
        if not user_query:
            return

        query_vector = self.embedder.encode(user_query, convert_to_numpy=True)
        self._cache.append({
            "timestamp": time.time(),
            "context_hash": self._hash_text(context_str),
            "query": user_query,
            "vector": query_vector,
            "response": response
        })