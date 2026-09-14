"""
contextflow/utils.py
Utility functions for cryptographic key generation, token counting, and vector operations.
"""

import hashlib
import json
import numpy as np
import tiktoken

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Counts tokens using cl100k_base encoding."""
    if not text:
        return 0
    return len(tokenizer.encode(text))


def generate_l1_hash(model: str, messages: list[dict], temperature: float) -> str:
    """Generates a canonical SHA-256 hash for exact request caching."""
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Computes cosine similarity between two 1D embedding vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return float(dot_product / (norm_vec1 * norm_vec2))