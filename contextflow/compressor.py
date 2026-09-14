"""
contextflow/compressor.py
Layer 3 (L3) Context Optimizer: Chunking, Relevance Scoring, and Token Budget Compression.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from contextflow.utils import count_tokens, cosine_similarity


class ContextCompressorL3:
    """Semantic context compression engine."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Shared open-source lightweight embedding model (~80MB)
        self.embedder = SentenceTransformer(model_name)

    def _chunk_text(self, text: str, max_chunk_tokens: int = 150) -> list[str]:
        """Splits context text into logical paragraphs or sentences."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        
        for p in paragraphs:
            if count_tokens(p) > max_chunk_tokens:
                sentences = p.split(". ")
                curr_chunk = ""
                for s in sentences:
                    s_fmt = s.strip() + "." if not s.endswith(".") else s.strip()
                    if count_tokens(curr_chunk + " " + s_fmt) <= max_chunk_tokens:
                        curr_chunk += " " + s_fmt
                    else:
                        if curr_chunk.strip():
                            chunks.append(curr_chunk.strip())
                        curr_chunk = s_fmt
                if curr_chunk.strip():
                    chunks.append(curr_chunk.strip())
            else:
                chunks.append(p)

        return chunks if chunks else [text]

    def compress(self, context: str, query: str, target_ratio: float = 0.35) -> tuple[str, dict]:
        """
        Compresses input context to fit target_ratio budget based on semantic query relevance.
        """
        original_tokens = count_tokens(context)
        query_tokens = count_tokens(query)
        total_orig = original_tokens + query_tokens

        # Bypass compression if prompt is small or budget ratio is 100%
        if original_tokens < 300 or target_ratio >= 1.0:
            return context, {
                "original_tokens": total_orig,
                "compressed_tokens": total_orig,
                "compression_ratio": 1.0,
                "compressed": False
            }

        chunks = self._chunk_text(context)
        if len(chunks) <= 1:
            return context, {
                "original_tokens": total_orig,
                "compressed_tokens": total_orig,
                "compression_ratio": 1.0,
                "compressed": False
            }

        # Embed query and chunks locally
        query_vec = self.embedder.encode(query, convert_to_numpy=True)
        chunk_vecs = self.embedder.encode(chunks, convert_to_numpy=True)

        # Calculate semantic relevance score for each chunk
        scores = [cosine_similarity(chunk_vecs[i], query_vec) for i in range(len(chunks))]

        # Token budget calculation
        target_context_budget = max(int(original_tokens * target_ratio), 200)

        # Rank chunks by score descending
        ranked_indices = np.argsort(scores)[::-1]

        selected_chunks = []
        accumulated_tokens = 0

        for idx in ranked_indices:
            c_tok = count_tokens(chunks[idx])
            if accumulated_tokens + c_tok <= target_context_budget:
                selected_chunks.append((idx, chunks[idx]))
                accumulated_tokens += c_tok

        # Re-sort chronologically to preserve original reading flow
        selected_chunks.sort(key=lambda x: x[0])
        compressed_context = "\n\n".join([item[1] for item in selected_chunks])

        compressed_total = count_tokens(compressed_context) + query_tokens
        actual_ratio = round(compressed_total / max(total_orig, 1), 3)

        meta = {
            "original_tokens": total_orig,
            "compressed_tokens": compressed_total,
            "compression_ratio": actual_ratio,
            "tokens_saved": total_orig - compressed_total,
            "compressed": True
        }

        return compressed_context, meta