"""Utility function to retrieve similar Math-500 examples using sentence embeddings."""

from typing import List, Tuple, Optional

import numpy as np
from datasets import Dataset
from sentence_transformers import SentenceTransformer


_ENCODER_CACHE = {}


def get_encoder(model_name: str) -> SentenceTransformer:
    """Lazy-load a sentence transformer encoder."""
    if model_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[model_name] = SentenceTransformer(model_name)
    return _ENCODER_CACHE[model_name]


def embed_texts(texts: List[str], encoder: SentenceTransformer) -> np.ndarray:
    """Encode a list of texts into normalized vectors."""
    embeddings = encoder.encode(texts, batch_size=32, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def retrieve_topk(
    query_text: str,
    example_pool: Dataset,
    top_k: int = 10,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    encoder: Optional[SentenceTransformer] = None,
) -> List[Tuple[float, int, dict]]:
    """
    Return the top-k examples sorted by semantic similarity.
    Each tuple is (similarity_score, rank_idx, example_dict) where rank_idx preserves
    the ordering of example_pool (e.g., curriculum ranking).
    """
    encoder = encoder or get_encoder(model_name)

    query_vec = embed_texts([query_text], encoder)[0]

    candidate_texts = []
    examples = []
    for idx, example in enumerate(example_pool):
        candidate_text = example.get("problem", example.get("question", ""))
        candidate_texts.append(candidate_text)
        examples.append((idx, example))

    if not candidate_texts:
        return []

    cand_vecs = embed_texts(candidate_texts, encoder)
    sims = cand_vecs @ query_vec  # cosine similarity because embeddings are normalized

    scored = sorted(
        [
            (score, idx_example[0], idx_example[1])
            for score, idx_example in zip(sims, examples)
        ],
        key=lambda x: x[0],
        reverse=True,
    )
    return scored[:top_k]

