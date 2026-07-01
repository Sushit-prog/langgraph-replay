"""Thin wrapper around sentence-transformers for embedding and similarity.

Reusable by Phase 4 (semantic diffing) — import embed() and cosine_similarity() directly.

Model choice: all-MiniLM-L6-v2
- 384-dimensional, ~80MB, runs efficiently on CPU
- Good balance of quality vs speed for state comparison tasks
- Already used in langgraph_replay.search for session search
"""

import numpy as np

# Model name matches langgraph_replay.search to avoid duplicating the dependency
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None

# In-memory cache for embeddings within a single analysis run
# Key: text string, Value: numpy array
_embedding_cache: dict[str, np.ndarray] = {}


def _get_model():
    """Lazy load sentence-transformers model (CPU-only)."""
    global _model
    if _model is None:
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """Embed text into a vector, with caching for repeated calls.

    Returns a numpy array of shape (384,) for all-MiniLM-L6-v2.
    """
    if text in _embedding_cache:
        return _embedding_cache[text]

    model = _get_model()
    vec = model.encode(text, convert_to_numpy=True)
    _embedding_cache[text] = vec
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def clear_cache() -> None:
    """Clear the embedding cache. Call between separate analysis runs."""
    _embedding_cache.clear()
