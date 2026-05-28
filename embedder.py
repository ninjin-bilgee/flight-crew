# Embedder Module: SBERT Embeddings and Cross-Encoder Scoring
"""
- Loads and caches the SBERT bi-encoder (all-mpnet-base-v2) for generating text embeddings
- Loads and caches the cross-encoder (ms-marco-MiniLM-L-6-v2) for JD-resume pair scoring
- Caches resume embeddings to disk as .npy files to avoid recomputation
"""

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

_model = None
_cross_encoder = None

"""
Function: lazily loads the SBERT bi-encoder model
- loads the model on first call, reuses the cached instance afterwards
- all-mpnet-base-v2 produces 768-dimensional embeddings
"""
def _get_model():
    global _model
    if _model is None:
        #_model = SentenceTransformer('all-MiniLM-L6-v2') 
        _model = SentenceTransformer('all-mpnet-base-v2')
    return _model

"""
Function: lazily loads the cross-encoder model
- loads the model on first call, reuses the cached instance afterwards
- ms-marco-MiniLM-L-6-v2 scores a JD-resume pair read together
"""
def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def cross_encoder_score(jd_text, resume_text):
    """Score a single JD–resume pair with the cross-encoder. Returns a raw logit."""
    # the cross-encoder reads the JD and resume together and outputs one relevance logit
    model = _get_cross_encoder()
    return float(model.predict([(jd_text, resume_text)])[0])

def get_embedding(text: str, resume_id: str = None) -> np.ndarray:
    """
    Returns 768-dim embedding for text.
    If resume_id is given, caches the embedding to disk.
    """
    # If resume_id provided, try to load from cache
    # NOTE: cache key is session-scoped resume_id — relies on cache being cleared between runs
    if resume_id is not None:
        cache_path = Path(f"data/embeddings/{resume_id}.npy")
        # if a cached embedding exists for this resume, load and return it
        if cache_path.exists():
            # Load and return cached embedding
            return np.load(cache_path)

    # Compute embedding
    # no cache hit (or no resume_id) — encode the text with the SBERT model
    model = _get_model()
    embedding = model.encode(text)  # returns numpy array of shape (768,)

    # If resume_id provided, save to cache
    if resume_id is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)  # ensures folder exists
        np.save(cache_path, embedding)

    return embedding