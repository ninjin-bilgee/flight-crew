# Embedder module

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from pathlib import Path

_model = None
_cross_encoder = None

# Load model if not done
def _get_model():
    global _model
    if _model is None:
        #_model = SentenceTransformer('all-MiniLM-L6-v2')
        _model = SentenceTransformer('all-mpnet-base-v2')
    return _model

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder

def cross_encoder_score(jd_text, resume_text):
    """Score a single JD–resume pair with the cross-encoder. Returns a raw logit."""
    model = _get_cross_encoder()
    return float(model.predict([(jd_text, resume_text)])[0])

def get_embedding(text: str, resume_id: str = None) -> np.ndarray:
    """
    Returns 768-dim embedding for text.
    If resume_id is given, caches the embedding to disk.
    """
    # If resume_id provided, try to load from cache
    if resume_id is not None:
        cache_path = Path(f"data/embeddings/{resume_id}.npy")
        if cache_path.exists():
            # Load and return cached embedding
            return np.load(cache_path)

    # Compute embedding
    model = _get_model()
    embedding = model.encode(text)  # returns numpy array of shape (768,)

    # If resume_id provided, save to cache
    if resume_id is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists
        np.save(cache_path, embedding)

    return embedding