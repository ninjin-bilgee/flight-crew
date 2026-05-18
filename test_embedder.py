# test_embedder.py
from embedder import get_embedding
import numpy as np

# First call – should compute and save
emb1 = get_embedding("Python developer", resume_id="test1")
print("First call shape:", emb1.shape)

# Second call – should load from cache (should be very fast)
emb2 = get_embedding("Python developer", resume_id="test1")
print("Are they equal?", np.allclose(emb1, emb2))

# Call without resume_id – no saving
emb3 = get_embedding("Job description", resume_id=None)
print("No cache shape:", emb3.shape)