# Ranker Service: Embedding, Indexing, and Resume Ranking
"""
- Builds and loads FAISS indexes from resume embeddings
- Ranks resumes against a job description using a two-stage retrieve-then-rerank pipeline
- Converts raw cross-encoder logits into 0-1 display scores and flags weak batches
"""

import os
import glob
import numpy as np
from embedder import get_embedding, cross_encoder_score
from ranker import ResumeRanker
from text_extraction_engine import extract_resumes, conn

# module-level ranker instance, set by build_index_from_folder() or init_ranker()
_ranker = None
# embedding dimension for the current SBERT model (all-mpnet-base-v2 = 768)
_embedding_dim = 768

"""
Function:
- Extracts every resume PDF in a folder, embeds them, builds a FAISS index, and saves it to disk
- Intended as a one-shot setup step so the index can be reloaded later without re-processing
Returns: the built ResumeRanker instance
"""
def build_index_from_folder(resume_folder, index_save_path="faiss_resume_index"):
    """Process all resume PDFs in a folder, build FAISS index, and save it."""
    # collect all resume PDF paths in the folder
    print(f"Scanning {resume_folder} for resume PDFs...")
    resume_paths = glob.glob(os.path.join(resume_folder, "*.pdf"))
    if not resume_paths:
        raise ValueError(f"No PDF files found in {resume_folder}")

    # extract + anonymize + clean all resumes into the SQLite DB
    extract_resumes(resume_paths)

    # pull every stored candidate back out for embedding
    rows = conn.execute("SELECT candidate_id, filename, cleaned_text FROM candidates").fetchall()
    if not rows:
        raise ValueError("No valid resumes found after extraction.")

    # generate one embedding per candidate (cached to disk by candidate_id)
    print(f"Generating embeddings for {len(rows)} resumes...")
    resume_embeddings = []
    resume_metadata = []
    for candidate_id, filename, cleaned_text in rows:
        emb = get_embedding(cleaned_text, resume_id=candidate_id)
        resume_embeddings.append(emb)
        resume_metadata.append({'filename': filename, 'resume_id': candidate_id})

    # build the FAISS index from the embeddings and persist it to disk
    ranker = ResumeRanker(embedding_dim=_embedding_dim)
    ranker.build_index(resume_embeddings, resume_metadata)
    ranker.save_index(index_save_path)
    print(f"Index saved to {index_save_path}")
    return ranker

"""
Function:
- Loads a previously saved FAISS index from disk into the module-level _ranker
- Lets ranking run without re-extracting or re-embedding resumes
"""
def init_ranker(index_path="faiss_resume_index"):
    """Load an existing FAISS index into memory."""
    global _ranker
    _ranker = ResumeRanker()
    _ranker.load_index(index_path)
    print(f"Ranker loaded with {_ranker.index.ntotal} resumes.")


"""
Function: rank resumes against a job description

Pipeline:
    1. FAISS retrieves the full candidate pool.
    2. Cross-encoder re-scores each candidate by reading the JD + resume together.
    3. Sort by raw cross-encoder logit.
    4. Per-batch standardized sigmoid converts logits into 0-1 display scores.
    5. Absolute quality check flags batches where even the best match is weak.

Returns a dict:
    {
    'results':    list of {rank, filename, similarity_score},
    'weak_batch': bool — True if no strong match exists in the batch,
    'best_logit': float — highest raw cross-encoder logit in the batch,
    }
"""
def rank_resumes(job_description_text, k=10):
    # the ranker must be built or loaded before ranking can run
    if _ranker is None:
        raise RuntimeError("Ranker not initialized. Call init_ranker() or build_index_from_folder() first.")

    # Stage 1: FAISS retrieves all candidates as a pool
    # k=total so the whole candidate set is passed on to the cross-encoder
    total = _ranker.index.ntotal
    jd_embedding = get_embedding(job_description_text)
    pool = _ranker.search(jd_embedding, k=total)

    # Stage 2: cross-encoder re-scores each candidate by reading JD + resume together
    # this is the stage that actually determines ranking quality
    for r in pool:
        # look up the candidate's cleaned text by filename
        row = conn.execute(
            "SELECT cleaned_text FROM candidates WHERE filename = ?",
            (r['filename'],)
        ).fetchone()
        resume_text = row[0] if row else ""
        # cross-encoder returns a raw logit (unbounded, not yet a 0-1 score)
        r['ce_score'] = cross_encoder_score(job_description_text, resume_text)
        print(f"  {r['filename']}: raw logit = {r['ce_score']:.4f}")

    # Stage 3: sort by raw cross-encoder logit, highest first
    pool.sort(key=lambda x: x['ce_score'], reverse=True)

    # Absolute quality check — raw logits are the honest signal, before batch rescaling.
    # If even the best resume can't clear this cutoff, nobody is a real match.
    best_logit = max(r['ce_score'] for r in pool)
    weak_batch = best_logit < -5.5   # tune from test data

    # Stage 4: per-batch standardized sigmoid — consistent spread across all job types
    # converts unbounded logits into 0-1 display scores relative to this batch
    logits = np.array([r['ce_score'] for r in pool])
    mean = logits.mean()
    # guard against divide-by-zero if every logit is identical
    std = logits.std() if logits.std() > 0 else 1.0
    for r in pool:
        # z-score the logit, then squash through a sigmoid (2.0 widens the spread)
        z = (r['ce_score'] - mean) / std
        r['similarity_score'] = float(1 / (1 + np.exp(-z * 2.0)))

    # Assign ranks — 1-indexed, in sorted order, for the top k results
    for i, r in enumerate(pool[:k]):
        r['rank'] = i + 1

    # return the top k results plus the batch-level quality signals
    return {
        'results': pool[:k],
        'weak_batch': weak_batch,
        'best_logit': best_logit,
    }