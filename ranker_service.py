# ranking_service.py
import os
import glob
import sqlite3
from embedder import get_embedding
from ranker import ResumeRanker
from text_extraction_engine import extract_resumes, extract_jd, conn

_ranker = None
_embedding_dim = 384

def build_index_from_folder(resume_folder, index_save_path="faiss_resume_index"):
    """Process all resume PDFs in a folder, build FAISS index, and save it."""
    print(f"Scanning {resume_folder} for resume PDFs...")
    resume_paths = glob.glob(os.path.join(resume_folder, "*.pdf"))
    if not resume_paths:
        raise ValueError(f"No PDF files found in {resume_folder}")

    extract_resumes(resume_paths)

    rows = conn.execute("SELECT candidate_id, filename, cleaned_text FROM candidates").fetchall()
    if not rows:
        raise ValueError("No valid resumes found after extraction.")

    print(f"Generating embeddings for {len(rows)} resumes...")
    resume_embeddings = []
    resume_metadata = []
    for candidate_id, filename, cleaned_text in rows:
        emb = get_embedding(cleaned_text, resume_id=candidate_id)
        resume_embeddings.append(emb)
        resume_metadata.append({'filename': filename, 'resume_id': candidate_id})

    ranker = ResumeRanker(embedding_dim=_embedding_dim)
    ranker.build_index(resume_embeddings, resume_metadata)
    ranker.save_index(index_save_path)
    print(f"Index saved to {index_save_path}")
    return ranker

def init_ranker(index_path="faiss_resume_index"):
    """Load an existing FAISS index into memory."""
    global _ranker
    _ranker = ResumeRanker()
    _ranker.load_index(index_path)
    print(f"Ranker loaded with {_ranker.index.ntotal} resumes.")

def rank_resumes(job_description_text, k=10):
    """Rank resumes against a job description (text, not PDF)."""
    if _ranker is None:
        raise RuntimeError("Ranker not initialized. Call init_ranker() or build_index_from_folder() first.")
    jd_embedding = get_embedding(job_description_text)
    return _ranker.search(jd_embedding, k=k)
