# ranker_service.py
import os
import glob
import pdfplumber
from embedder import get_embedding
from ranker import ResumeRanker

_ranker = None
_embedding_dim = 384

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text.strip()

def build_index_from_folder(resume_folder, index_save_path="faiss_resume_index"):
    """Process all resume PDFs in a folder, build FAISS index, and save it."""
    print(f"Scanning {resume_folder} for resume PDFs...")
    resume_paths = glob.glob(os.path.join(resume_folder, "*.pdf"))
    if not resume_paths:
        raise ValueError(f"No PDF files found in {resume_folder}")

    resume_texts = []
    resume_metadata = []
    resume_ids = []

    for path in resume_paths:
        resume_id = os.path.splitext(os.path.basename(path))[0]
        text = extract_text_from_pdf(path)
        if text:
            resume_texts.append(text)
            resume_metadata.append({
                'filename': os.path.basename(path),
                'resume_id': resume_id,
                'path': path
            })
            resume_ids.append(resume_id)
        else:
            print(f"Warning: No text extracted from {path}")

    if not resume_texts:
        raise ValueError("No valid resumes with text found.")

    print(f"Generating embeddings for {len(resume_texts)} resumes...")
    resume_embeddings = []
    for i, text in enumerate(resume_texts):
        emb = get_embedding(text, resume_id=resume_ids[i])
        resume_embeddings.append(emb)

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
    
def rank_resumes(job_description_text, k):
    """Rank resumes against a job description (text, not PDF)."""
    if _ranker is None:
        raise RuntimeError("Ranker not initialized. Call init_ranker() or build_index_from_folder() first.")
    jd_embedding = get_embedding(job_description_text)
    return _ranker.search(jd_embedding, k=k)