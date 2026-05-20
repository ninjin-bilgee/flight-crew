import os
import glob
import pdfplumber
from embedder import get_embedding
from ranker import ResumeRanker

# Configuration
RESUME_FOLDER = "test_pdfs/resume/"
JD_PDF_PATH = "test_pdfs/jd/job_description.pdf"
INDEX_PATH = "faiss_resume_index"          # where to save/load index

# Helper: extract text from PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text.strip()

# 1. Process all resumes
print("Scanning resume folder...")
resume_paths = glob.glob(os.path.join(RESUME_FOLDER, "*.pdf"))
print(f"Found {len(resume_paths)} resumes.")

resume_texts = []
resume_metadata = []
resume_ids = []

for path in resume_paths:
    # Use file name without extension as a simple resume_id
    resume_id = os.path.splitext(os.path.basename(path))[0]
    text = extract_text_from_pdf(path)
    if text:
        resume_texts.append(text)
        resume_metadata.append({
            'filename': os.path.basename(path),
            'resume_id': resume_id,
            'cleaned_text_path': path
        })
        resume_ids.append(resume_id)
    else:
        print(f"Warning: No text extracted from {path} – skipping.")

if not resume_texts:
    print("No valid resumes found. Exiting.")
    exit(1)

# 2. Generate embeddings with caching
print("Generating resume embeddings...")
resume_embeddings = []
for i, text in enumerate(resume_texts):
    resume_id = resume_ids[i]
    emb = get_embedding(text, resume_id=resume_id)   # caches to .npy
    resume_embeddings.append(emb)

# 3. Build FAISS index
ranker = ResumeRanker(embedding_dim=384)
ranker.build_index(resume_embeddings, resume_metadata)
ranker.save_index(INDEX_PATH)
print(f"Index saved to {INDEX_PATH}")

# 4. Process job description
print("\nProcessing job description...")
jd_text = extract_text_from_pdf(JD_PDF_PATH)
if not jd_text:
    print("ERROR: Could not extract text from job description PDF.")
    exit(1)
jd_embedding = get_embedding(jd_text)   # no resume_id (no caching)

# 5. Search and rank
k = min(5, len(resume_texts))
results = ranker.search(jd_embedding, k=k)

print("\n=== Ranking Results ===")
for r in results:
    print(f"Rank {r['rank']}: {r['filename']} - Similarity Score: {r['similarity_score']:.4f}")