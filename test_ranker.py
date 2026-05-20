import os
import glob
import pdfplumber
from embedder import get_embedding
from ranker import ResumeRanker


# Paths to documents
JD_PDF_PATH = "test_pdfs/jd/job_description.pdf"
RESUME_FOLDER = "test_pdfs/resume/"   # folder containing resume PDFs


# Helper: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# Load job description
print("Loading job description...")
jd_text = extract_text_from_pdf(JD_PDF_PATH)
if not jd_text:
    print("ERROR: Could not extract text from job description PDF.")
    exit(1)
print(f"JD text length: {len(jd_text)} characters\n")

# Load all resume PDFs
resume_paths = glob.glob(os.path.join(RESUME_FOLDER, "*.pdf"))
print(f"Found {len(resume_paths)} resume PDFs.")

resume_texts = []
resume_metadata = []

for path in resume_paths:
    text = extract_text_from_pdf(path)
    if text:
        resume_texts.append(text)
        resume_metadata.append({
            'filename': os.path.basename(path),
            'cleaned_text_path': path   # or where you store cleaned text later
        })
    else:
        print(f"Warning: No text extracted from {path} – skipping.")

if len(resume_texts) == 0:
    print("No usable resumes found. Exiting.")
    exit(1)

# Generate embeddings
print("Generating embeddings for resumes...")
resume_embeddings = [get_embedding(text) for text in resume_texts]

print("Generating embedding for job description...")
jd_embedding = get_embedding(jd_text)

# Build FAISS index and search
ranker = ResumeRanker(embedding_dim=384)
ranker.build_index(resume_embeddings, resume_metadata)

# Search for top matches
results = ranker.search(jd_embedding, k=min(4, len(resume_texts)))

# Display results
print("\n=== Ranking Results ===")
for r in results:
    print(f"Rank {r['rank']}: {r['filename']} - Similarity Score: {r['similarity_score']:.4f}")

# (Optional) Save index for later
ranker.save_index("faiss_resume_index")