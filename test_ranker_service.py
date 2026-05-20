# test_ranker_service_real_pdfs.py
from ranker_service import build_index_from_folder, init_ranker, rank_resumes
import pdfplumber

RESUME_FOLDER = "test_pdfs/resume/"
JD_PDF_PATH = "test_pdfs/jd/job_description.pdf"
INDEX_PATH = "faiss_resume_index"

# 1. Build index from resumes
build_index_from_folder(RESUME_FOLDER, INDEX_PATH)

# 2. Load the index (sets _ranker)
init_ranker(INDEX_PATH)

# 3. Load job description PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return text.strip()

jd_text = extract_text_from_pdf(JD_PDF_PATH)

# 4. Rank
results = rank_resumes(jd_text, k=5)

for r in results:
    print(f"Rank {r['rank']}: {r['filename']} - Score: {r['similarity_score']:.4f}")