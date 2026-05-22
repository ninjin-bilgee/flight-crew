# test_ranker_service_real_pdfs.py
from ranker_service import build_index_from_folder, init_ranker, rank_resumes
import pdfplumber
import os

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
files = [f for f in os.listdir(RESUME_FOLDER) if os.path.isfile(os.path.join(RESUME_FOLDER, f))]
file_count = len(files)
num_ranked = input(f'How many top candidates would you like to see? (Please enter number 1-{file_count}) \n')

if(not num_ranked.isdigit()):
    print("Invalid input, displaying all ranks")
    num_ranked = file_count
if(int(num_ranked) > file_count):
    print("Input too big, displaying all ranks")
    num_ranked = file_count
elif(int(num_ranked) < 1):
    print("Input too small, displaying all ranks")
    num_ranked = file_count

results = rank_resumes(jd_text, k=int(num_ranked))

for r in results:
    score = r['similarity_score']  # e.g., 0.8476
    percentage = score * 100
    print(f"Rank {r['rank']}: {r['filename']} - Score: {percentage:.4f}%")