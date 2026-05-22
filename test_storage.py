from storage import save_candidate, export_to_csv
from ranking_service import build_index_from_folder, init_ranker, rank_resumes

RESUME_FOLDER = "resumes_pdf/"
INDEX_PATH = "faiss_resume_index"
JD_TEXT = "Looking for a software engineer skilled in Python, machine learning, APIs, React, and AWS."

build_index_from_folder(RESUME_FOLDER, INDEX_PATH)
init_ranker(INDEX_PATH)

results = rank_resumes(JD_TEXT, k=5)

for r in results:
    candidate_id = f"Candidate_{r['rank']}"
    save_candidate(
        candidate_id=candidate_id,
        filename=r["filename"],
        score=r["score"],
        skills=[]
    )

export_to_csv()
print("Done!")

from storage import save_candidate, export_to_csv
from ranking_service import build_index_from_folder, init_ranker, rank_resumes

RESUME_FOLDER = "resumes_pdf/"
INDEX_PATH = "faiss_resume_index"
JD_TEXT = "Looking for a software engineer skilled in Python, machine learning, APIs, React, and AWS."

build_index_from_folder(RESUME_FOLDER, INDEX_PATH)
init_ranker(INDEX_PATH)

results = rank_resumes(JD_TEXT, k=5)

for r in results:
    candidate_id = f"Candidate_{r['rank']}"
    save_candidate(
        candidate_id=candidate_id,
        filename=r["filename"],
        score=r["score"],
        skills=[]
    )

export_to_csv()
print("Done!")
