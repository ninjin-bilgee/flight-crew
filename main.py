"""
Resume Ranking Pipeline: End-to-End CLI Runner

- Ingests a folder of resume PDFs and a single job description PDF
- Extracts, anonymizes, and cleans text, then stores it in a SQLite DB
- Generates SBERT embeddings, builds a FAISS index, and ranks resumes against the JD
- Prints ranked results and exports them to a downloadable CSV

HOW TO RUN THIS TEST:

Before running the test, make sure you delete these files with these commands:
rm candidate_map.db
rm -rf data/embeddings/

You need to delete both because embedder.py caches embeddings as .npy files in data/embeddings/.
If you delete the db but keep the cache, the candidate IDs won't match up anymore.

Usage:
    python main.py <resume_folder> <jd_pdf_path>

Example:
    python main.py data/resumes/ data/jd.pdf
"""
import sys
import os
import shutil
import atexit
import csv

# extract_resumes, extract_jd, and conn all live in text_extraction_engine
# importing conn here means run.py reuses the engine's DB connection — no second connection
from text_extraction_engine import extract_resumes, extract_jd, conn
from embedder import get_embedding
from ranker import ResumeRanker
import ranker_service

# require exactly 2 command-line arguments: the resume folder and the JD PDF path
if len(sys.argv) != 3:
    print("Usage: python run.py <resume_folder> <jd_pdf_path>")
    sys.exit(1)

# read the resume folder and JD path from the command line
RESUME_FOLDER = sys.argv[1]
JD_PATH = sys.argv[2]
# local SQLite DB file and the embedding cache folder
DB_PATH = "candidate_map.db"
EMBEDDINGS_PATH = "data/embeddings/"

"""
Function: session cleanup — runs automatically when the script exits
- closes the DB connection
- deletes the SQLite DB and the embedding cache so the next run starts fresh
- this prevents stale candidate IDs / cached embeddings from leaking between sessions
"""
def _cleanup():
    conn.close()
    # delete the SQLite DB if it exists
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("\nSession ended — candidate_map.db deleted.")
    # delete the embedding cache folder if it exists
    if os.path.exists(EMBEDDINGS_PATH):
        shutil.rmtree(EMBEDDINGS_PATH)
        print("Session ended — embeddings cache cleared.")

# register the cleanup to fire on normal interpreter exit
atexit.register(_cleanup)


# --- Main pipeline --- #

# Step 1: ingest all resumes and the job description into the DB
# extract_resumes takes a LIST of PDF paths, so glob the folder first
resume_pdf_paths = [
    os.path.join(RESUME_FOLDER, f)
    for f in os.listdir(RESUME_FOLDER)
    if f.endswith(".pdf")
]
if not resume_pdf_paths:
    print(f"No PDFs found in {RESUME_FOLDER}")
    sys.exit(1)

# delegate to the engine — handles staging, dedup, extraction, anonymization, and cleaning
extract_resumes(resume_pdf_paths)
extract_jd(JD_PATH)

# pull every stored candidate back out for embedding
rows = conn.execute("SELECT candidate_id, filename, cleaned_text FROM candidates").fetchall()
# stop if nothing was ingested
if not rows:
    print("No candidates found.")
    sys.exit(1)

print(f"\nFound {len(rows)} candidates.")

# filename -> cleaned text, used to show a resume snippet next to each result
snippets = {fn: ct for _, fn, ct in rows}

# Step 2: generate an embedding for each candidate
embeddings = []
metadata = []
for candidate_id, filename, cleaned_text in rows:
    print(f"Embedding {filename}...")
    # get_embedding caches by candidate_id; embeddings are 768-dim (all-mpnet-base-v2)
    emb = get_embedding(cleaned_text, resume_id=candidate_id)
    embeddings.append(emb)
    metadata.append({"filename": filename})

# Step 3: build the FAISS index from the candidate embeddings (768-dim)
ranker_service._ranker = ResumeRanker(embedding_dim=768)
ranker_service._ranker.build_index(embeddings, metadata)

# load the stored JD text to rank against
row = conn.execute("SELECT cleaned_text FROM job_descriptions LIMIT 1").fetchone()
if not row:
    print("No job description found.")
    sys.exit(1)

# ask the user how many top candidates to display
file_count = len(rows)
num_ranked = input(f'How many top candidates would you like to see? (Please enter number 1-{file_count}) \n')

# input validation — fall back to showing all candidates on any invalid entry
if not num_ranked.isdigit():
    print("Invalid input, displaying all ranks")
    num_ranked = file_count
elif int(num_ranked) > file_count:
    print("Input too big, displaying all ranks")
    num_ranked = file_count
elif int(num_ranked) < 1:
    print("Input too small, displaying all ranks")
    num_ranked = file_count

# Step 4: rank the resumes against the JD
# rank_resumes returns a dict: {'results': [...], 'weak_batch': bool, 'best_logit': float}
ranking = ranker_service.rank_resumes(row[0], k=int(num_ranked))
results = ranking['results']

print("\n=== RANKING RESULTS ===")
# if even the top candidate is a weak match, warn the user
if ranking['weak_batch']:
    print("⚠  WARNING: No strong matches found — even the top candidate is a weak fit.\n")

# print each ranked result with a short snippet of the resume so the user can see who it is
for r in results:
    snippet = snippets.get(r['filename'], '')[:120].strip()
    print(f"Rank {r['rank']}: {r['filename']} — Score: {r['similarity_score']:.4f}")
    print(f"    {snippet}...")

# Step 5: export the ranked results to a downloadable CSV
OUTPUT_CSV = "ranked_results.csv"
with open(OUTPUT_CSV, "w", newline="") as f:
    # CSV columns: rank, resume filename, and the 0-1 similarity score
    writer = csv.DictWriter(f, fieldnames=["rank", "filename", "similarity_score"])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "rank": r["rank"],
            "filename": r["filename"],
            "similarity_score": round(r["similarity_score"], 4),
        })

print(f"\nRanked results saved to {OUTPUT_CSV}")