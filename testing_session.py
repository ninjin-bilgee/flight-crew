"""
HOW TO RUN THIS TEST:

Before running the test, make sure you delete these files with these commands:
rm candidate_map.db
rm -rf data/embeddings/

You need to delete both because embedder.py caches embeddings as .npy files in data/embeddings/.
If you delete the db but keep the cache, the candidate IDs won't match up anymore.

HOWEVER, if you are adding new resumes to an existing batch, just do steps 1-3 directly.

1. Extract resumes from your own folder:
   python cli.py upload_resumes --folder /your/path/to/resumes

   or you can upload a single resume with:
   python cli.py upload_resume /your/path/to/resume.pdf

2. Upload the job description:
   python cli.py upload_jd /path/to/jd.pdf

3. Run the pipeline:
   python testing_session.py
"""
import sqlite3
import sys
import ranker_service
from embedder import get_embedding
from ranker import ResumeRanker
import atexit
import shutil
import os

# Connect to database
conn = sqlite3.connect("candidate_map.db")

# Cleanup runs automatically when script finishes
DB_PATH = "candidate_map.db"
EMBEDDINGS_PATH = "data/embeddings/"

def _cleanup():
    conn.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("\nSession ended — candidate_map.db deleted.")
    if os.path.exists(EMBEDDINGS_PATH):
        shutil.rmtree(EMBEDDINGS_PATH)
        print("Session ended — embeddings cache cleared.")

atexit.register(_cleanup)

# Check if candidates exist
rows = conn.execute("SELECT candidate_id, filename, cleaned_text FROM candidates").fetchall()

if not rows:
    print("No candidates found in database.")
    print("Run this first:")
    print("  python cli.py upload_resumes --folder /path/to/resumes")
    sys.exit(1)

print(f"Found {len(rows)} candidates.")

# Generate embeddings
embeddings = []
metadata = []
for candidate_id, filename, cleaned_text in rows:
    print(f"Embedding {filename}...")
    emb = get_embedding(cleaned_text, resume_id=candidate_id)
    embeddings.append(emb)
    metadata.append({"filename": filename})

# Build FAISS index via ranker_service
ranker_service._ranker = ResumeRanker(embedding_dim=768)
ranker_service._ranker.build_index(embeddings, metadata)

# Read JD from SQLite DB
row = conn.execute("SELECT cleaned_text FROM job_descriptions LIMIT 1").fetchone()
if not row:
    print("No job description found. Run:")
    print("  python cli.py upload_jd /path/to/jd.pdf")
    sys.exit(1)

# Ask how many results to display
file_count = len(rows)
num_ranked = input(f'How many top candidates would you like to see? (Please enter number 1-{file_count}) \n')

if not num_ranked.isdigit():
    print("Invalid input, displaying all ranks")
    num_ranked = file_count
elif int(num_ranked) > file_count:
    print("Input too big, displaying all ranks")
    num_ranked = file_count
elif int(num_ranked) < 1:
    print("Input too small, displaying all ranks")
    num_ranked = file_count

# Rank
ranking = ranker_service.rank_resumes(row[0], k=int(num_ranked))
results = ranking['results']

print("\n=== RANKING RESULTS ===")
if ranking['weak_batch']:
    print("⚠  WARNING: No strong matches found — even the top candidate is a weak fit for this job.\n")

for r in results:
    print(f"Rank {r['rank']}: {r['filename']} — Score: {r['similarity_score']:.4f}")