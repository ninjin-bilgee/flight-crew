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
   python test_pipeline_1.py
"""
import sqlite3
import sys
from embedder import get_embedding
from ranker import ResumeRanker
from preprocessing import clean_text_for_sbert

# Connect to database
conn = sqlite3.connect("candidate_map.db")

# Check if candidates exist
rows = conn.execute("SELECT candidate_id, cleaned_text FROM candidates").fetchall()

if not rows:
    print("No candidates found in database.")
    print("Run this first:")
    print("  python test_scripts/text_extraction_engine.py run_all --folder /path/to/resumes")
    sys.exit(1)

print(f"Found {len(rows)} candidates.")

# Generate embeddings
embeddings = []
metadata = []
for candidate_id, cleaned_text in rows:
    print(f"Embedding {candidate_id}...")
    emb = get_embedding(cleaned_text, resume_id=candidate_id)
    embeddings.append(emb)
    metadata.append({"filename": candidate_id})

# Build FAISS index
ranker = ResumeRanker(embedding_dim=384)
ranker.build_index(embeddings, metadata)

# read JD from SQLite DB
row = conn.execute("SELECT cleaned_text FROM job_descriptions LIMIT 1").fetchone()
if not row:
    print("No job description found. Run:")
    print("  python test_scripts/text_extraction_engine.py upload_jd /path/to/jd.pdf")
    sys.exit(1)
jd_embedding = get_embedding(row[0])

# Rank
results = ranker.search(jd_embedding, k=5)

print("\n=== RANKING RESULTS ===")
for r in results:
    print(f"Rank {r['rank']}: {r['filename']} — Score: {r['similarity_score']:.4f}")