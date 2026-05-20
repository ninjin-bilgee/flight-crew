import sqlite3
from embedder import get_embedding
from ranker import ResumeRanker

# Load cleaned text from SQLite
conn = sqlite3.connect("candidate_map.db")
rows = conn.execute("SELECT candidate_id, cleaned_text FROM candidates").fetchall()

if not rows:
    print("No candidates in database. Run text_extraction_engine.py first.")
    exit()

# Generate embeddings for all candidates
print(f"Found {len(rows)} candidates in SQLite...")
embeddings = []
metadata = []

for candidate_id, cleaned_text in rows:
    print(f"Generating embedding for {candidate_id}...")
    emb = get_embedding(cleaned_text, resume_id=candidate_id)
    embeddings.append(emb)
    metadata.append({"filename": candidate_id})

# Build FAISS index
ranker = ResumeRanker(embedding_dim=384)
ranker.build_index(embeddings, metadata)

# Test with a sample job description
job_description = """
Looking for a software engineer with experience in Python, 
machine learning, APIs, and cloud platforms.
"""

from preprocessing import clean_text_for_sbert
clean_jd = clean_text_for_sbert(job_description)
jd_embedding = get_embedding(clean_jd)

# Rank resumes
results = ranker.search(jd_embedding, k=5)

print("\n=== RANKING RESULTS ===")
for r in results:
    print(f"Rank {r['rank']}: {r['filename']} — Score: {r['similarity_score']:.4f}")