"""
Pipeline using extraction.py for text extraction instead of text_extraction_engine.py.

Usage:
    python run.py <resume_folder> <jd_pdf_path>

Example:
    python run.py data/resumes/ data/jd.pdf
"""
import sqlite3
import sys
import os
import hashlib
import shutil
import atexit
import ranker_service
from embedder import get_embedding
from ranker import ResumeRanker
from extraction import extract_text
from text_extraction_engine import anonymize
from preprocessing import clean_text_for_sbert, filter_jd_sections, filter_excluded_sentences

if len(sys.argv) != 3:
    print("Usage: python pipeline_extraction.py <resume_folder> <jd_pdf_path>")
    sys.exit(1)

RESUME_FOLDER = sys.argv[1]
JD_PATH = sys.argv[2]
DB_PATH = "candidate_map.db"
EMBEDDINGS_PATH = "data/embeddings/"

conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        candidate_id  TEXT PRIMARY KEY,
        filename      TEXT,
        hash          TEXT UNIQUE,
        cleaned_text  TEXT
    )
""")
conn.execute("""
    CREATE TABLE IF NOT EXISTS job_descriptions (
        jd_id        TEXT PRIMARY KEY,
        filename     TEXT,
        hash         TEXT UNIQUE,
        cleaned_text TEXT
    )
""")
conn.commit()

def _cleanup():
    conn.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("\nSession ended — candidate_map.db deleted.")
    if os.path.exists(EMBEDDINGS_PATH):
        shutil.rmtree(EMBEDDINGS_PATH)
        print("Session ended — embeddings cache cleared.")

atexit.register(_cleanup)

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def ingest_resumes(resume_folder):
    pdf_paths = [
        os.path.join(resume_folder, f)
        for f in os.listdir(resume_folder)
        if f.endswith(".pdf")
    ]
    if not pdf_paths:
        print(f"No PDFs found in {resume_folder}")
        sys.exit(1)

    row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
    id_counter = row[0] + 1

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
        file_hash = get_file_hash(file_bytes)

        existing = conn.execute(
            "SELECT candidate_id FROM candidates WHERE hash = ?", (file_hash,)
        ).fetchone()
        if existing:
            print(f"Skipping {filename} — already processed as {existing[0]}")
            continue

        candidate_id = f"Candidate_{id_counter}"
        id_counter += 1

        text = extract_text(pdf_path)
        text = anonymize(text)
        cleaned = clean_text_for_sbert(text)

        conn.execute(
            "INSERT INTO candidates (candidate_id, filename, hash, cleaned_text) VALUES (?, ?, ?, ?)",
            (candidate_id, filename, file_hash, cleaned)
        )
        conn.commit()
        print(f"Processed {filename} → {candidate_id}")

def ingest_jd(jd_path):
    with open(jd_path, "rb") as f:
        file_bytes = f.read()
    file_hash = get_file_hash(file_bytes)

    existing = conn.execute(
        "SELECT jd_id FROM job_descriptions WHERE hash = ?", (file_hash,)
    ).fetchone()
    if existing:
        print(f"JD already processed — reusing {existing[0]}")
        return existing[0]

    text = extract_text(jd_path)
    text = filter_jd_sections(text.strip())
    text = filter_excluded_sentences(text)
    cleaned = clean_text_for_sbert(text)

    filename = os.path.basename(jd_path)
    jd_id = f"JD_{file_hash[:8]}"

    conn.execute(
        "INSERT OR REPLACE INTO job_descriptions (jd_id, filename, hash, cleaned_text) VALUES (?, ?, ?, ?)",
        (jd_id, filename, file_hash, cleaned)
    )
    conn.commit()
    print(f"Job description processed → {jd_id}")
    return jd_id


# --- Main pipeline ---

ingest_resumes(RESUME_FOLDER)
ingest_jd(JD_PATH)

rows = conn.execute("SELECT candidate_id, filename, cleaned_text FROM candidates").fetchall()
if not rows:
    print("No candidates found.")
    sys.exit(1)

print(f"\nFound {len(rows)} candidates.")

for cid, fn, ct in rows:
    print(f"\n=== {fn} ({len(ct)} chars) ===")
    print(ct[:800])

# filename -> cleaned text, used to show a resume snippet next to each result
snippets = {fn: ct for _, fn, ct in rows}

embeddings = []
metadata = []
for candidate_id, filename, cleaned_text in rows:
    print(f"Embedding {filename}...")
    emb = get_embedding(cleaned_text, resume_id=candidate_id)
    embeddings.append(emb)
    metadata.append({"filename": filename})

ranker_service._ranker = ResumeRanker(embedding_dim=768)
ranker_service._ranker.build_index(embeddings, metadata)

row = conn.execute("SELECT cleaned_text FROM job_descriptions LIMIT 1").fetchone()
if not row:
    print("No job description found.")
    sys.exit(1)

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

ranking = ranker_service.rank_resumes(row[0], k=int(num_ranked))
results = ranking['results']

print("\n=== RANKING RESULTS ===")
if ranking['weak_batch']:
    print("⚠  WARNING: No strong matches found — even the top candidate is a weak fit.\n")

for r in results:
    snippet = snippets.get(r['filename'], '')[:120].strip()
    print(f"Rank {r['rank']}: {r['filename']} — Score: {r['similarity_score']:.4f}")
    print(f"    {snippet}...")