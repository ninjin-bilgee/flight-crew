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

HOWEVER, if you are adding new resumes to an existing batch, just do steps 1-3 directly.

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
import csv
import ranker_service
from embedder import get_embedding
from ranker import ResumeRanker
from extraction import extract_text
from text_extraction_engine import anonymize
from preprocessing import clean_text_for_sbert, filter_jd_sections, filter_excluded_sentences

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

# open (or create) the SQLite DB used to map candidates and JDs to their cleaned text
conn = sqlite3.connect(DB_PATH)
# candidates table: one row per resume, keyed by a unique candidate_id
conn.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        candidate_id  TEXT PRIMARY KEY,
        filename      TEXT,
        hash          TEXT UNIQUE,
        cleaned_text  TEXT
    )
""")
# job_descriptions table: one row per JD, keyed by a unique jd_id
conn.execute("""
    CREATE TABLE IF NOT EXISTS job_descriptions (
        jd_id        TEXT PRIMARY KEY,
        filename     TEXT,
        hash         TEXT UNIQUE,
        cleaned_text TEXT
    )
""")
conn.commit()

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

# returns a SHA-256 hash of the file bytes — used as a unique fingerprint for duplicate detection
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

"""
Function:
- Extracts text from every PDF resume in the given folder
- Anonymizes PII and cleans the text for embedding
- Saves each resume to the SQLite DB with a unique candidate ID (Candidate_#)
- Skips any resume whose file hash already exists (duplicate detection)
"""
def ingest_resumes(resume_folder):
    # collect all PDF file paths in the resume folder
    pdf_paths = [
        os.path.join(resume_folder, f)
        for f in os.listdir(resume_folder)
        if f.endswith(".pdf")
    ]
    # stop early if the folder has no PDFs
    if not pdf_paths:
        print(f"No PDFs found in {resume_folder}")
        sys.exit(1)

    # get the current candidate count so ID numbering continues correctly
    row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
    # id_counter starts at 1 if no candidates, otherwise continues from last count + 1
    id_counter = row[0] + 1

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        # read bytes for duplicate detection
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()
        file_hash = get_file_hash(file_bytes)

        # duplicate check against the UNIQUE hash column
        existing = conn.execute(
            "SELECT candidate_id FROM candidates WHERE hash = ?", (file_hash,)
        ).fetchone()
        # if this exact file was already processed, skip it
        if existing:
            print(f"Skipping {filename} — already processed as {existing[0]}")
            continue

        # assign a new unique candidate ID
        candidate_id = f"Candidate_{id_counter}"
        id_counter += 1

        # extract raw text (layout-aware, OCR fallback) -> anonymize PII -> clean for SBERT
        text = extract_text(pdf_path)
        text = anonymize(text)
        cleaned = clean_text_for_sbert(text)

        # save the candidate mapping with the final cleaned text
        conn.execute(
            "INSERT INTO candidates (candidate_id, filename, hash, cleaned_text) VALUES (?, ?, ?, ?)",
            (candidate_id, filename, file_hash, cleaned)
        )
        conn.commit()
        print(f"Processed {filename} → {candidate_id}")

"""
Function:
- Extracts text from a single job description PDF
- Applies JD-specific filtering (drops boilerplate sections and future-skill sentences)
- Cleans the text for embedding and saves it to the SQLite DB with a unique JD ID
- Reuses an existing JD if the same file (by hash) was already processed
Returns: the JD's unique ID (JD_#)
"""
def ingest_jd(jd_path):
    # read bytes for duplicate detection
    with open(jd_path, "rb") as f:
        file_bytes = f.read()
    file_hash = get_file_hash(file_bytes)

    # if this exact JD was already processed, reuse it instead of reprocessing
    existing = conn.execute(
        "SELECT jd_id FROM job_descriptions WHERE hash = ?", (file_hash,)
    ).fetchone()
    if existing:
        print(f"JD already processed — reusing {existing[0]}")
        return existing[0]

    # extract text, then apply the two JD-specific filters before cleaning
    text = extract_text(jd_path)
    text = filter_jd_sections(text.strip())       # drop whole boilerplate sections by header
    text = filter_excluded_sentences(text)        # drop future-skill / boilerplate sentences
    cleaned = clean_text_for_sbert(text)

    filename = os.path.basename(jd_path)
    # JD ID is derived from the first 8 chars of the file hash
    jd_id = f"JD_{file_hash[:8]}"

    # save the JD mapping with the final cleaned text
    conn.execute(
        "INSERT OR REPLACE INTO job_descriptions (jd_id, filename, hash, cleaned_text) VALUES (?, ?, ?, ?)",
        (jd_id, filename, file_hash, cleaned)
    )
    conn.commit()
    print(f"Job description processed → {jd_id}")
    return jd_id


# --- Main pipeline --- #

# Step 1: ingest all resumes and the job description into the DB
ingest_resumes(RESUME_FOLDER)
ingest_jd(JD_PATH)

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
    print(f"Rank {r['rank']}: {r['filename']} — Score: {r['similarity_score'] * 100:.2f}%")
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
            "similarity_score": round(r["similarity_score"] * 100, 2),
        })

print(f"\nRanked results saved to {OUTPUT_CSV}")