# Extraction Engine: PDF → Cleaned Text with Candidate Mapping
import spacy
import re
import pymupdf4llm
import os
import hashlib
import sys
import shutil
import tempfile
import sqlite3
import atexit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import filter_jd_sections, filter_excluded_sentences, clean_text_for_sbert

# Fallback for windows since we had to use lg model for it, just in case!!
try:
    nlp = spacy.load("en_core_web_trf")
except OSError:
    nlp = spacy.load("en_core_web_lg")

# Set up SQLite database for candidate mapping
conn = sqlite3.connect("candidate_map.db")

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

def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def anonymize(text):
    # Strip markdown syntax from raw extraction so spaCy sees plain text — ## and ** around the name
    # cause the transformer to treat it as a section header instead of a person
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*+', '', text)

    # Header name: resumes always lead with the candidate name
    # but spaCy needs surrounding prose context to detect isolated headers reliably. 
    # Catches names sharing a line with contact info (e.g. "Erica Engineer, E.I.T.  email • phone")
    # Without this, the regex would skip that line and wrongly match the first section header instead (e.g. "SUMMARY OF QUALIFICATIONS")
    text = re.sub(
        r'^([A-Z][A-Za-z\'\-\.]+(?:[ ]+[A-Z][A-Za-z\'\-\.]+){1,3})(?=[,\s]|$)',
        '[name removed]',
        text,
        count=1,
        flags=re.MULTILINE,
    )

    # Person names via spaCy NER
    doc = nlp(text)
    for ent in sorted(doc.ents, reverse=True, key=lambda e: e.start_char):
        if ent.label_ == "PERSON":
            text = text[:ent.start_char] + '[name removed]' + text[ent.end_char:]

    # Spaced-out initials spaCy may miss (e.g. "J O H N")
    text = re.sub(r'\b([A-Z]\s){2,}[A-Z]\b', '[name removed]', text)

    # Email — must run before the social-handle regex so @domain isn't consumed
    text = re.sub(r'[\w\.\+\-]+@[\w\.\-]+\.\w+', '[removed]', text)

    # Social handle (bare @handle, emails already gone)
    text = re.sub(r'@[\w]+', '[removed]', text)

    # Street address
    text = re.sub(r'\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Blvd|Lane|Ln|Way|Court|Ct)[\w\s,\.]*\d{5}(?:-\d{4})?', '[removed]', text, flags=re.IGNORECASE)

    # Phone - US (DDD-DDD-DDDD), with optional +1 country code, and short
    # template placeholders like 1-234-5678. A second pass mops up any DDD-DDD-
    # prefix left behind by a prior partial match.
    text = re.sub(r'(?:\+?1[\s.\-])?\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}', '[removed]', text)
    text = re.sub(r'\b1[\s.\-]\d{3}[\s.\-]\d{4}\b', '[removed]', text)
    text = re.sub(r'\b\d{3}[\s.\-]\d{3}[\s.\-]', '[removed]', text)

    # LinkedIn
    text = re.sub(r'(?:https?://)?(?:www\.)?linkedin\.com/\S*', '[removed]', text, flags=re.IGNORECASE)
    text = re.sub(r'\blinkedin\b', '[removed]', text, flags=re.IGNORECASE)

    # GitHub
    text = re.sub(r'(?:https?://)?(?:www\.)?github\.com/\S*', '[removed]', text, flags=re.IGNORECASE)
    text = re.sub(r'\bgithub\b', '[removed]', text, flags=re.IGNORECASE)

    # Personal websites
    text = re.sub(r'(?:https?://)?(?:www\.)?[\w\-]+\.(?:com|io|dev|me|net|org)/?\S*', '[removed]', text)

    # Catch leftover contact placeholders
    text = re.sub(r'\b\w[\w\s/\-,]*(?:url|link|portfolio|website|optional)\b\s*(\(optional\))?', '[removed]', text, flags=re.IGNORECASE)

    return text

def extract_resumes(pdf_paths):

    # Use a temporary directory to stage files for processing, ensuring cleanup even if errors occur
    tmp_dir_obj = tempfile.TemporaryDirectory()

    try:
        tmp_dir = tmp_dir_obj.name
        print(f"Staging files in temp dir: {tmp_dir}")

        # Copy uploaded PDFs into the temp directory
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            shutil.copy(pdf_path, os.path.join(tmp_dir, filename))
        
        # Get current candidate count from SQLite to continue ID numbering correctly
        row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
        id_counter = row[0] + 1

        # Extract text from each file while temp dir is still open
        for filename in os.listdir(tmp_dir):
            if not filename.endswith(".pdf"):
                continue

            tmp_path = os.path.join(tmp_dir, filename)

            # Read bytes for duplicate detection
            with open(tmp_path, "rb") as f:
                file_bytes = f.read()

            file_hash = get_file_hash(file_bytes)

            # Duplicate check with SQLite UNIQUE constraint on hash column
            existing = conn.execute(
                "SELECT candidate_id FROM candidates WHERE hash = ?",
                (file_hash,)
            ).fetchone()
            if existing:
                print(f"Skipping {filename} — already processed as {existing[0]}")
                continue

            candidate_id = f"Candidate_{id_counter}"
            id_counter += 1

            # Extract and anonymize
            md = pymupdf4llm.to_markdown(tmp_path)
            md_clean = anonymize(md)

            # Further clean text for SBERT embedding (e.g. remove newlines, excessive whitespace) but keep the original cleaned markdown for storage and potential future use
            cleaned_text = clean_text_for_sbert(md_clean)

            # Save candidate mapping to SQLite with cleaned (not raw) text
            conn.execute(
                "INSERT INTO candidates (candidate_id, filename, hash, cleaned_text) VALUES (?, ?, ?, ?)",
                (candidate_id, filename, file_hash, cleaned_text)
            )
            conn.commit()

            print(f"Processed {filename} → {candidate_id}")

    # Guaranteed cleanup of the temp directory even if errors/crashes occur in-session
    finally:
        tmp_dir_obj.cleanup()
        print("Temp directory cleaned up.")

# Extract text from a job description PDF — no anonymization needed!!
def extract_jd(pdf_path):

    # Read bytes for duplicate detection
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()
    file_hash = get_file_hash(file_bytes)

    # Reuse existing JD if already processed, based on file hash
    existing = conn.execute(
        "SELECT jd_id FROM job_descriptions WHERE hash = ?",
        (file_hash,)
    ).fetchone()
    if existing:
        print(f"JD already processed — reusing {existing[0]}")
        return existing[0]
    
    # Process for new JDs:
    text = pymupdf4llm.to_markdown(pdf_path)
    text = filter_jd_sections(text.strip())
    text = filter_excluded_sentences(text)
    
    cleaned = clean_text_for_sbert(text)
    
    filename = os.path.basename(pdf_path)
    jd_id = f"JD_{file_hash[:8]}"
    
    conn.execute(
        "INSERT OR REPLACE INTO job_descriptions (jd_id, filename, hash, cleaned_text) VALUES (?, ?, ?, ?)",
        (jd_id, filename, file_hash, cleaned)
    )

    conn.commit()
    print(f"Job description processed → {jd_id}")
    return jd_id

