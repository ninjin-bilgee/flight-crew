import spacy
import re
import pymupdf4llm
import os
import hashlib
import json
import sys
import shutil
import tempfile
import fire

nlp = spacy.load("en_core_web_trf")

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

# Core processing function that can be called by both run() and run_all()
def process_resumes(pdf_paths):
    map_output_path = "candidate_map.json"
    os.makedirs('resumes_extracted_txt', exist_ok=True)

    # Use a temporary directory to stage files for processing, ensuring cleanup even if errors occur
    tmp_dir_obj = tempfile.TemporaryDirectory()
    try:
        tmp_dir = tmp_dir_obj.name
        print(f"Staging files in temp dir: {tmp_dir}")

        # Copy uploaded PDFs into the temp directory
        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            shutil.copy(pdf_path, os.path.join(tmp_dir, filename))

        # Load existing map
        if os.path.exists(map_output_path):
            with open(map_output_path, "r") as f:
                candidate_map = json.load(f)
        else:
            candidate_map = {}

        # Build reverse lookup: hash → candidate_id from previous runs
        hash_to_id = {v["hash"]: k for k, v in candidate_map.items()}
        id_counter = len(candidate_map) + 1

        # Extract text from each file while temp dir is still open
        for filename in os.listdir(tmp_dir):
            if not filename.endswith(".pdf"):
                continue

            tmp_path = os.path.join(tmp_dir, filename)

            # Read bytes for duplicate detection
            with open(tmp_path, "rb") as f:
                file_bytes = f.read()

            file_hash = get_file_hash(file_bytes)

            if file_hash in hash_to_id:
                print(f"Skipping {filename} — already processed as {hash_to_id[file_hash]}")
                continue

            candidate_id = f"Candidate_{id_counter}"
            id_counter += 1

            hash_to_id[file_hash] = candidate_id
            candidate_map[candidate_id] = {"filename": filename, "hash": file_hash}

            # Extract and anonymize
            md = pymupdf4llm.to_markdown(tmp_path)
            md_clean = anonymize(md)

            # Save to output
            output_path = f"resumes_extracted_txt/{candidate_id}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(md_clean)

            print(f"Processed {filename} → {candidate_id}")

    # Guaranteed cleanup of the temp directory even if errors/crashes occur in-session
    finally:
        tmp_dir_obj.cleanup()
        print("Temp directory cleaned up.")

    # Save candidate map
    with open("candidate_map.json", "w") as f:
        json.dump(candidate_map, f, indent=2)
    print("Candidate map saved to candidate_map.json")

# Process specific PDF files
def run(*pdf_paths):
    process_resumes(list(pdf_paths))

# Process specific PDF files
def run_all(folder="resumes_pdf"):
    pdf_paths = sorted(
        os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pdf")
    )
    if not pdf_paths:
        print(f"No PDFs found in {folder}/")
        return
    process_resumes(pdf_paths)

# Command-line interface using fire
if __name__ == "__main__":
    fire.Fire({"run": run, "run_all": run_all})

