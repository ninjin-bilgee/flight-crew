import spacy
import re
import pymupdf4llm
import os
import hashlib
import json

# Import the spacy model for entity recognition
nlp = spacy.load("en_core_web_sm")

# Set to track seen file hashes to avoid processing duplicates
seen_hashes = set()

# Create map of candidates to their anonymized IDs for reference
candidate_map = {}

# Computes a SHA-256 hash of the file bytes to identify duplicates
def get_file_hash(file_bytes):
    return hashlib.sha256(file_bytes).hexdigest()

def anonymize(text, candidate_id):
    doc = nlp(text)
    
    # Strip names spaCy detects
    for ent in sorted(doc.ents, reverse=True, key=lambda e: e.start_char):
        if ent.label_ == "PERSON":
            text = text[:ent.start_char] + candidate_id + text[ent.end_char:]
    
    # Detect oddly spelled out names
    text = re.sub(r'\b([A-Z]\s){2,}[A-Z]\b', '[name removed]', text)
    
    # Detect other PIIs
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[email removed]', text)
    text = re.sub(r'\(?\d{3}\)?[\s.\-]\d{3}[\s.\-]\d{4}', '[phone removed]', text)
    text = re.sub(r'(https?://)?(www\.)?linkedin\.com/\S+', '[linkedin removed]', text)
    text = re.sub(r'(https?://)?(www\.)?github\.com/\S+', '[github removed]', text)
    
    return text

# Change these numbers to test different subsets
RESUMES_TO_TEST = [6, 7, 8, 9, 10]

os.makedirs('resumes_extracted_txt', exist_ok=True)

# Process each resume PDF, extract markdown, anonymize, and save
for num in RESUMES_TO_TEST:

    # Construct the file path for the current resume PDF
    pdf_path = f"resumes_pdf/sample_resume_{num}.pdf"

    # Read the file bytes to compute the hash for duplicate detection
    with open(pdf_path, "rb") as f:
        file_bytes = f.read()

    # Assign the file hash
    file_hash = get_file_hash(file_bytes)

    # Check if the file hash has already been seen to avoid processing duplicates
    if file_hash in seen_hashes:
        print(f"Skipping sample_resume_{num}.pdf — duplicate")
        continue

    # This file has been seen, so add to seen hashes
    seen_hashes.add(file_hash)

    # Assign a candidate ID based on the number of unique files processed so far
    candidate_id = f"Candidate_{len(seen_hashes)}"
    candidate_map[candidate_id] = f"sample_resume_{num}.pdf"

    md = pymupdf4llm.to_markdown(pdf_path)
    md_clean = anonymize(md, candidate_id)

    output_path = f"resumes_extracted_txt/{candidate_id}.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_clean)

    print(f"Processed sample_resume_{num}.pdf → {candidate_id}")

with open("candidate_map.json", "w") as f:
    json.dump(candidate_map, f, indent=2)
print("Candidate map saved to candidate_map.json")