# This script allows users to upload PDF resumes through a Streamlit interface.
# Extracts the content as markdown using pymupdf4llm
# Anonymizes personally identifiable information (PII) before saving the markdown files. 
# The script also ensures that duplicate uploads are not processed multiple times by hashing the file contents.

import spacy
import re
import pymupdf4llm
import os
import hashlib
import streamlit as st

st.title("Resume Parser - UI Test 1")

nlp = spacy.load("en_core_web_sm")

seen_hashes = set()

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

uploaded_files = st.file_uploader(
    "Upload resumes", 
    type=["pdf"], 
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs('temp', exist_ok=True)                  
    os.makedirs('resumes_extracted_md', exist_ok=True)
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        file_hash = get_file_hash(file_bytes)
        
        if file_hash in seen_hashes:
            continue
        
        seen_hashes.add(file_hash)
        
        pdf_path = f"temp/{uploaded_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)
        
        candidate_id = f"Candidate_{len(seen_hashes)}"
        
        md = pymupdf4llm.to_markdown(pdf_path)
        
        # Anonymize the markdown content before saving
        md_clean = anonymize(md, candidate_id)

filename = os.path.splitext(os.path.basename(pdf_path))[0] 
output_path = f'resumes_extracted_md/{filename}.md'

# Create markdown file with anonymized content
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_clean)

st.success(f"Processed: {uploaded_file.name} → {candidate_id}")


