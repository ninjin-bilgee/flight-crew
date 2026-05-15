import spacy
import re
import pymupdf4llm
import os

nlp = spacy.load("en_core_web_sm")

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

pdf_path = 'resumes_pdf/sample_resume_19.pdf'
candidate_id = 'Candidate_A'

md = pymupdf4llm.to_markdown(pdf_path)

# Anonymize the content before saving
md_clean = anonymize(md, candidate_id)

filename = os.path.splitext(os.path.basename(pdf_path))[0]
output_path = f'resumes_extracted_txt/{filename}.txt'

os.makedirs('resumes_extracted_txt', exist_ok=True)

# Create txt file with anonymized content
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md_clean)

print(f"Saved to {output_path}")