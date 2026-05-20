import pdfplumber
from embedder import get_embedding
from sentence_transformers import util
import numpy as np

def extract_text_from_pdf(pdf_path):
    #Extract raw text from a PDF file.
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

def clean_text_simple(raw_text):
    #Basic cleaning: lowercase, remove extra spaces, keep only letters/digits/spaces.
    import re
    text = raw_text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main(pdf_resume_path, pdf_job_path):
    # Extract raw text
    print("Extracting text from resume...")
    raw_resume = extract_text_from_pdf(pdf_resume_path)
    raw_job = extract_text_from_pdf(pdf_job_path)
    
    if not raw_resume or not raw_job:
        print("Error: Could not extract text from one or both PDFs.")
        return
    
    # Clean text
    clean_resume = clean_text_simple(raw_resume)
    clean_job = clean_text_simple(raw_job)
    
    # Generate embeddings
    print("Generating embeddings...")
    emb_resume = get_embedding(clean_resume)          # no resume_id – not caching
    emb_job = get_embedding(clean_job)
    
    # Compute cosine similarity
    similarity = util.cos_sim(emb_resume, emb_job)
    score = similarity.item()
    
    print(f"\nSimilarity score between resume and job description: {score:.4f}")
    print("(1.0 = very similar, 0.0 = unrelated)")
    
if __name__ == "__main__":
    # Replace with your actual file paths
    resume_file = "test pdfs/resume.pdf"
    job_file = "test pdfs/jd_mismatch.pdf"
    main(resume_file, job_file)