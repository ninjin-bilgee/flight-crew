import fire
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test_scripts"))

from text_extraction_engine import process_resumes, extract_jd

# Upload and process all resumes in a folder
def upload_resumes(folder):
    pdf_paths = sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".pdf")
    )
    if not pdf_paths:
        print(f"No PDFs found in {folder}")
        return
    process_resumes(pdf_paths)

# Upload and process a single resume PDF
def upload_resume(pdf_path):
    process_resumes([pdf_path])

# Upload and process a job description PDF
def upload_jd(pdf_path):
    extract_jd(pdf_path)

# Delete the local candidate DB
# Run this when done with a session
def clear():
    db_path = "candidate_map.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print("Session cleared — candidate_map.db deleted.")
    else:
        print("Nothing to clear.")

if __name__ == "__main__":
    fire.Fire({
        "upload_resumes": upload_resumes, 
        "upload_resume": upload_resume,   
        "upload_jd": upload_jd,
        "clear": clear
    })

