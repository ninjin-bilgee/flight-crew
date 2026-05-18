from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uuid

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary in-memory storage
stored_results = {}

@app.get("/")
def home():
    return {"message": "Resume Parser API is running"}


# Upload Endpoint


@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    jd_text: str = Form(None)
):

    job_id = str(uuid.uuid4())[:8]

    filenames = []

    cleaned_text = []

    for file in files:
        filenames.append(file.filename)

        # dummy preprocessing
        cleaned_text.append({
            "filename": file.filename,
            "cleaned_text": "preprocessed resume text"
        })

    stored_results[job_id] = {
        "filenames": filenames,
        "jd_text": jd_text,
        "cleaned_text": cleaned_text
    }

    return {
        "job_id": job_id,
        "filenames": filenames
    }


# Rank Endpoint


@app.post("/rank")
async def rank_resumes(jd_text: str = Form(...)):

    # dummy ranking response
    rankings = [
        {
            "filename": "resume1.pdf",
            "score": 0.91
        },
        {
            "filename": "resume2.pdf",
            "score": 0.84
        }
    ]

    return {
        "jd_text": jd_text,
        "rankings": rankings
    }


# Results Endpoint


@app.get("/results/{job_id}")
async def get_results(job_id: str):

    if job_id not in stored_results:
        return {"error": "Job ID not found"}

    return stored_results[job_id]