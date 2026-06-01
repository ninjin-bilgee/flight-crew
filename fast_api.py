from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from typing import List
import uuid
import os
import numpy as np

from storage import save_candidate, export_to_csv

from embedder import get_embedding
from compare import extract_text_from_pdf, clean_text_simple

from sentence_transformers import util

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stored_results = {}

# create folder for uploaded resumes
os.makedirs("resumes_pdf", exist_ok=True)


@app.get("/")
def home():
    return {"message": "Resume Parser API is running"}


@app.post("/upload", openapi_extra={
    "requestBody": {
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "format": "binary"
                            }
                        },
                        "jd_text": {
                            "type": "string"
                        }
                    },
                    "required": ["files"]
                }
            }
        }
    }
})
async def upload_files(
    files: List[UploadFile] = File(...),
    jd_text: str = Form(None)
):

    # generate unique upload session id
    job_id = str(uuid.uuid4())[:8]

    filenames = []

    processed_resumes = []

    for file in files:

        filenames.append(file.filename)

        # save uploaded pdf locally
        file_path = f"resumes_pdf/{file.filename}"

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # extract raw text from pdf
        raw_text = extract_text_from_pdf(file_path)

        # handle empty extraction
        if not raw_text:
            raw_text = "empty resume"

        # clean extracted text
        cleaned_resume_text = clean_text_simple(raw_text)

        # generate embedding
        resume_embedding = get_embedding(
            cleaned_resume_text,
            resume_id=file.filename
        )

        processed_resume = {
            "filename": file.filename,
            "cleaned_text": cleaned_resume_text,
            "embedding": resume_embedding.tolist()
        }

        processed_resumes.append(processed_resume)

        # generate candidate id
        candidate_id = str(uuid.uuid4())[:8]

        # save candidate info into JSON
        save_candidate(
            candidate_id=candidate_id,
            filename=file.filename,
            score=0.0,
            skills=["python", "fastapi"],
            embedding=resume_embedding.tolist()
        )

    # export ranked csv
    export_to_csv()

    # store upload session results
    stored_results[job_id] = {
        "filenames": filenames,
        "jd_text": jd_text,
        "processed_resumes": processed_resumes
    }

    return {
        "job_id": job_id,
        "filenames": filenames,
        "message": "Files uploaded, processed, and embeddings generated successfully"
    }


@app.post("/rank/{job_id}")
async def rank_resumes(
    job_id: str,
    jd_text: str = Form(...)
):

    # check if job id exists
    if job_id not in stored_results:
        return {
            "error": "Job ID not found"
        }
        
    # generate embedding for job description
    jd_embedding = get_embedding(jd_text)

    # fix tensor datatype mismatch
    jd_embedding = np.array(
        jd_embedding,
        dtype=np.float32
    )

    rankings = []

    data = stored_results[job_id]

    # compare each resume embedding with JD embedding
    for resume in data["processed_resumes"]:

        resume_embedding = np.array(
            resume["embedding"],
            dtype=np.float32
        )

        similarity = util.cos_sim(
            jd_embedding,
            resume_embedding
        )    

        score = similarity.item()

        rankings.append({
            "filename": resume["filename"],
            "score": round(score, 4)
        })

    # sort highest similarity first
    rankings = sorted(
        rankings,
        key=lambda x: x["score"],
        reverse=True
    )

    return {
        "job_id": job_id,
        "jd_text": jd_text,
        "rankings": rankings
    }


@app.get("/results")
async def get_results():

    return stored_results