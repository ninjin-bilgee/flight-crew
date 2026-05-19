from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from typing import List
import uuid

from storage import save_candidate, export_to_csv

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stored_results = {}


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
                            "items": {"type": "string", "format": "binary"}
                        },
                        "jd_text": {"type": "string"}
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
    job_id = str(uuid.uuid4())[:8]
    filenames = []
    cleaned_text = []

    for file in files:
        filenames.append(file.filename)

        processed_resume = {
            "filename": file.filename,
            "cleaned_text": "preprocessed resume text"
        }
        cleaned_text.append(processed_resume)

        candidate_id = str(uuid.uuid4())[:8]
        save_candidate(
            candidate_id=candidate_id,
            filename=file.filename,
            score=0.0,
            skills=["python", "fastapi"]
        )

    export_to_csv()

    stored_results[job_id] = {
        "filenames": filenames,
        "jd_text": jd_text,
        "cleaned_text": cleaned_text
    }

    return {
        "job_id": job_id,
        "filenames": filenames,
        "message": "Files uploaded and saved successfully"
    }


@app.post("/rank")
async def rank_resumes(
    jd_text: str = Form(...)
):
    rankings = [
        {"filename": "resume1.pdf", "score": 0.91},
        {"filename": "resume2.pdf", "score": 0.84}
    ]

    return {
        "jd_text": jd_text,
        "rankings": rankings
    }


@app.get("/results")
async def get_results():
    return stored_results