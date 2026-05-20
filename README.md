# flight-crew

## Resume Parser and Ranker Tool

### Dependencies and Installment Versions
#### pip install:
- pymupdf4llm 1.27.2.3
- spacy 3.8.13
- Sentence-transformers 5.5.0
- pdfplumber 0.11.9
- torch 2.12.0
- Scikit-learn 1.8.0
- faiss-cpu 2.3.2
- fire 0.7.1
- Python 3.14.2
#### python -m spacy download en_core_web_trf

### Pipeline Overview
PDF upload --> PII anonymization --> Text extraction into .txt format --> Text preprocessing, cleaning, and preparation for SBERT's semantic comparison between resumes and job descriptions --> SBERT embeddings --> FAISS ranking --> ranked results

### How to Run
**Process all resumes in a folder:**
python test_scripts/pdf_to_txt_extract_mapped.py run_all --folder /path/to/resumes
**Process specific resumes:**
python test_scripts/pdf_to_txt_extract_mapped.py run resume1.pdf resume2.pdf
**Test full pipeline (in full-test branch):**
python test_pipeline_1.py

### Functions
#### Extraction
- `anonymize()` - censors PII like name, location, email, social handles, phone number, and websites
- `process_resumes()` - core processing function called by CLI commands; uses temporary directory to stage files for processing; stores censored files in an in-memory SQLite DB that will delete itself after session finishes
**CLI Commands**
- run(): processes specific PDF files (cherry-picking concept)
- run_all: processes multiple PDF files in a folder path (processes all files in a named folder)

#### Preprocessing
- `clean_text_for_sbert()` - lowercases text, removes leftover PII placeholders, normalizes whitespace and special characters for SBERT input
- `process_resume()` - loads a txt file and runs it through the cleaning pipeline

#### SBERT Embedding (`embedder.py`)
- `get_embedding()` — loads `all-MiniLM-L6-v2` model and converts text into a 384-dimensional vector
- Caches embeddings to `data/embeddings/` as `.npy` files to avoid recomputation

#### Ranking (`ranker.py`, `ranker_service.py`)
- `ResumeRanker` — wraps a FAISS index using inner product for cosine similarity
- `build_index()` — builds FAISS index from a list of resume embeddings
- `search()` — takes a job description embedding, returns top-k most similar resumes with scores
- `rank_resumes()` — high level function for FastAPI backend integration


