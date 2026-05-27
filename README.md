# flight-crew

## Resume Parser and Ranker Tool

A tool that parses uploaded resumes, anonymizes candidate PII, and ranks them
against a job description using semantic embeddings and a cross-encoder re-ranker.


### Dependencies and Installment Versions

Direct dependencies (all other packages install automatically as sub-dependencies):

```
pip install \
  PyMuPDF==1.27.2.3 \
  pymupdf4llm==1.27.2.3 \
  pdfplumber==0.11.9 \
  spacy==3.8.13 \
  sentence-transformers==5.5.0 \
  faiss-cpu==1.13.2 \
  numpy==2.4.5 \
  fire==0.7.1 \
  streamlit==1.57.0
```

#### spaCy model:
```
python -m spacy download en_core_web_trf
```
(On Windows, `en_core_web_lg` is used as a fallback if `en_core_web_trf` is unavailable.)

#### OCR support (optional but recommended):
The pipeline falls back to OCR for image-based PDFs. This requires **Tesseract**,
which is separate software, not a pip package.
- macOS: `brew install tesseract`
- Windows: install the Tesseract installer, then set the `TESSDATA_PREFIX`
  environment variable to the `tessdata` folder (e.g. `C:/Program Files/Tesseract-OCR/tessdata`).

#### Python version
- Python 3.13.13

### Pipeline Overview
```
PDF upload
  --> Text extraction (layout-aware, OCR fallback for image-based PDFs)
  --> PII anonymization (resumes only)
  --> Text cleaning / preprocessing for SBERT
  --> SBERT embeddings (all-mpnet-base-v2, 768-dim)
  --> FAISS retrieval (retrieves the full candidate pool)
  --> Cross-encoder re-ranking (reads JD + resume together, drives ranking quality)
  --> Per-batch score calibration (0-1 display scores) + weak-batch flag
  --> Ranked results + CSV export
```
 
The ranking uses a standard **retrieve-then-rerank** design: FAISS retrieves the candidate pool, then a cross-encoder re-scores each candidate by reading the job description and resume together. The cross-encoder output is the actual ranking signal.

### How to Run
 
```
python run.py <resume_folder> <jd_pdf_path>
```
 
Example:
```
python run.py data/resumes/ data/jd.pdf
```
 
This ingests all resume PDFs in the folder and the job description, builds the index, ranks the resumes, prints the results, and writes `ranked_results.csv`.
 
**Resetting between sessions:** the database and embedding cache are cleared
automatically when `run.py` finishes. If a run is interrupted, manually delete:
```
rm candidate_map.db
rm -rf data/embeddings/
```
Both must be deleted together — `embedder.py` caches embeddings by session-scoped candidate ID, so a stale cache will not line up with a fresh database.
 
### Storage (`candidate_map.db`)
 
File-based SQLite database with two tables:
- `candidates` — candidate_id, filename, hash, cleaned_text
- `job_descriptions` — jd_id, filename, hash, cleaned_text
- Hash-based duplicate detection — reprocessing the same file is skipped automatically.
- Deleted automatically at the end of each `run.py` session.

### Modules

#### Extraction (`extraction.py`)
- `extract_text()` — adaptive PDF text extraction; detects single vs. two-column
  layouts and reads them in correct order. Falls back to `pymupdf4llm` (with
  Tesseract OCR) for image-based pages that yield little text.
#### Extraction Engine (`text_extraction_engine.py`)
- `anonymize()` — removes PII (names via regex + spaCy NER, emails, phone numbers,
  addresses, LinkedIn/GitHub, personal websites).
- `extract_resumes()` — extracts, anonymizes, and cleans a batch of resume PDFs,
  storing them in the SQLite database with a unique candidate ID.
- `extract_jd()` — extracts and filters a job description PDF, storing it with a unique JD ID.
#### Preprocessing (`preprocessing.py`)
- `clean_text_for_sbert()` — lowercases text, strips Workday boilerplate and OCR
  artifacts, removes leftover PII placeholders, normalizes whitespace.
- `filter_jd_sections()` — drops non-requirement JD sections by header.
- `filter_excluded_sentences()` — drops sentences describing skills to be gained
  on the job, plus boilerplate.
#### Embedding (`embedder.py`)
- `get_embedding()` — loads `all-mpnet-base-v2` and converts text into a
  768-dimensional vector. Caches embeddings to `data/embeddings/` as `.npy` files.
- `cross_encoder_score()` — scores a single JD-resume pair with the cross-encoder
  (`ms-marco-MiniLM-L-6-v2`); returns a raw logit.
#### Ranking (`ranker.py`, `ranker_service.py`)
- `ResumeRanker` — wraps a FAISS index; used for retrieving the candidate pool.
- `rank_resumes()` — the main ranking function. Runs FAISS retrieval, cross-encoder
  re-ranking, and per-batch score calibration. Returns a dict:
  `{'results': [...], 'weak_batch': bool, 'best_logit': float}`.
  `weak_batch` is `True` when no candidate in the batch is a strong match.
#### CLI (`cli.py`)
- `upload_resume` — processes a single resume PDF.
- `upload_resumes` — processes all resume PDFs in a folder.
- `upload_jd` — processes a job description PDF.


### Notes
- Real-world Workday application-export PDFs are multi-column and noisy; the
  extraction layer handles them adaptively, but layout-aware extraction libraries are a recommended future improvement.
- The `similarity_score` shown in results is a **batch-relative** score — it
  reflects fit relative to the other resumes uploaded for the same job, not an absolute hireability score.