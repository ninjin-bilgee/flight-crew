# flight-crew

## Resume Parser and Ranker Tool

### Dependencies and Installment Versions
pymupdf4llm 1.27.2.3
spacy 3.8.13
en_core_web_trf 3.8.0
Sentence-transformers 5.5.0
pdfplumber 0.11.9
torch 2.12.0
Scikit-learn 1.8.0
faiss-cpu 2.3.2
fire 0.7.1
Python 3.14.2

### Pipeline Overview
Text extraction into .txt format --> Text preprocessing and preparation for SBERT's semantic comparison between resumes and job descriptions --> 

### Functions
#### Extraction
anonymize():
- Censors PII like name, location, email, social handles, phone number, and websites
process_resumes():
- Core processing function called by CLI commands run() and run_all()
- Uses temporary directory to stage files for processing
- Stores censored files in an in-memory SQLite DB that will delete itself after session finishes
##### CLI Commans
- run(): processes specific PDF files (cherry-picking concept)
- run_all: processes multiple PDF files in a folder path (processes all files in a named folder)
#### Preprocessing

#### SBERT Embedding and Similarity Comparison


