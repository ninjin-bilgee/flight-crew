# This is a simple test script to verify that the PyMuPDF library can read PDF files correctly.
# Takeaway: Has a better performance in structural maintenance and text quality.
# Challenge: None so far.

# Version: PyMuPDF 1.18.19

# RUN COMMAND: python test_scripts/test_pymupdf_reader.py

import pymupdf
import pymupdf4llm

doc = pymupdf.open('resumes_pdf/sample_resume_19.pdf')
for page in doc:
    print(page.get_text())

md = pymupdf4llm.to_markdown('resumes_pdf/sample_resume_19.pdf')
print(md)

