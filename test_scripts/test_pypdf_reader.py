# This is a simple test script to check if the PyPDF library is working correctly.
# Takeaway: PyPDF does not do well with structural maintenance.
# Challenge: It will be hard for sectional separation in data.

# Version: PyPDF 3.0.0

# RUN COMMAND: python test_scripts/test_pypdf_reader.py

from pypdf import PdfReader

reader = PdfReader('resumes_pdf/sample_resume_1.pdf')
print(len(reader.pages))
page = reader.pages[0]
text = page.extract_text()
print(text)