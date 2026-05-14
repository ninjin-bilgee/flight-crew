import pymupdf
import os

pdf_path = 'resumes_pdf/sample_resume_19.pdf'

doc = pymupdf.open(pdf_path)
md = "".join(page.get_text() for page in doc)

filename = os.path.splitext(os.path.basename(pdf_path))[0] 
output_path = f'resumes_extracted_txt/{filename}.txt'

os.makedirs('resumes_extracted_txt', exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(md)

print(f"Saved to {output_path}")


