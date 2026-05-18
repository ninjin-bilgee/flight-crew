# importing the preprocessing functions
from preprocessing import process_resume
from preprocessing import extract_keywords

# processing sample resume pdf
resume_text = process_resume("sample_resume.pdf")

print("CLEANED TEXT:")
print(resume_text)

# printing extracted keywords
print("\nKEYWORDS:")
print(extract_keywords(resume_text))