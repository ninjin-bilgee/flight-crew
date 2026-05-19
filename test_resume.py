# import preprocessing pipeline
from preprocessing import process_resume


# process sample resume PDF
sentences = process_resume("sample_resume.pdf")


# print processed SBERT-ready sentences
print("SBERT READY SENTENCES:")

# print every cleaned sentence
for sentence in sentences:
    print("-", sentence)