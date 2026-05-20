# import preprocessing pipeline
from preprocessing import process_resume


# process extracted txt resume
cleaned_resume = process_resume(
    "resumes_extracted_txt/Candidate_1.txt"
)


# print cleaned output
print("CLEANED RESUME:")

print(cleaned_resume)