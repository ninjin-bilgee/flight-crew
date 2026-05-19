from storage import save_candidate, export_to_csv

# pretend these are real resumes that got scored
save_candidate("Candidate_1", "sample_resume_1.pdf", 0.85, ["Python", "SQL"])
save_candidate("Candidate_2", "sample_resume_2.pdf", 0.62, ["Java", "Excel"])
save_candidate("Candidate_3", "sample_resume_3.pdf", 0.91, ["Python", "Machine Learning"])

# export the ranked results to a CSV
export_to_csv()

print("Done!")