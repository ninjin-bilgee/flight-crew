from preprocessing import extract_keywords

sample_resume = """
Python developer with FastAPI experience.
Built machine learning pipelines using SQL and Docker.
"""

keywords = extract_keywords(sample_resume)

print(keywords)