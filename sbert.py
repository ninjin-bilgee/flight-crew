from sentence_transformers import SentenceTransformer

#Load the model (this will download it the first time you run it)
model = SentenceTransformer('all-MiniLM-L6-v2')

#Sample sentences
resume_text = "Experienced Python developer with a focus on machine learning."
job_description = "Seeking a Machine Learning Engineer proficient in Python."

#Generate embeddings
resume_embedding = model.encode(resume_text)
job_embedding = model.encode(job_description)

#Print the results
print(f"Resume Embedding Type: {type(resume_embedding)}")
print(f"Resume Embedding Shape: {resume_embedding.shape}")
print(f"First 5 numbers: {resume_embedding[:5]}")

from sentence_transformers import util

#Calculate cosine similarity
similarity = util.cos_sim(resume_embedding, job_embedding)

print(f"Similarity Score: {similarity.item()}")
