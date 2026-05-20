import faiss #help search numbers of resume embeddings
import numpy as np #store embedding
import json # store resume
import os #check if file exist

class ResumeRanker:
    #runs auto when theres a new ResumeRanker object created
    def __init__(self, embedding_dim=384): # dimension of embedding vectors
        #Initializes an empty FAISS index for the resumes.
        self.index = faiss.IndexFlatIP(embedding_dim)  # Use Inner Product for cosine similarity
        self.resume_metadata = []  # Stores filename and a path to the cleaned text
    
    """
    Builds the FAISS index from a list of resume embeddings.
    Args:
        resume_embeddings_list: A list of embedding vectors.
        resume_metadata_list: A list of dictionaries, each with 'filename' and 'cleaned_text_path'.
    """
    def build_index(self, resume_embeddings_list, resume_metadata_list):
        # Ensure embeddings are in the correct float32 format FAISS expects
        embeddings_array = np.array(resume_embeddings_list).astype('float32') #embedding to numpy array
        self.index.add(embeddings_array)
        self.resume_metadata = resume_metadata_list
        print(f"Index built with {self.index.ntotal} resumes.")

    """
    Searches the FAISS index for the top K most similar resumes.
    Args:
        query_embedding: The embedding of the job description.
        k: The number of top matches to return.
    Returns:
        A list of dictionaries, each containing the rank, filename, and similarity score.
    """
    def search(self, query_embedding, k=10):
        #convert jd embedding to numpy array (384,) -> (1, 384)
        query_array = np.array(query_embedding).astype('float32').reshape(1, -1)
        #FAISS return 2 arrays one for scoreing and one for position number for matching resume
        scores, indices = self.index.search(query_array, k) 

        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            results.append({
                'rank': i + 1,
                'filename': self.resume_metadata[idx]['filename'],
                'similarity_score': float(scores[0][i])
            })
        return results
    
    """Saves the FAISS index and metadata to disk."""
    def save_index(self, path):
        faiss.write_index(self.index, path)
        with open(f"{path}.meta.json", 'w') as f:
            json.dump(self.resume_metadata, f)

    """Loads the FAISS index and metadata from disk."""
    def load_index(self, path):
        
        self.index = faiss.read_index(path)
        with open(f"{path}.meta.json", 'r') as f:
            self.resume_metadata = json.load(f)