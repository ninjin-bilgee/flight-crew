# Ranker Module: FAISS Index for Resume Embeddings
"""
- Wraps a FAISS index that stores resume embeddings
- Builds the index from embedding vectors, searches it against a job description embedding
- Saves and loads the index + metadata to/from disk
"""

import faiss      # vector search library — finds nearest resume embeddings
import numpy as np # array handling for embedding vectors
import json       # save/load resume metadata to disk
import os         # check if files exist

class ResumeRanker:
    """
    Function: initializes an empty FAISS index for resume embeddings
    - runs automatically when a new ResumeRanker object is created
    """
    def __init__(self, embedding_dim=768): # dimension of embedding vectors (all-mpnet-base-v2 = 768)
        # Inner Product index — used only for pool retrieval; the cross-encoder does the real ranking
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.resume_metadata = []  # stores per-resume metadata (e.g. filename) aligned with index order

    """
    Function: builds the FAISS index from a list of resume embeddings
    Args:
        resume_embeddings_list: A list of embedding vectors.
        resume_metadata_list: A list of dictionaries, each with 'filename' and 'cleaned_text_path'.
    """
    def build_index(self, resume_embeddings_list, resume_metadata_list):
        # convert the embeddings to a float32 numpy array — the format FAISS expects
        embeddings_array = np.array(resume_embeddings_list).astype('float32')
        # add all embeddings to the index
        self.index.add(embeddings_array)
        # store metadata in the same order as the embeddings so indices line up
        self.resume_metadata = resume_metadata_list
        print(f"Index built with {self.index.ntotal} resumes.")

    """
    Function: searches the FAISS index for the top K most similar resumes
    Args:
        query_embedding: The embedding of the job description.
        k: The number of top matches to return.
    Returns:
        A list of dictionaries, each containing the rank, filename, and similarity score.
    """
    def search(self, query_embedding, k=10):
        # reshape the JD embedding to a 2D array (1, dim) — FAISS expects a batch of queries
        query_array = np.array(query_embedding).astype('float32').reshape(1, -1)
        # FAISS returns two arrays: similarity scores and the matching index positions
        scores, indices = self.index.search(query_array, k)

        results = []
        # iterate over the returned index positions
        for i, idx in enumerate(indices[0]):
            # FAISS returns -1 for empty slots when fewer than k results exist — skip them
            if idx == -1:
                continue
            # map each index back to its filename via the stored metadata
            results.append({
                'rank': i + 1,
                'filename': self.resume_metadata[idx]['filename'],
                'similarity_score': float(scores[0][i])
            })
        return results

    """
    Function: saves the FAISS index and its metadata to disk
    - index is written as a FAISS binary file, metadata as a sidecar .meta.json file
    """
    def save_index(self, path):
        faiss.write_index(self.index, path)
        with open(f"{path}.meta.json", 'w') as f:
            json.dump(self.resume_metadata, f)

    """
    Function: loads a previously saved FAISS index and its metadata from disk
    - lets ranking run without rebuilding the index from scratch
    """
    def load_index(self, path):
        # read the FAISS binary index file
        self.index = faiss.read_index(path)
        # read the sidecar metadata file
        with open(f"{path}.meta.json", 'r') as f:
            self.resume_metadata = json.load(f)