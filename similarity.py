import faiss  # the FAISS tool for searching number lists
import numpy as np  # tool for handling number lists
import json  # same json you used in storage.py
import os  # same os you used in storage.py

DIMENSION = 384  # our number lists are always 384 numbers long
index = faiss.IndexFlatL2(DIMENSION)  # create the empty filing cabinet
id_map = []  # empty list that tracks which position belongs to which candidate

def add_to_index(resume_id, embedding):
    # this function takes one resume's number list and files it away
    # resume_id is something like "Candidate_1"
    # embedding is the list of 384 numbers Michelle's code produced
    
    vector = np.array([embedding], dtype='float32')  
    # convert the embedding into the exact format FAISS needs
    
    index.add(vector)  
    # add those numbers into the filing cabinet
    
    id_map.append(resume_id)  
    # remember which candidate this belongs to
    # position 0 = Candidate_1, position 1 = Candidate_2 etc

def save_index(filepath="faiss_index"):
    # this function saves everything to disk so nothing disappears
    # filepath is just the name of the file we're saving to
    
    faiss.write_index(index, filepath + ".bin")  
    # save the filing cabinet itself to a file called faiss_index.bin
    
    with open(filepath + "_map.json", "w") as f:  
    # open a new json file to write to
        json.dump(id_map, f)  
        # save the id_map list so we still know which candidate is which

def load_index(filepath="faiss_index"):
    # this function reloads everything when the server starts back up
    # without this everything would disappear every time the server restarts
    
    global index, id_map  
    # global means we're updating the index and id_map we created at the top
    # not creating new local ones inside this function
    
    if os.path.exists(filepath + ".bin"):  
    # check if a saved file actually exists before trying to load it
        
        index = faiss.read_index(filepath + ".bin")  
        # reload the filing cabinet from the saved file
        
        with open(filepath + "_map.json") as f:  
        # open the saved json file
            id_map = json.load(f)  
            # reload the candidate labels back into memory

def search_index(query_embedding, k=5):
    # this function searches the filing cabinet for the most similar resumes
    # query_embedding is the job description converted to a number list
    # k means how many results to return, default is top 5
    
    query_vector = np.array([query_embedding], dtype='float32')  
    # convert the job description embedding into the format FAISS needs
    
    distances, positions = index.search(query_vector, k)  
    # search the filing cabinet
    # distances = how far apart each resume is from the job description
    # positions = which spots in the filing cabinet were closest
    
    results = []  
    # empty list to store our results
    
    for i, position in enumerate(positions[0]):  
    # loop through each position that was found
        
        if position < len(id_map):  
        # make sure the position actually exists in our id_map
            
            results.append({
                "resume_id": id_map[position],  
                # look up which candidate is at this position
                "score": float(1 - distances[0][i])  
                # convert the distance into a similarity score between 0 and 1
                # closer distance = higher similarity score
            })
    
    return results  
    # return the list of matching candidates with their scores