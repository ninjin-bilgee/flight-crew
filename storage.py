# Storage module for saving and exporting candidate ranking results

import json  # lets us read and write JSON files
import csv   # lets us read and write CSV files
import os    # lets us check if a file exists on the computer

RESULTS_FILE = "results.json"  # the name of the file where we save all candidate results

def save_candidate(candidate_id, filename, score, skills=[]):
    # this function saves one candidate's info into the JSON file
    
    if os.path.exists(RESULTS_FILE):  # check if the results file already exists
        with open(RESULTS_FILE, "r") as f:  # if yes, open it to read
            data = json.load(f)  # load everything already saved in it
    else:
        data = {}  # if no file yet, start with an empty dictionary

    data[candidate_id] = {  # add this candidate to the data using their ID as the key
        "filename": filename,  # the name of their resume file
        "score": score,        # their similarity score (0 to 1)
        "skills": skills       # list of their skills
    }

    with open(RESULTS_FILE, "w") as f:  # open the file to write/save
        json.dump(data, f, indent=2)    # save everything back into the JSON file neatly

def export_to_csv(output_file="ranked_results.csv"):
    # this function takes everything in the JSON and exports it as a ranked CSV

    if not os.path.exists(RESULTS_FILE):  # if there's no results file yet
        print("No results yet.")          # tell the user nothing has been saved
        return                            # stop the function here

    with open(RESULTS_FILE, "r") as f:  # open the JSON file to read
        data = json.load(f)             # load all the candidate data

    with open(output_file, "w", newline="") as f:  # create/open the CSV file to write
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "filename", "score", "skills"])
        # ^ sets up the CSV with these column headers
        writer.writeheader()  # writes the header row at the top of the CSV

        for candidate_id, info in sorted(data.items(), key=lambda x: x[1]["score"], reverse=True):
        # ^ loops through candidates sorted by score, highest first
            writer.writerow({
                "candidate_id": candidate_id,          # write their ID
                "filename": info["filename"],           # write their resume filename
                "score": round(info["score"], 4),       # write their score rounded to 4 decimal places
                "skills": ", ".join(info.get("skills", []))  # write their skills as a comma separated string
            })

    print(f"Exported to {output_file}")  # tell the user the CSV was created successfully