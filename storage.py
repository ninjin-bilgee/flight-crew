# Storage Module: Export Ranking Results
"""
- Exports a rank_resumes() result dict to CSV
- Shared utility so both the CLI runner and the Streamlit app produce identical output
- write_ranking_csv() writes a file (for CLI use)
- ranking_to_csv_string() returns a CSV string (for Streamlit's st.download_button)
"""

import csv
import io

# CSV column order used by both export functions
_FIELDNAMES = ["rank", "filename", "similarity_score"]


def _write_rows(writer, ranking):
    """Write the header and one row per ranked result into the given csv writer."""
    writer.writeheader()
    for r in ranking["results"]:
        writer.writerow({
            "rank": r["rank"],
            "filename": r["filename"],
            "similarity_score": round(r["similarity_score"], 4),
        })


"""
Function: writes a ranking result dict to a CSV file on disk
- intended for the CLI runner, where a file on disk is what the user wants
Args:
    ranking:     the dict returned by rank_resumes() (must contain 'results')
    output_path: where to write the CSV
Returns: the output path
"""
def write_ranking_csv(ranking, output_path="ranked_results.csv"):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        _write_rows(writer, ranking)
    return output_path


"""
Function: builds the same CSV in memory and returns it as a string
- intended for Streamlit: st.download_button needs string/bytes data, not a file path
Args:
    ranking: the dict returned by rank_resumes() (must contain 'results')
Returns: the CSV content as a string
"""
def ranking_to_csv_string(ranking):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=_FIELDNAMES)
    _write_rows(writer, ranking)
    return buffer.getvalue()