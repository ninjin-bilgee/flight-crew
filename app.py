# streamlit is the library that builds the web UI (buttons, file uploaders, etc.)
import streamlit as st

# tempfile lets us create a temporary folder on disk that auto-deletes when done
import tempfile

# os lets us do file system stuff like check if a file exists or build file paths
import os

# shutil lets us delete entire folders at once (used for clearing the embeddings cache)
import shutil

# sqlite3 is Python's built-in library for talking to SQLite databases
import sqlite3

# ── page config ────────────────────────────────────────────────────────────────

# set the browser tab title and use centered layout instead of full-width
st.set_page_config(page_title="Resume Parser & Ranker", layout="centered")

# display the main title at the top of the page
st.title("📄 Resume Parser & Ranker")

# ── helper functions ───────────────────────────────────────────────────────────

def _clear_session_data():
    # this function deletes leftover data from a previous run
    # so each new session starts completely fresh

    # path to the SQLite database file
    db_path = "candidate_map.db"

    # path to the folder where embeddings are cached as .npy files
    emb_path = "data/embeddings/"

    # if the database file exists, delete it
    if os.path.exists(db_path):
        os.remove(db_path)

    # if the embeddings folder exists, delete the whole folder and everything inside
    if os.path.exists(emb_path):
        shutil.rmtree(emb_path)


def _save_uploads_to_tmpdir(uploaded_files, tmpdir: str) -> list[str]:
    # streamlit gives us uploaded files as in-memory objects, not real files on disk
    # but our extraction engine needs actual file paths to open
    # so this function writes each uploaded file to a real temporary folder on disk

    # empty list to collect the file paths we create
    paths = []

    for f in uploaded_files:
        # build the full destination path e.g. /tmp/abc123/resume1.pdf
        dest = os.path.join(tmpdir, f.name)

        # open that path and write the file's bytes to disk
        with open(dest, "wb") as out:
            out.write(f.read())

        # add the path to our list
        paths.append(dest)

    # return the list of real file paths so the extraction engine can use them
    return paths


# ── sidebar ────────────────────────────────────────────────────────────────────

# everything inside this block appears in the left sidebar panel
with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
1. Upload one or more **resume PDFs**
2. Paste the **job description** text
3. Choose how many top candidates to show
4. Click **Rank Resumes**
        """
    )

# ── main UI inputs ─────────────────────────────────────────────────────────────

# section label for the file uploader
st.subheader("Upload Resumes")

# file uploader widget — only accepts PDFs, allows multiple files at once
# resume_files will be a list of uploaded file objects (or empty list if nothing uploaded)
resume_files = st.file_uploader(
    "Upload PDF resumes",
    type="pdf",
    accept_multiple_files=True,
)

# section label for the job description input
st.subheader("Job Description")

# large text box where the user pastes the job description
# jd_text will be a string containing whatever the user typed/pasted
jd_text = st.text_area(
    "Paste the full job description here",
    height=220,
    placeholder="Copy and paste the job posting text…",
)

# slider to choose how many top results to display (1 to 20, default 5)
k = st.slider("How many top candidates to show?", min_value=1, max_value=20, value=5)

# the main action button — when clicked, run will be True for that one render cycle
run = st.button("Rank Resumes", type="primary", use_container_width=True)

# ── pipeline runs only when the button is clicked ──────────────────────────────

if run:

    # --- input validation: stop early if inputs are missing ---

    # if no files were uploaded, show a warning and stop
    if not resume_files:
        st.warning("Please upload at least one resume PDF.")
        st.stop()

    # if the job description box is empty (or just whitespace), show a warning and stop
    if not jd_text.strip():
        st.warning("Please paste a job description.")
        st.stop()

    # delete any leftover database or embeddings from a previous session
    _clear_session_data()

    # create a temporary directory on disk — it will be cleaned up in the finally block
    tmpdir_obj = tempfile.TemporaryDirectory()

    # get the actual folder path as a string e.g. /tmp/abc123
    tmpdir = tmpdir_obj.name

    try:

        # show a loading spinner while resumes are being extracted
        with st.spinner("Extracting and processing resumes…"):

            # write the uploaded PDF files from memory to real files in the temp folder
            pdf_paths = _save_uploads_to_tmpdir(resume_files, tmpdir)

            # import extract_resumes and the DB connection from the extraction engine
            # we import here (not at the top) so the DB connection is brand new
            # after _clear_session_data() deleted the old database
            from text_extraction_engine import extract_resumes, conn as db_conn

            # run the extraction pipeline on each PDF
            # this anonymizes each resume, cleans the text, and saves it to SQLite
            extract_resumes(pdf_paths)

        # show a loading spinner while embeddings are being generated
        with st.spinner("Building search index…"):

            # import the embedding function and ranker classes
            from embedder import get_embedding
            from ranker import ResumeRanker
            import ranker_service

            # read all the candidates that extract_resumes just saved to the database
            rows = db_conn.execute(
                "SELECT candidate_id, filename, cleaned_text FROM candidates"
            ).fetchall()

            # if somehow nothing got extracted, show an error and stop
            if not rows:
                st.error("No candidates could be extracted from the uploaded PDFs.")
                st.stop()

            # empty lists to hold embeddings and their matching metadata
            embeddings = []
            metadata = []

            # loop through each candidate row from the database
            for candidate_id, filename, cleaned_text in rows:

                # convert the cleaned resume text into a 768-dimensional embedding vector
                # resume_id is passed so the embedding gets cached to disk
                emb = get_embedding(cleaned_text, resume_id=candidate_id)

                # add the embedding vector to our list
                embeddings.append(emb)

                # store the filename and candidate ID so we can show them in results
                metadata.append({"filename": filename, "resume_id": candidate_id})

            # create a new FAISS index — this is the search engine that finds similar resumes
            # embedding_dim=768 must match the size of vectors our model produces
            ranker_service._ranker = ResumeRanker(embedding_dim=768)

            # load all the embeddings into the FAISS index so it can search them
            ranker_service._ranker.build_index(embeddings, metadata)

        # show a loading spinner while ranking is running
        with st.spinner("Ranking candidates against the job description…"):

            # run the full ranking pipeline:
            # 1. FAISS finds candidates by embedding similarity
            # 2. cross-encoder re-scores each one by reading JD + resume together
            # 3. results are sorted and returned with normalized 0-1 scores
            ranking = ranker_service.rank_resumes(jd_text.strip(), k=k)

        # --- display results ---

        # pull just the list of ranked candidates out of the returned dict
        results = ranking["results"]

        # show the results section header
        st.subheader("🏆 Top Candidates")

        # if weak_batch is True, no candidate was a strong match — warn the user
        if ranking.get("weak_batch"):
            st.warning(
                "No strong matches found — even the top candidate is a weak fit "
                "for this job description. Consider uploading more resumes or revising the JD."
            )

        # loop through each ranked result and display it
        for r in results:

            # split each row into two columns: 3/4 width for name, 1/4 for score
            col1, col2 = st.columns([3, 1])

            with col1:
                # show the rank number and resume filename in bold
                st.write(f"**#{r['rank']} — {r['filename']}**")

            with col2:
                # convert 0-1 score to a percentage for display
                score_pct = r["similarity_score"] * 100

                # pick a color emoji based on how strong the match is
                color = (
                    "Strong match" if score_pct >= 60    # strong match
                    else "Moderate match" if score_pct >= 40  # moderate match
                    else "Weak match"                      # weak match
                )

                # show the colored emoji and the percentage score
                st.write(f"{color} {score_pct:.1f}%")

            # show a visual progress bar representing the score (0.0 to 1.0)
            st.progress(r["similarity_score"])

            # add a horizontal divider line between candidates
            st.divider()

        # --- CSV download button ---

        # import csv for writing, io for building the file in memory (no disk write needed)
        import csv, io

        # create an in-memory text buffer to write the CSV into
        buf = io.StringIO()

        # set up a CSV writer with these three column headers
        writer = csv.DictWriter(buf, fieldnames=["rank", "filename", "score"])

        # write the header row
        writer.writeheader()

        # write one row per result
        for r in results:
            writer.writerow({
                "rank": r["rank"],
                "filename": r["filename"],
                # round score to 4 decimal places to keep it clean
                "score": round(r["similarity_score"], 4),
            })

        # show a download button — clicking it downloads the CSV to the user's computer
        st.download_button(
            label="⬇Download Results as CSV",
            data=buf.getvalue(),       # the CSV content as a string
            file_name="ranked_results.csv",
            mime="text/csv",           # tells the browser this is a CSV file
        )

    except Exception as e:
        # if anything crashes, show the error message in the UI
        st.error(f"Something went wrong: {e}")

        # also re-raise so the full traceback prints in the terminal for debugging
        raise

    finally:
        # this block ALWAYS runs, even if there was an error
        # clean up the temporary folder with the uploaded PDFs
        tmpdir_obj.cleanup()

        # delete the database and embeddings cache so nothing leaks to the next session
        _clear_session_data()