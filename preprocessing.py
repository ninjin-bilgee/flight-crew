# importing libraries used to extract text and regex text cleaning
import re

# section headers that should NOT contribute to job requirement similarity
EXCLUDED_JD_HEADERS = [

    "skills to gain",
    "what you will learn",
    "preferred learning",
    "nice to have",
    "bonus skills",
    "growth opportunities",
    "future skills",
    "training provided",
    "learn",
    "opportunities to learn"
]

# valid section headers that STOP skipping
VALID_HEADERS = {

    "required",
    "requirements",
    "responsibilities",
    "qualifications",
    "about the role",
    "preferred qualifications"
}

# this loads extracted txt resume file
def load_txt_file(filepath):

    # opening txt file with utf-8 encoding
    with open(filepath, "r", encoding="utf-8") as file:

        # read all text from file
        text = file.read()

    # return raw extracted text
    return text


# removes non-requirement JD sections
def filter_jd_sections(text):

    # split text into lines
    lines = text.splitlines()

    filtered_lines = []

    skip_section = False

    for line in lines:

        # clean line
        stripped = line.strip()

        # skip empty lines
        if not stripped:
            continue

        # detect likely headers
        is_header = len(stripped) < 60

        if is_header:

            lower_header = stripped.lower()

            # start skipping excluded sections
            if any(
                keyword in lower_header
                for keyword in EXCLUDED_JD_HEADERS
            ):

                skip_section = True
                continue

            # ONLY stop skipping if we hit
            # a real new section header
            VALID_HEADERS = {

                "required",
                "requirements",
                "responsibilities",
                "qualifications",
                "about the role",
                "preferred qualifications"
            }

            if lower_header in VALID_HEADERS:
                skip_section = False
    

        # keep non-excluded content
        if not skip_section:
            filtered_lines.append(stripped)

    return "\n".join(filtered_lines)


# cleans text for SBERT embeddings
def clean_text_for_sbert(raw_text):

    # lowercase text for consistent comparisons
    text = raw_text.lower()


    text = re.sub(r'\bremoved\b', ' ', text)

    # my OCR cleanup
    # fixes broken spacing/newlines/tabs
    text = re.sub(r'[\r\n\t]+', ' ', text)

    text = re.sub(r'_+', ' ', text)
    
    # normalize whitespace/newlines into single spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # lightly remove weird special characters
    text = re.sub(r'[^\w\s.,()-]', ' ', text)

    # normalize spaces again after regex cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    # return cleaned text
    return text


# this extracts meaningful keywords
def extract_keywords(text):

    STOPWORDS = {

        "the", "and", "for", "with",
        "this", "that", "from", "have",
        "will", "your", "our", "are",
        "you", "their", "about",

        # generic resume words
        "using",
        "built",
        "experience",
        "worked",
        "developed"
    }

    words = re.findall(r'\b[a-zA-Z][a-zA-Z+\-#.]+\b', text)

    keywords = []

    for word in words:

        clean_word = word.lower()

        # this remove short/generic tokens
        if (
            len(clean_word) > 2
            and clean_word not in STOPWORDS
        ):

            keywords.append(clean_word)

    # remove duplicates while preserving order
    unique_keywords = list(dict.fromkeys(keywords))

    return unique_keywords


# full preprocessing pipeline
def process_resume(
    txt_filepath,
    is_job_description=False
):

    # loading txt file
    raw_text = load_txt_file(txt_filepath)

    # apply JD filtering ONLY to JDs
    if is_job_description:
        raw_text = filter_jd_sections(raw_text)

    # clean text
    cleaned_text = clean_text_for_sbert(raw_text)

    return cleaned_text