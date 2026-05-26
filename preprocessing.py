# importing libraries used to extract text and regex text cleaning
import re

# section headers that should NOT contribute to job requirement similarity
EXCLUDED_JD_HEADERS = [
    "skills to gain",
    "what you will learn",
    "what skills will",     
    "what the intern will",
    "preferred learning",
    "nice to have",
    "bonus skills",
    "growth opportunities",
    "future skills",
    "training provided",
    "opportunities to learn",
    "gain exposure to",
    "will learn",  
    "pay & benefits",
    "pay and benefits",
    "benefits and perks",
    "compensation and benefits",
    "contact us",
    "contact information",
    "how to apply",
    "about the company",
]

# ONLY stop skipping if we hit a real new section header
VALID_HEADERS = {

    "required",
    "requirements",
    "responsibilities",
    "qualifications",
    "about the role",
    "preferred qualifications",
    "experience",
    "what we're looking for",
    "what we need",
    "what you bring",
    "what you should have",
    "what you must have",
    "language ability",
    "language abilities",
    "duties",
    "duties and responsibilities",
}

# Sentence-level exclusions (catches stuff buried in paragraphs)
FUTURE_SKILL_PHRASES = [
    "the intern will gain exposure to",
    "you will gain exposure to",
    "will have the opportunity to learn",
    "opportunity to gain experience",
    "gain exposure to",
    "will gain exposure",
    "you will learn",
    "will be trained on",
    "training will be provided",
    "will receive training",
    "opportunity to learn",
    "introduction to",         
]

# ending phrases that indicate boilerplate non-requirement sections we can skip
# marketing pitches etc
JD_BOILERPLATE_PHRASES = [
    "we offer a competitive",
    "we offer competitive",
    "competitive salary",
    "comprehensive benefits",
    "benefits package",
    "please send your application",
    "send your application",
    "send your resume",
    "how to apply",
    "to apply please",
    "deadline for submitting",
    "application deadline",
    "equal opportunity employer",
    "submit your application",
]

# Return True if a sentence describes skills to be gained, not required
def _sentence_contains_future_phrase(sentence: str) -> bool:
    lower = sentence.lower()
    return any(phrase in lower for phrase in FUTURE_SKILL_PHRASES)

"""
Remove individual sentences that describe what the candidate WILL learn,
rather than what they need to already know.

Drops a trigger line AND everything after it until the next section header (## ...). 
Handles inline lists, semicolons, bullets — any format after the trigger.
"""
def filter_excluded_sentences(text: str) -> str:

    lines = text.splitlines()
    filtered = []
    skip_until_header = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # A real section header restarts normal processing
        is_section_header = stripped.startswith('#') or any(
            h in lower for h in VALID_HEADERS
        )

        if skip_until_header:
            if not stripped:
                continue
            if is_section_header:
                skip_until_header = False   
                filtered.append(line)
            continue

        # Trigger: line contains a future-skill phrase
        if any(phrase in lower for phrase in FUTURE_SKILL_PHRASES + JD_BOILERPLATE_PHRASES):
            skip_until_header = True
            continue 

        filtered.append(line)

    return " ".join(filtered)

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

            # strip markdown syntax + punctuation before matching
            lower_header = re.sub(r'[#*:]', '', stripped).strip().lower()

            # start skipping excluded sections
            if any(kw in lower_header for kw in EXCLUDED_JD_HEADERS):
                skip_section = True
                continue

            if any(vh in lower_header for vh in VALID_HEADERS):
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
        # Pass 1: drop entire excluded sections by header
        raw_text = filter_jd_sections(raw_text)

        # Pass 2: drop individual sentences with "gain exposure to" etc
        raw_text = filter_excluded_sentences(raw_text)

    # clean text
    cleaned_text = clean_text_for_sbert(raw_text)

    return cleaned_text