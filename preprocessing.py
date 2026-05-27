# Preprocessing Engine: Clean Extracted Text for SBERT Embedding
"""
Prepares raw text for SBERT embedding by:
- Stripping boilerplate sections that don't reflect candidate skills
- Removing sentences that describe what the candidate will learn rather than what they need to know
- Cleaning OCR artifacts and normalizing whitespace
- Extracting keywords (not currently used in ranking but available for future features)
"""

import re

# JD: section headers that should NOT contribute to job requirement similarity
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

# JD: valid new section headers in a JD
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

# JD: sentence-level exclusions (catches stuff buried in paragraphs)
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

# JD: ending phrases that indicate boilerplate non-requirement sections we can skip
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

# JD: Workday application-export boilerplate
# carry no signal
WORKDAY_NOISE_PATTERNS = [
    r'for:?\s+\d{3,}[\w\s]*?(?:leader|manager|associate|clerk|member|supervisor)',         
    r'view job application',
    r'added by external career site',
    r'page\s+\d+\s+of\s+\d+',
    r'\d{1,2}\s+\d{2}\s+[ap]m\s+\d{2}\s+\d{2}\s+\d{4}',  
    r'picture\s+\d+\s*x\s*\d+\s*intentionally omitted',
    r'-+\s*start of picture text\s*-+',
    r'-+\s*end of picture text\s*-+',
    r'jobs applied to',
    r'candidate information',
    r'\busername\b',
    r'none entered',
    r'overview overview',
    r'application name',
]


"""
Function: header-driven filter to filter out irrelevant sections of the JD
- stops matching (starts skipping) when it hits a section header that matches an EXCLUDED_JD_HEADERS keyword
- resumes matching (stops skipping) when it hits a section header that matches a VALID_HEADERS keyword
- returns lines of text separated by newlines for filter_excluded_sentences
"""
def filter_jd_sections(text):

    # split text into lines
    lines = text.splitlines()
    filtered = []
    # state variable to track whether we're currently in an excluded section
    skip_section = False

    for line in lines:
        # clean line of whitespace
        stripped = line.strip()
        # if nothing left, skip it
        if not stripped:
            continue

        # detect likely headers
        # threshold: if the line is short (<60 chars)
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
            filtered.append(stripped)

    return "\n".join(filtered)



"""
Function: line-by-line filter to filter out irrelevant sentences and sections from JD
- stops matching (starts skipping) when it hits a sentence that contains a FUTURE_SKILL_PHRASE or JD_BOILERPLATE_PHRASE
- resumes matching (stops skipping) when it hits a section header that matches a VALID_HEADERS keyword
- relies on filter_excluded_sentences to join by newlines
- returns one block of text with relevant sections separated by spaces
- run AFTER filter_jd_sections (complementary filter)
"""
def filter_excluded_sentences(text: str) -> str:

    # split text into lines for processing
    lines = text.splitlines()
    filtered = []
    # state variable to track whether we're currently skipping lines until the next header
    skip_until_header = False

    for line in lines:
        # clean line of whitespace
        stripped = line.strip()
        # lowercase for case-insensitive matching
        lower = stripped.lower()

        # a real section header restarts normal processing
        # check for likely headers either from VALID_HEADERS or section-like formatting (e.g. markdown '## Header')
        is_section_header = stripped.startswith('#') or any(
            h in lower for h in VALID_HEADERS
        )

        # if we hit a section header, we can stop skipping and resume normal processing
        # - decreases noise
        if skip_until_header:
            if not stripped:
                continue
            if is_section_header:
                skip_until_header = False  
                # append this section content to filtered since it's under a valid header 
                filtered.append(line)
            continue

        # Trigger: line contains a future-skill phrase or boilerplate phrase
        if any(phrase in lower for phrase in FUTURE_SKILL_PHRASES + JD_BOILERPLATE_PHRASES):
            # this triggers skipping again
            skip_until_header = True
            continue 

        # if we get here, this line is not excluded and we're not in a skip state, so we keep it
        filtered.append(line)

    # join the filtered lines back into a single text block
    return " ".join(filtered)



"""
Function: cleans text for SBERT embeddings
- lowercases for more consistent embedding space comparisons
- strips Workday boilerplate that would otherwise dominate the embedding space and cause poor matches
- removes common OCR artifacts (e.g. 'e' in place of bullets)
- normalizes whitespaces and newlines to normal spaces 
- removes weird special characters
- returns cleaned text ready for embedding
"""
def clean_text_for_sbert(raw_text):
    # lowercase text for consistent comparisons
    text = raw_text.lower()

    # Workday boilerplate strip layer
    for pattern in WORKDAY_NOISE_PATTERNS:
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    # remove stray column-break 'br' tokens left by multi-column extraction
    text = re.sub(r'\bbr\b', ' ', text)
    # end of workday strip layer

    # OCR bullet artifacts: '●' often OCRs as a stray 'e'.
    # Strip ' - e ' and ' e ' sequences that are orphaned bullet markers.
    text = re.sub(r'\s+-\s+e\s+', ' ', text)      # "- e point of sale" -> " point of sale"
    #text = re.sub(r'(?:^|\s)e\s+(?=[a-z])', ' ', text)  # leading orphan 'e' before a word
    # stray 'ae' at the very start is an OCR header artifact
    text = re.sub(r'^\s*ae\s+', '', text)

    text = re.sub(r'\bremoved\b', ' ', text)

    # fixes broken spacing/newlines/tabs
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'_+', ' ', text)

    # normalize whitespace/newlines into single spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # lightly remove weird special characters
    text = re.sub(r'[^\w\s.,()-]', ' ', text)

    # normalize spaces again after regex cleanup
    text = re.sub(r'\s+', ' ', text).strip()

    return text

"""
Function: extracts keywords from text for potential future use in enhanced ranking features or explainability
- currently NOT USED in the ranking pipeline but available for future iterations
"""
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

