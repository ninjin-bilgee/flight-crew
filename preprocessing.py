# importing libraries used to extract text and regex text cleaning
import pdfplumber
import re
from nltk.tokenize import sent_tokenize

def extract_raw_text(filepath):
    text = ""

    # opening pdf file
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
            # adding page text to full text string
                text += page_text + "\n"
    
    # removing extra whitespace and returning
    return text.strip()

# cleaning the extracted text whule preserving sentence meaning for SBERT
def clean_text_for_sbert(raw_text):

    # normalizing whitespace
    text = re.sub(r'\s+', ' ', raw_text).strip()

    # removing non-english characters while keeping punctuation
    text = re.sub(r'[^\x00-\x7f]+', '', text)

    # splitting text into sentences
    sentences = sent_tokenize(text)

    # removing empty sentences and trim spaces
    cleaned_sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return cleaned_sentences

# this is the full preprocessing pipeline
def process_resume(filepath):

    # extracting raw text from resume/clean extracted text
    raw_text = extract_raw_text(filepath)

    cleaned_sentences = clean_text_for_sbert(raw_text)

    return cleaned_sentences
