# importing libraries used to extract text and regex text cleaning
import re

# this loads extracted txt resume file
def load_txt_file(filepath):

    # opening txt file with utf-8 encoding
    with open(filepath, "r", encoding="utf-8") as file:

        # read all text from file
        text = file.read()

    # return raw extracted text
    return text


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


# full preprocessing pipeline
def process_resume(txt_filepath):

    # load extracted txt file
    raw_text = load_txt_file(txt_filepath)

    
    cleaned_text = clean_text_for_sbert(raw_text)

    
    return cleaned_text