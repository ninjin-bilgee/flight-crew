# importing libraries used to extract text and regex text cleaning
import pdfplumber
import re

def extract_raw_text(filepath):
    text = ""

    # opening pdf file
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "
            
    return text

# cleaning the extracted text
def clean_text(text):

    # converting to lowercase
    text = text.lower()

    # removing punc/sym and returning
    return text

# this is the full preprocessing pipeline
def process_resume(filepath):

    # extracting raw text from resume/clean extracted text
    raw_text = extract_raw_text(filepath)

    cleaned_text = clean_text(raw_text)

    return cleaned_text

# here I will extract important skills/keywords
def extract_keywords(text):

    keywords = [
        "python",
        "java",
        "sql",
        "react",
        "aws",
        "docker",
        "flask",
        "javascript",
        "typescript"
    ]

    found_keywords = []

    # looping thru every word in text
    for word in text.split():

        if word in keywords:

            found_keywords.append(word)

    return list(set(found_keywords))


