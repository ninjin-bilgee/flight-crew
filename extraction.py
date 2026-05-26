import pymupdf 
import pymupdf
import spacy
import re
import pymupdf4llm
import os
import hashlib
import sys
import shutil
import tempfile
import sqlite3

def _extract_page_columns(page):
    """
    Extract one page's text, detecting single vs. two-column layout.
    Returns the page text in correct reading order.
    """
    # Get text blocks: each is (x0, y0, x1, y1, text, block_no, block_type)
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    if not blocks:
        return ""

    page_width = page.rect.width
    mid = page_width / 2

    # Classify each block as left-of-center or right-of-center by its start x
    left_blocks  = [b for b in blocks if b[0] < mid]
    right_blocks = [b for b in blocks if b[0] >= mid]

    # Decide: is this genuinely a two-column page?
    # Conservative — only split if BOTH sides hold a real share of the content.
    # A single-column page with one indented block won't trip this.
    total = len(blocks)
    is_two_column = (
        len(left_blocks)  >= max(2, total * 0.2) and
        len(right_blocks) >= max(2, total * 0.2)
    )

    if is_two_column:
        # Read the full left column top-to-bottom, then the full right column
        left_blocks.sort(key=lambda b: b[1])   # sort by y (top to bottom)
        right_blocks.sort(key=lambda b: b[1])
        ordered = left_blocks + right_blocks
    else:
        # Single column — just read everything top-to-bottom
        ordered = sorted(blocks, key=lambda b: b[1])

    return "\n".join(b[4].strip() for b in ordered)


def extract_text(pdf_path):
    """
    Adaptive PDF text extraction.
    - Per page: detects single vs. two-column layout and reads in correct order.
    - If a page yields almost no text (image-based PDF), falls back to
      pymupdf4llm for that page, which OCRs it automatically.
    """
    doc = pymupdf.open(pdf_path)
    page_texts = []
    needs_ocr_fallback = False

    for page in doc:
        page_text = _extract_page_columns(page)
        # Almost no text usually means an image-based page → flag for OCR fallback
        if len(page_text.strip()) < 50:
            needs_ocr_fallback = True
        page_texts.append(page_text)

    doc.close()

    # If any page came back nearly empty, re-extract the whole doc with
    # pymupdf4llm, which runs Tesseract OCR on image-based pages.
    if needs_ocr_fallback:
        print(f"  {pdf_path}: sparse text detected, falling back to pymupdf4llm/OCR")
        return pymupdf4llm.to_markdown(pdf_path)

    return "\n".join(page_texts)