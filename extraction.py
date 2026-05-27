# PDF Text Extraction with Layout Detection and OCR Fallback
import pymupdf 
import pymupdf4llm

"""
Function: Extracts one page's text; handles single vs 2-column cases
Returns: Page's text in correct reading order
"""
def _extract_page_columns(page):
    # get text blocks: each is (x0, y0, x1, y1, text, block_no, block_type)
    # strip meaningless image-only or empty blocks
    blocks = [b for b in page.get_text("blocks") if b[4].strip()]
    if not blocks:
        return ""
    
    # assign the page's own width to a variable
    page_width = page.rect.width
    # midpoint of the page width for classifying left and right half of page
    mid = page_width / 2

    # classify each block as left-of-center or right-of-center by its start x
    # [0] - x0 coordinate, horizontal left of block
    left_blocks  = [b for b in blocks if b[0] < mid]
    right_blocks = [b for b in blocks if b[0] >= mid]

    # Final identification for genuine 2-col layout
    # only split if BOTH sides hold a real share of the content 
    total = len(blocks)
    # require at least 20% of blocks on each side
    # AND at least 2 blocks on each side to avoid false positives on small pages
    is_two_column = (
        len(left_blocks)  >= max(2, total * 0.2) and
        len(right_blocks) >= max(2, total * 0.2)
    )

    if is_two_column:
        # [1] - y0 coordinate, vertical top of block
        # sorts blocks top to bottom within their columns
        # read the full left column top-to-bottom
        left_blocks.sort(key=lambda b: b[1])
        # read the full right column top-to-bottom
        right_blocks.sort(key=lambda b: b[1])
        # combine left and right blocks in reading order: left first, then right
        ordered = left_blocks + right_blocks
    else:
        # read everything top-to-bottom (single column)
        ordered = sorted(blocks, key=lambda b: b[1])

    # join all the block texts together in the final reading order separated by newlines
    return "\n".join(b[4].strip() for b in ordered)

"""
Function: Adaptive PDF text extraction for different layouts and formats 
- detects single vs. two-column layout and reads in correct order
- OCR fallback (pymupdf4llm) for pages yielding very little text (image-based PDF)
Returns: Extracted text in txt format, md if OCR fallback is triggered
- goes through each page of a PDF in case of a multi-page resume file
"""
def extract_text(pdf_path):
    # open the PDF with pymupdf, which can read text and layout info
    doc = pymupdf.open(pdf_path)
    # variable to hold all page texts in correct reading order
    page_texts = []
    # decision flag for OCR fallback necessity
    ocr_fallback = False

    # iterate through each page to extract text for multi-page PDFs
    for page in doc:
        page_text = _extract_page_columns(page)
        # activate OCR in case of sparse text
        # threshold: 50 characters per page
        if len(page_text.strip()) < 50:
            ocr_fallback = True
        # append the extracted text for this page to the overall list of the PDF file
        page_texts.append(page_text)

    doc.close()

    # OCR fallback activation: runs Tesseract OCR on image-based pages
    # returns markdown text
    if ocr_fallback:
        print(f"  {pdf_path}: sparse text detected, falling back to pymupdf4llm/OCR")
        return pymupdf4llm.to_markdown(pdf_path)
    
    # finalized raw text extraction
    return "\n".join(page_texts)