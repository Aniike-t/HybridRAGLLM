# data_processing/pdf_extractor.py
import os
import re
import fitz  
from typing import List, Tuple, Dict

import config

def extract_text_from_pdf(pdf_path: str) -> Tuple[str, Dict]:
    """
    Extracts text and metadata (page numbers to character indices) from a PDF.
    Handles multi-column layouts better.
    """
    if config.PDF_EXTRACTOR != "PyMuPDF":
        raise NotImplementedError(f"PDF Extractor '{config.PDF_EXTRACTOR}' not implemented.")

    text = ""
    metadata = {}
    char_index = 0

    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                blocks = page.get_text("blocks")
                page_text = ""
                for block in blocks:
                    # block is (x0, y0, x1, y1, text, block_no, block_type)
                    block_text = block[4]  
                    page_text += block_text + "\n" # Add newline between blocks

                metadata[page_num + 1] = char_index
                text += page_text
                char_index += len(page_text)

    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return "", {}  

    return text, metadata

def clean_text(text: str) -> str:
    """Cleans and normalizes the extracted text."""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()      
    text = text.lower()                      
    return text

def process_pdf(pdf_path: str) -> Tuple[str, Dict]:
    raw_text, metadata = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(raw_text)
    return cleaned_text, metadata