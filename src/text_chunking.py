"""Module for processing and chunking Samsung product manuals.

This module extracts text from PDF manuals, processes them into manageable chunks,
and enriches them with metadata about models, features, and software versions.
"""

import os
import re
from datetime import datetime
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTTextBox
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR)

MANUALS_DIR = "./samsung_manuals"
OUTPUT_FILE = "langchain_documents.jsonl"

MODEL_NAME_PATTERN = re.compile(
    r"""
    (Galaxy\s+(?:
        S\d{1,2}(?:\s+Ultra|\s+\+|\s+FE)?|
        Z\s+(?:Fold|Flip)\d{0,2}|
        A\d{1,2}(?:\s+\d{2})?|
        M\d{1,2}(?:\s+\d{2})?|
        Note\s+\d{1,2}(?:\s+Ultra|\s+\+)?
    ))|
    ((?:SM-[A-Z]\d{3,5})|
     (?:S\d{2,3}[A-Z](?:\s+Ultra|\s+\+)?))
    """,
    re.VERBOSE | re.IGNORECASE
)

VERSION_PATTERN = re.compile(r"(Android|One UI)\s*([\d\.]+)", re.IGNORECASE)
FEATURE_PATTERN = re.compile(
    r"""(
    Camera(?:\s+System)?|
    Battery(?:\s+Life)?|
    Security(?:\s+Features)?|
    Display(?:\s+Screen)?|
    Storage|
    (?:5G|4G|LTE)\s+Network|
    Wi-?Fi(?:\s+\d)?|
    Bluetooth(?:\s+\d\.?\d*)?|
    Fingerprint(?:\s+Sensor)?|
    Face\s+Recognition|
    Wireless\s+(?:Charging|DeX)|
    S-?Pen
    )""",
    re.VERBOSE | re.IGNORECASE
)
SECTION_HEADERS_PATTERN = re.compile(
    r"""
    ^(?:
        (?:Chapter|Section)\s+\d+[\.:]\s*(.+)|
        (\d+\.(?:\d+)*\s+.+)|
        ([A-Z][^a-z\n]{2,}(?:\s+[A-Z][^a-z\n]*)*)
    )$
    """,
    re.VERBOSE | re.MULTILINE
)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

def extract_model_names(text: str) -> list:
    """Extract all unique Samsung model names from text.
    
    Args:
        text (str): Text to extract model names from
        
    Returns:
        list: List of unique model names, or ["Unknown Model"] if none found
    """
    matches = MODEL_NAME_PATTERN.finditer(text)
    models = set()
    for match in matches:
        model = next(group for group in match.groups() if group is not None)
        models.add(model.strip())
    return list(models) if models else ["Unknown Model"]

def extract_software_versions(text: str) -> dict:
    """Extract Android and One UI version numbers from text.
    
    Args:
        text (str): Text to extract version numbers from
        
    Returns:
        dict: Dictionary with android_version and oneui_version keys
    """
    versions = {
        'android_version': None,
        'oneui_version': None
    }
    for match in VERSION_PATTERN.finditer(text):
        if 'android' in match.group(1).lower():
            versions['android_version'] = match.group(2)
        elif 'one ui' in match.group(1).lower():
            versions['oneui_version'] = match.group(2)
    return versions

def extract_features(text: str) -> list:
    """Extract Samsung-specific features mentioned in text.
    
    Args:
        text (str): Text to extract features from
        
    Returns:
        list: List of unique features found in the text
    """
    return list(set(match.group(1) for match in FEATURE_PATTERN.finditer(text)))

def extract_section_info(text: str) -> str:
    """Extract section title from text based on common header patterns.
    
    Args:
        text (str): Text to extract section title from
        
    Returns:
        str: Section title if found, "General Information" otherwise
    """
    matches = SECTION_HEADERS_PATTERN.finditer(text)
    sections = []
    for match in matches:
        section = next(group for group in match.groups() if group is not None)
        sections.append(section.strip())
    return sections[0] if sections else "General Information"

def extract_page_number(page_layout) -> int:
    """Extract page number from PDF page layout.
    
    Args:
        page_layout: PDFMiner page layout object
        
    Returns:
        int or None: Page number if found, None otherwise
    """
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            text = element.get_text().strip()
            if text.isdigit():
                return int(text)
    return None

def process_pdf(pdf_path: str) -> list:
    """Process a PDF manual and extract documents with metadata.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        list: List of Document objects with extracted content and metadata
    """
    try:
        text = extract_text(pdf_path)
        model_names = extract_model_names(text)
        versions = extract_software_versions(text)
        
        file_stats = os.stat(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        documents = []
        pages = list(extract_pages(pdf_path))
        
        for page_num, page_layout in enumerate(pages, 1):
            page_text = ""
            for element in page_layout:
                if isinstance(element, LTTextBox):
                    page_text += element.get_text()
            
            page_models = extract_model_names(page_text)
            chunks = text_splitter.create_documents([page_text])
            
            for chunk in chunks:
                chunk_features = extract_features(chunk.page_content)
                section = extract_section_info(chunk.page_content)
                chunk_models = extract_model_names(chunk.page_content)
                
                relevant_models = chunk_models if chunk_models != ["Unknown Model"] else \
                                page_models if page_models != ["Unknown Model"] else \
                                model_names
                
                metadata = {
                    'source_file': file_name,
                    'model_names': relevant_models,
                    'android_version': versions['android_version'],
                    'oneui_version': versions['oneui_version'],
                    'page_number': page_num,
                    'section': section,
                    'features': chunk_features,
                    'file_last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'chunk_length': len(chunk.page_content),
                    'manual_language': 'en'
                }
                
                documents.append(Document(
                    page_content=chunk.page_content,
                    metadata=metadata
                ))
        
        return documents
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []

def main():
    """Process all PDF manuals in the MANUALS_DIR directory."""
    all_documents = []
    pdf_files = [f for f in os.listdir(MANUALS_DIR) if f.endswith('.pdf')]
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        pdf_path = os.path.join(MANUALS_DIR, pdf_file)
        documents = process_pdf(pdf_path)
        all_documents.extend(documents)
    
    print(f"Total documents extracted: {len(all_documents)}")
    
    import json
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for doc in all_documents:
            f.write(json.dumps({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            }) + '\n')
    
    print(f"Documents saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()