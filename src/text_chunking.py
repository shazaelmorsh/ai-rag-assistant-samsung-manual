import os
import re
from datetime import datetime
from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer, LTTextBox
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

import logging
logging.getLogger("pdfminer").setLevel(logging.ERROR) # Too many warning

# Config
MANUALS_DIR = "./samsung_manuals"
OUTPUT_FILE = "langchain_documents.jsonl"

# Enhanced regex patterns
MODEL_NAME_PATTERN = re.compile(
    r"""
    # Match Galaxy series with various suffixes
    (Galaxy\s+(?:
        S\d{1,2}(?:\s+Ultra|\s+\+|\s+FE)?|  # S series with Ultra, Plus, FE variants
        Z\s+(?:Fold|Flip)\d{0,2}|            # Foldable series
        A\d{1,2}(?:\s+\d{2})?|               # A series with optional year
        M\d{1,2}(?:\s+\d{2})?|               # M series with optional year
        Note\s+\d{1,2}(?:\s+Ultra|\s+\+)?    # Note series with variants
    ))|
    # Match standalone model numbers
    ((?:SM-[A-Z]\d{3,5})|                    # Model numbers like SM-A515F
     (?:S\d{2,3}[A-Z](?:\s+Ultra|\s+\+)?))   # Model numbers like S23 Ultra
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
        (?:Chapter|Section)\s+\d+[\.:]\s*(.+)|  # Traditional chapter/section headers
        (\d+\.(?:\d+)*\s+.+)|                   # Numbered sections like 1.2.3
        ([A-Z][^a-z\n]{2,}(?:\s+[A-Z][^a-z\n]*)*) # ALL CAPS headers
    )$
    """,
    re.VERBOSE | re.MULTILINE
)

# LangChain text splitter (adjust chunk_size & overlap as needed)
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200
)

def extract_model_names(text):
    """Extract all unique model names from the text."""
    matches = MODEL_NAME_PATTERN.finditer(text)
    models = set()
    for match in matches:
        # Get the first non-None group from the match
        model = next(group for group in match.groups() if group is not None)
        models.add(model.strip())
    return list(models) if models else ["Unknown Model"]

def extract_software_versions(text):
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

def extract_features(text):
    return list(set(match.group(1) for match in FEATURE_PATTERN.finditer(text)))

def extract_section_info(text):
    matches = SECTION_HEADERS_PATTERN.finditer(text)
    sections = []
    for match in matches:
        # Get the first non-None group from the match
        section = next(group for group in match.groups() if group is not None)
        sections.append(section.strip())
    return sections[0] if sections else "General Information"

def extract_page_number(page_layout):
    # Try to find page numbers in the footer area
    for element in page_layout:
        if isinstance(element, LTTextContainer):
            text = element.get_text().strip()
            if text.isdigit():
                return int(text)
    return None

def process_pdf(pdf_path):
    try:
        # Extract full text for content
        text = extract_text(pdf_path)
        
        # Extract basic metadata
        model_names = extract_model_names(text)
        versions = extract_software_versions(text)
        
        # Get file metadata
        file_stats = os.stat(pdf_path)
        file_name = os.path.basename(pdf_path)
        
        # Process pages for detailed metadata
        documents = []
        pages = list(extract_pages(pdf_path))
        
        # Process each page
        for page_num, page_layout in enumerate(pages, 1):
            page_text = ""
            for element in page_layout:
                if isinstance(element, LTTextBox):
                    page_text += element.get_text()
            
            # Extract models mentioned on this specific page
            page_models = extract_model_names(page_text)
            
            # Create chunks from page text
            chunks = text_splitter.create_documents([page_text])
            
            for chunk in chunks:
                # Extract features and section info for this chunk
                chunk_features = extract_features(chunk.page_content)
                section = extract_section_info(chunk.page_content)
                
                # Try to find models mentioned in this specific chunk
                chunk_models = extract_model_names(chunk.page_content)
                
                # Use the most specific model information available
                # Priority: chunk models > page models > document models
                relevant_models = chunk_models if chunk_models != ["Unknown Model"] else \
                                page_models if page_models != ["Unknown Model"] else \
                                model_names
                
                metadata = {
                    'source_file': file_name,
                    'model_names': relevant_models,  # Now it's a list of models
                    'android_version': versions['android_version'],
                    'oneui_version': versions['oneui_version'],
                    'page_number': page_num,
                    'section': section,
                    'features': chunk_features,
                    'file_last_modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                    'chunk_length': len(chunk.page_content),
                    'manual_language': 'en'  # You could enhance this with language detection
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
    all_documents = []
    for filename in tqdm(os.listdir(MANUALS_DIR)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(MANUALS_DIR, filename)
            documents = process_pdf(pdf_path)
            all_documents.extend(documents)
    
    # Save to a JSONL file
    import json
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for doc in all_documents:
            f.write(json.dumps({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(all_documents)} LangChain Documents to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
