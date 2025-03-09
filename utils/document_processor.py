"""
Document processing utilities for handling PDFs and other document formats.
Uses functional programming patterns where appropriate.
"""

import PyPDF2
import pdfplumber
import docx
from typing import Dict, List, Union, Callable, Optional, BinaryIO
import io
from functools import partial

def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """
    Extract text from a PDF file using PyPDF2
    
    Args:
        file_obj: A file-like object containing the PDF
        
    Returns:
        str: Extracted text from the PDF
    """
    reader = PyPDF2.PdfReader(file_obj)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text

def extract_text_from_pdf_with_layout(file_obj: BinaryIO) -> str:
    """
    Extract text from a PDF with better layout preservation using pdfplumber
    
    Args:
        file_obj: A file-like object containing the PDF
        
    Returns:
        str: Extracted text from the PDF with layout preserved
    """
    with pdfplumber.open(file_obj) as pdf:
        text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    return text

def extract_text_from_docx(file_obj: BinaryIO) -> str:
    """
    Extract text from a DOCX file
    
    Args:
        file_obj: A file-like object containing the DOCX
        
    Returns:
        str: Extracted text from the DOCX
    """
    doc = docx.Document(file_obj)
    text = "\n\n".join(paragraph.text for paragraph in doc.paragraphs)
    return text

def extract_text_from_txt(file_obj: BinaryIO) -> str:
    """
    Extract text from a plain text file
    
    Args:
        file_obj: A file-like object containing the text file
        
    Returns:
        str: Extracted text from the file
    """
    return file_obj.read().decode('utf-8')

def get_extractor_for_file_type(file_type: str) -> Optional[Callable]:
    """
    Get the appropriate text extractor function based on file type
    
    Args:
        file_type: The file extension or MIME type
        
    Returns:
        Callable: A function that extracts text from the given file type
    """
    extractors = {
        'pdf': extract_text_from_pdf_with_layout,
        'docx': extract_text_from_docx,
        'doc': extract_text_from_docx,  # Note: may not work well with .doc format
        'txt': extract_text_from_txt,
    }
    
    file_type = file_type.lower().strip('.')
    return extractors.get(file_type)

def extract_text_from_file(file_obj: BinaryIO, file_type: str) -> str:
    """
    Extract text from a file based on its type
    
    Args:
        file_obj: A file-like object
        file_type: The file extension or MIME type
        
    Returns:
        str: Extracted text from the file
        
    Raises:
        ValueError: If the file type is not supported
    """
    extractor = get_extractor_for_file_type(file_type)
    
    if extractor is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return extractor(file_obj)

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split a large text into overlapping chunks for processing
    
    Args:
        text: The text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List[str]: List of text chunks
    """
    if not text:
        return []
    
    # Split text into chunks with overlap
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a good breaking point (newline or space)
        if end < len(text):
            # Look for newline first
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start + chunk_size // 2:
                end = newline_pos + 1
            else:
                # Fall back to space
                space_pos = text.rfind(' ', start, end)
                if space_pos > start + chunk_size // 2:
                    end = space_pos + 1
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def extract_structured_text(text: str) -> Dict:
    """
    Extract structured information from legal text
    This is a placeholder function - would need to be expanded based on actual requirements
    
    Args:
        text: Legal document text
        
    Returns:
        Dict: Structured information extracted from the text
    """
    # This is a simplified example - in a real implementation, this would
    # use more sophisticated techniques like regex patterns or NER
    structure = {
        "title": "",
        "parties": [],
        "dates": [],
        "clauses": [],
        "key_terms": []
    }
    
    lines = text.split('\n')
    if lines and lines[0].strip():
        structure["title"] = lines[0].strip()
    
    return structure

# Partial function applications for common use cases
extract_pdf_text = extract_text_from_pdf
extract_pdf_text_with_layout = extract_text_from_pdf_with_layout
chunk_text = partial(split_text_into_chunks, chunk_size=1000, overlap=200)