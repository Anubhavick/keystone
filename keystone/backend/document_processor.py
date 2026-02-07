import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import pdfplumber
import docx

# REMOVED: from langchain_text_splitters import RecursiveCharacterTextSplitter
# Reason: Triggers 'spacy' import which crashes on Python 3.14 due to Pydantic V1 conflict.

class SimpleRecursiveTextSplitter:
    """
    Lightweight implementation of recursive character text splitting 
    to avoid heavy dependencies like LangChain/Spacy in this environment.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks recursively."""
        final_chunks = []
        if not text:
            return final_chunks
            
        # If text is small enough, return it
        if len(text) <= self.chunk_size:
            return [text]
            
        # Find best separator
        separator = ""
        for sep in self.separators:
            if sep in text:
                separator = sep
                break
        
        if not separator:
            # No separators found, hard split
            return [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            
        # Split
        splits = text.split(separator)
            
        # Merge
        valid_chunks = []
        current_chunk_parts = []
        current_len = 0
        
        for s in splits:
            s_len = len(s)
            sep_len = len(separator) if separator else 0
            
            # Check if adding the next part (s) plus the separator would exceed chunk_size
            # If current_chunk_parts is empty, we just add s.
            # If not empty, we add s + separator.
            potential_new_len = current_len + (s_len + sep_len if current_chunk_parts else s_len)

            if potential_new_len > self.chunk_size:
                # Flush current_chunk_parts if it's not empty
                if current_chunk_parts:
                    doc = separator.join(current_chunk_parts).strip()
                    if doc:
                        valid_chunks.append(doc)
                
                # Start new chunk with overlap if possible
                # For simplicity, this basic implementation focuses on splitting rather than complex overlap merging.
                # A more robust overlap would involve re-evaluating previous parts.
                # Here, we just ensure the new chunk starts with 's' and potentially some overlap from the end of the previous chunk.
                
                # Simple overlap: take the end of the last chunk and prepend to the new one
                overlap_text = ""
                if valid_chunks:
                    last_chunk = valid_chunks[-1]
                    overlap_text = last_chunk[max(0, len(last_chunk) - self.chunk_overlap):]
                
                current_chunk_parts = [overlap_text.strip(), s] if overlap_text.strip() else [s]
                current_len = len(separator.join(current_chunk_parts))
                
            else:
                if current_chunk_parts:
                    current_chunk_parts.append(s)
                    current_len += s_len + sep_len
                else:
                    current_chunk_parts.append(s)
                    current_len = s_len
                
        # Add any remaining parts as a final chunk
        if current_chunk_parts:
            doc = separator.join(current_chunk_parts).strip()
            if doc:
                 valid_chunks.append(doc)
                 
        return valid_chunks

    def create_documents(self, texts: List[str], metadatas: List[Dict] = None) -> List[Dict]:
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            chunks = self.split_text(text)
            for chunk in chunks:
                docs.append({'text': chunk, 'metadata': meta})
        return docs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Handles ingestion, text extraction, cleaning, and chunking of documents (PDF, DOCX, TXT).
    
    Example Usage:
        processor = DocumentProcessor()
        chunks = processor.process_file("medical_report.pdf")
        # Returns: [
        #   {
        #     'text': 'Patient presents with...',
        #     'page_number': 1,
        #     'chunk_id': 'doc_page_1_chunk_0',
        #     'source': 'medical_report.pdf',
        #     'metadata': {'title': 'Medical Report', 'date': '2024-03-15'}
        #   },
        #   ...
        # ]
    """

    def __init__(self):
        """Initialize the DocumentProcessor with a text splitter."""
        self.chunk_size = 500
        self.chunk_overlap = 50
        logger.info("DocumentProcessor initialized")

    def process_file(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Auto-detects file type (.pdf, .docx, .txt) and processes it into chunks.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            List of processed text chunks with metadata
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {path}")
            return []

        suffix = path.suffix.lower()
        logger.info(f"Processing file: {path.name} (Type: {suffix})")
        
        extracted_data = []

        try:
            if suffix == '.pdf':
                extracted_data = self.extract_text_from_pdf(path)
            elif suffix == '.docx':
                extracted_data = self.extract_text_from_docx(path)
            elif suffix == '.txt':
                extracted_data = self.extract_text_from_txt(path)
            else:
                logger.warning(f"Unsupported file type: {suffix}")
                return []
            
            if not extracted_data:
                logger.warning(f"No text extracted from {path}")
                return []

            return self.chunk_documents(extracted_data)

        except Exception as e:
            logger.error(f"Error processing file {path}: {str(e)}")
            return []

    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text from a PDF file using pdfplumber.
        
        Returns:
            List of dicts with {text, page_number, source_file, metadata}
        """
        results = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract basic metadata
                metadata = pdf.metadata if pdf.metadata else {}
                
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        cleaned_text = self.clean_text(text)
                        if cleaned_text:
                            data = {
                                'text': cleaned_text,
                                'page_number': i + 1,
                                'source_file': pdf_path.name,
                                'metadata': metadata
                            }
                            results.append(data)
                            
            logger.info(f"Extracted {len(results)} pages from PDF: {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error extracting PDF {pdf_path}: {e}")
            
        return results

    def extract_text_from_docx(self, docx_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text from a DOCX file using python-docx.
        
        Returns:
            List of dicts with {text, paragraph_number, source_file, metadata}
        """
        results = []
        try:
            doc = docx.Document(docx_path)
            
            # Extract core properties metadata
            core_props = doc.core_properties
            metadata = {
                'title': core_props.title,
                'author': core_props.author,
                'created': str(core_props.created) if core_props.created else None,
            }
            # Filter None values
            metadata = {k: v for k, v in metadata.items() if v}
            
            for i, para in enumerate(doc.paragraphs):
                text = para.text
                if text:
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        data = {
                            'text': cleaned_text,
                            'paragraph_number': i + 1,
                            'source_file': docx_path.name,
                            'metadata': metadata
                        }
                        results.append(data)
            
            logger.info(f"Extracted {len(results)} paragraphs from DOCX: {docx_path.name}")

        except Exception as e:
            logger.error(f"Error extracting DOCX {docx_path}: {e}")
            
        return results

    def extract_text_from_txt(self, txt_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text from a simple text file.
        
        Returns:
            List of dicts with {text, paragraph_number, source_file}
        """
        results = []
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Split by double newlines to detect paragraphs
            paragraphs = content.split('\n\n')
            
            for i, para in enumerate(paragraphs):
                cleaned = self.clean_text(para)
                if cleaned:
                    data = {
                        'text': cleaned,
                        'paragraph_number': i + 1,
                        'source_file': txt_path.name,
                        'metadata': {'source': 'text_file'}
                    }
                    results.append(data)
            
            logger.info(f"Extracted {len(results)} paragraphs from TXT: {txt_path.name}")

        except Exception as e:
            logger.error(f"Error extracting TXT {txt_path}: {e}")

        return results

    def chunk_documents(self, documents: List[Dict[str, Any]], chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Splits extracted documents into smaller text chunks.
        
        Args:
            documents: List of dicts containing text and metadata
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries ready for vector DB
        """
        # Update splitter params dynamically if needed
        self.text_splitter._chunk_size = chunk_size
        self.text_splitter._chunk_overlap = overlap
        
        chunked_docs = []
        
        for doc in documents:
            text = doc.get('text', '')
            if not text:
                continue
                
            chunks = self.text_splitter.split_text(text)
            
            # Determine numbering key (page_number for PDF, paragraph_number for others)
            # Default to page_number if present, else paragraph_number, else 0
            page_num = doc.get('page_number') or doc.get('paragraph_number', 0)
            source = doc.get('source_file', 'unknown')
            metadata = doc.get('metadata', {})
            
            for i, chunk in enumerate(chunks):
                # Format: doc_page_X_chunk_Y
                chunk_id = f"doc_page_{page_num}_chunk_{i}"
                
                chunk_data = {
                    'text': chunk,
                    'page_number': page_num,
                    'chunk_id': chunk_id,
                    'source': source,
                    'metadata': metadata
                }
                chunked_docs.append(chunk_data)
                
        logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} source items")
        return chunked_docs

    def clean_text(self, text: str) -> str:
        """
        Cleans text by removing excessive whitespace, headers/footers, and fixing encoding.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned string
        """
        if not text:
            return ""
        
        # 1. Remove null bytes and fix basic encoding artifacts
        text = text.replace('\x00', '')
        
        # 2. Remove separation lines or decorative characters often found in headers
        text = re.sub(r'^\s*[-_=]{3,}\s*$', '', text, flags=re.MULTILINE)
        
        # 3. Remove common page number patterns e.g., "Page 1 of 10"
        text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
        
        # 4. Normalize whitespace (replace multiple spaces/tabs/newlines with single space)
        # Note: We replace newlines with space to form continuous text block for chunking,
        # unless preservation of specific formatting is strictly required. 
        # Standard NLP practice involves normalizing whitespace.
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def detect_sections(self, text: str) -> List[Dict[str, str]]:
        """
        Identifies potential section headers in text.
        
        Args:
            text: Input text string
            
        Returns:
            List of detected sections with titles and content (heuristics based)
        """
        sections = []
        lines = text.split('\n')
        current_section = {"title": "Introduction", "content": []}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Heuristic: Upper case line, short length, starts with number or standard word
            is_header = (
                line.isupper() and len(line) < 50
            ) or (
                re.match(r'^\d+\.\s+[A-Z]', line)
            ) or (
                line.endswith(':') and len(line) < 50
            )
            
            if is_header:
                # Save previous section if it has content
                if current_section["content"]:
                    current_section["content"] = " ".join(current_section["content"])
                    sections.append(current_section)
                
                # Start new section
                current_section = {"title": line, "content": []}
            else:
                current_section["content"].append(line)
        
        # Append final section
        if current_section["content"]:
            current_section["content"] = " ".join(current_section["content"])
            sections.append(current_section)
            
        return sections
