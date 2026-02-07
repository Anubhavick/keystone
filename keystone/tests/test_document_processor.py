import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import os

# Ensure backend module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.document_processor import DocumentProcessor

@pytest.fixture
def processor():
    return DocumentProcessor()

@pytest.fixture
def sample_txt(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("Paragraph 1.\n\nParagraph 2 is here.", encoding="utf-8")
    return p

def test_clean_text(processor):
    raw = "  Hello   World! \n Page 1 of 5 "
    cleaned = processor.clean_text(raw)
    assert cleaned == "Hello World!"

def test_chunk_documents(processor):
    docs = [
        {'text': 'A' * 600, 'metadata': {'source': 'test'}},
        {'text': 'Short text', 'metadata': {'source': 'test'}}
    ]
    chunks = processor.chunk_documents(docs, chunk_size=500, overlap=50)
    
    assert len(chunks) >= 2
    assert chunks[0]['chunk_id'].startswith("doc_page_0_chunk_")
    assert chunks[1]['text']

@patch('backend.document_processor.pdfplumber.open')
def test_extract_text_from_pdf(mock_pdf_open, processor):
    # Setup Mock
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Content Page 1"
    mock_pdf.pages = [mock_page]
    mock_pdf.metadata = {'Author': 'Test'}
    
    context_manager = MagicMock()
    context_manager.__enter__.return_value = mock_pdf
    mock_pdf_open.return_value = context_manager
    
    # Run
    path = Path("fake.pdf")
    results = processor.extract_text_from_pdf(path)
    
    # Verify
    assert len(results) == 1
    assert results[0]['text'] == "PDF Content Page 1"
    assert results[0]['metadata']['Author'] == 'Test'

def test_process_txt_file(processor, sample_txt):
    results = processor.process_file(sample_txt)
    assert len(results) > 0
    assert "Paragraph 1" in results[0]['text'] or "Paragraph 2" in results[0]['text']

def test_process_file_not_found(processor):
    results = processor.process_file("nonexistent.pdf")
    assert results == []
