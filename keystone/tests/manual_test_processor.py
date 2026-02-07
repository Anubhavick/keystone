import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.document_processor import DocumentProcessor

# Setup dummy files
def setup_dummy_files():
    data_dir = Path("data/test_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    txt_path = data_dir / "sample.txt"
    with open(txt_path, "w") as f:
        f.write("Section 1: Introduction\n\nThis is a test paragraph for the document processor.\nIt supports multiple lines.\n\nSection 2: details\n\nAnother paragraph here.")
    
    return txt_path

def test_processor():
    logging.basicConfig(level=logging.INFO)
    processor = DocumentProcessor()
    
    txt_path = setup_dummy_files()
    print(f"Testing with file: {txt_path}")
    
    chunks = processor.process_file(txt_path)
    
    print(f"\nProcessed {len(chunks)} chunks.")
    for chunk in chunks:
        print("-" * 40)
        print(f"ID: {chunk['chunk_id']}")
        print(f"Source: {chunk['source']}")
        print(f"Text: {chunk['text'][:50]}...")
        print(f"Metadata: {chunk['metadata']}")

if __name__ == "__main__":
    test_processor()
