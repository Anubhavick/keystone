import pytest
from unittest.mock import MagicMock, patch
import shutil
from pathlib import Path
import sys
import os

# Ensure backend module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.vector_store import VectorStore

@pytest.fixture
def vector_store(tmp_path):
    # Use a temporary directory for the vector store
    persist_dir = tmp_path / "chroma_db_test"
    vs = VectorStore(collection_name="test_collection", persist_directory=str(persist_dir))
    yield vs
    # Teardown handled by tmp_path fixture usually, but we can explicit delete if needed
    vs.delete_collection()

def test_add_documents(vector_store):
    chunks = [
        {'text': 'Test chunk 1', 'chunk_id': '1', 'metadata': {'source': 'test'}},
        {'text': 'Test chunk 2', 'chunk_id': '2', 'metadata': {'source': 'test'}}
    ]
    success = vector_store.add_documents(chunks)
    assert success is True
    
    stats = vector_store.get_collection_stats()
    assert stats['total_chunks'] == 2

def test_search(vector_store):
    chunks = [
        {'text': 'The capital of France is Paris.', 'chunk_id': '1', 'metadata': {'category': 'geo'}},
        {'text': 'Python is a programming language.', 'chunk_id': '2', 'metadata': {'category': 'tech'}}
    ]
    vector_store.add_documents(chunks)
    
    # Search for Paris
    results = vector_store.search_with_scores("Paris", top_k=1)
    assert len(results) == 1
    assert "France" in results[0]['text']
    
    # Search for Python
    results = vector_store.search("Python", top_k=1)
    assert len(results['documents'][0]) == 1
    assert "programming" in results['documents'][0][0]

def test_delete_collection(vector_store):
    chunks = [{'text': 'To be deleted', 'chunk_id': '1', 'metadata': {}}]
    vector_store.add_documents(chunks)
    
    assert vector_store.get_collection_stats()['total_chunks'] == 1
    
    vector_store.delete_collection()
    assert vector_store.get_collection_stats()['total_chunks'] == 0

def test_hybrid_search_boost(vector_store):
    chunks = [
        {'text': 'Apple fruit is red', 'chunk_id': '1', 'metadata': {}},
        {'text': 'Apple computer is fast', 'chunk_id': '2', 'metadata': {}}
    ]
    vector_store.add_documents(chunks)
    
    # Query with specific keyword
    results = vector_store.hybrid_search("fruit", top_k=1)
    assert "red" in results[0]['text']
