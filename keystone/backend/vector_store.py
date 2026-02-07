import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional

class VectorStore:
    def __init__(self, collection_name: str = "keystone", persist_directory: str = None):
        """
        Initialize ephemeral/in-memory VectorStore.
        Args:
            collection_name: Name of the collection (default 'keystone')
            persist_directory: Ignored in this simplified version (using in-memory Client)
        """
        self.client = chromadb.Client()
        # Use get_or_create to be safe against re-init in same session if logic changes
        try:
            self.collection = self.client.get_or_create_collection(name=collection_name)
        except Exception:
             # Fallback if get_or_create fails (some old versions only have create_collection)
             self.collection = self.client.create_collection(name=collection_name)
             
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add documents to the collection.
        Expects chunks to be a list of dicts with 'text', 'chunk_id', and optionally 'metadata'.
        """
        if not chunks:
            return
            
        texts = [c['text'] for c in chunks]
        ids = [c['chunk_id'] for c in chunks]
        metadatas = [c.get('metadata', {}) for c in chunks]
        
        # Ensure metadata is flat/scalar for Chroma
        clean_metadatas = []
        for m in metadatas:
            clean_m = {}
            for k, v in m.items():
                if isinstance(v, (str, int, float, bool)):
                    clean_m[k] = v
                else:
                    clean_m[k] = str(v)
            clean_metadatas.append(clean_m)

        self.collection.add(
            documents=texts,
            ids=ids,
            metadatas=clean_metadatas
        )

    def search(self, query: str, top_k: int = 3):
        """
        Search for documents. Returns raw Chroma result.
        """
        return self.collection.query(query_texts=[query], n_results=top_k)

    def search_with_scores(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search and return structured results with scores (for FactVerifier compatibility).
        """
        results = self.search(query, top_k)
        
        if not results['ids'] or not results['ids'][0]:
            return []
            
        clean_results = []
        # Safely handle potentially missing 'distances' if older client version
        distances = results.get('distances', [[0]*len(results['ids'][0])])
        
        for i in range(len(results['ids'][0])):
            doc_text = results['documents'][0][i]
            meta = results['metadatas'][0][i] if results['metadatas'] else {}
            chunk_id = results['ids'][0][i]
            dist = distances[0][i]
            
            # Simple similarity conversion (1 - distance) assuming cosine distance
            score = max(0.0, 1.0 - dist)
            
            clean_results.append({
                'text': doc_text,
                'metadata': meta,
                'chunk_id': chunk_id,
                'score': score,
                'source': meta.get('source', 'unknown'),
                'page': meta.get('page_number', 0)
            })
            
        return clean_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return collection statistics (for app.py compatibility).
        """
        return {
            'total_chunks': self.collection.count(),
            'name': self.collection.name
        }

    def delete_collection(self):
        """Reset/Clear collection."""
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
