import logging
try:
    import chromadb
    from chromadb.utils import embedding_functions
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except (ImportError, RuntimeError, Exception) as e:
    # Handle Python 3.14 pydantic incompatibility or missing module
    logging.warning(f"ChromaDB import failed (likely Python 3.14 compatibility issue): {e}")
    CHROMADB_AVAILABLE = False
    chromadb = None
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Wrapper for ChromaDB to store and retrieve document chunks using semantic search.
    
    Example Usage:
        vector_store = VectorStore(collection_name="medical_docs")
        
        # Add documents
        chunks = [
            {'text': 'Patient has diabetes...', 'chunk_id': 'doc1_chunk0', 'page': 1},
            {'text': 'Treatment plan...', 'chunk_id': 'doc1_chunk1', 'page': 2}
        ]
        vector_store.add_documents(chunks)
        
        # Search
        results = vector_store.search("diabetes treatment", top_k=3)
        
        # Clean search results
        clean_results = vector_store.search_with_scores("diabetes", top_k=3)
    """

    def __init__(self, collection_name: str = "source_documents", persist_directory: str = "./data/chroma_db"):
        """
        Initialize ChromaDB client and embedding model.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to save the database
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model_name = "all-MiniLM-L6-v2"
        
        try:
            # Initialize embedding function
            # We use SentenceTransformer directly for more control, but Chroma has a wrapper too.
            # Here keeping it simple with the default or a custom wrapper if needed.
            # For simplicity and consistnecy with requirements, let's load the model 
            # and use it to create embeddings or use Chroma's built-in if compatible.
            # Requirement says "Initialize sentence-transformer model".
            
            logger.info(f"Initializing VectorStore with model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Custom embedding function for Chroma
            class SentenceTransformerEmbeddingFunction(chromadb.EmbeddingFunction):
                def __init__(self, model):
                    self.model = model
                def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                    return self.model.encode(input).tolist()

            self.embedding_fn = SentenceTransformerEmbeddingFunction(self.embedding_model)

            # Initialize Chroma Client
            if CHROMADB_AVAILABLE:
                self.client = chromadb.PersistentClient(path=persist_directory)
                
                # Get or create collection
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"} # Use cosine similarity
                )
                logger.info(f"VectorStore initialized with collection '{collection_name}' at '{persist_directory}'")
            else:
                 logger.warning("ChromaDB not available. VectorStore running in MOCK mode.")
                 self.client = None
                 self.collection = None
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            # If we are in mock mode, suppress error? No, let's allow partial init
            if not CHROMADB_AVAILABLE:
                pass
            else:
                raise

    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to vector database.
        
        Args:
            chunks: List of dictionaries containing text and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            logger.warning("No chunks provided to add_documents")
            return False
            
        try:
            # Prepare data
            documents = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Ensure we have essential fields
                text = chunk.get('text', '')
                if not text:
                    continue
                    
                chunk_id = chunk.get('chunk_id')
                # If chunk_id is missing, generate one or skip. Skipping for safety.
                if not chunk_id:
                    logger.warning(f"Skipping chunk without ID: {chunk}")
                    continue
                
                # Flatten metadata (Chroma requires flat dicts of str/int/float/bool)
                # We need to make sure 'page' or 'page_number' is consistent
                base_metadata = chunk.get('metadata', {}).copy()
                
                # Add top-level fields to metadata for filtering
                if 'page_number' in chunk:
                    base_metadata['page_number'] = chunk['page_number']
                if 'source' in chunk:
                    base_metadata['source'] = chunk['source']
                
                # Convert non-primitive metadata values to string ensures Chroma compatibility
                safe_metadata = {}
                for k, v in base_metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        safe_metadata[k] = v
                    else:
                        safe_metadata[k] = str(v)
                
                documents.append(text)
                metadatas.append(safe_metadata)
                ids.append(chunk_id)
            
            if not documents:
                return False
                
            if self.collection:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Successfully added {len(documents)} documents to '{self.collection_name}'")
                return True
            else:
                logger.warning("Mock VectorStore: Documents not added (DB unavailable).")
                return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False

    def batch_add_documents(self, chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Add documents in batches for large datasets.
        
        Args:
            chunks: List of document chunks
            batch_size: Number of chunks per batch
        """
        total_chunks = len(chunks)
        logger.info(f"Starting batch ingestion of {total_chunks} chunks (Batch size: {batch_size})")
        
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            success = self.add_documents(batch)
            if success:
                if total_chunks > 1000:
                    logger.info(f"Processed batch {i // batch_size + 1}/{(total_chunks + batch_size - 1) // batch_size}")
            else:
                logger.error(f"Failed to process batch starting at index {i}")

    def search(self, query: str, top_k: int = 5, filter_metadata: Dict = None) -> Dict[str, Any]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            filter_metadata: Dictionary for filtering (e.g. {'source': 'file.pdf'})
            
        Returns:
            ChromaDB search results dict
        """
        try:
            if not self.collection:
                return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {'documents': [], 'metadatas': [], 'distances': [], 'ids': []}

    def search_with_scores(self, query: str, top_k: int = 5, filter_metadata: Dict = None) -> List[Dict[str, Any]]:
        """
        Search and return cleaner results with similarity scores.
        
        Args:
            query: Search query
            top_k: Number of results
            filter_metadata: Optional filters
            
        Returns:
            List of dicts with text, metadata, score, id
        """
        try:
            if not self.collection:
                 logger.warning("Mock VectorStore returning empty results.")
                 return []

            # Fetch explicitly to ensure we get what we need
            raw_results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata,
                include=['documents', 'metadatas', 'distances']
            )
            
            clean_results = []
            
            # Check if we got any results
            if not raw_results['ids'] or not raw_results['ids'][0]:
                return []
                
            # Iterate through the first query's results (since we only search one query)
            count = len(raw_results['ids'][0])
            
            for i in range(count):
                distance = raw_results['distances'][0][i]
                score = self._calculate_similarity_score(distance)
                
                result = {
                    'text': raw_results['documents'][0][i],
                    'metadata': raw_results['metadatas'][0][i],
                    'chunk_id': raw_results['ids'][0][i],
                    'score': score
                }
                clean_results.append(result)
                
            return clean_results
            
        except Exception as e:
            logger.error(f"Search with scores failed: {e}")
            return []

    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Combine semantic search with keyword matching.
        Boosts results that contain exact query terms.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of re-ranked results
        """
        # Get more results initially to filter/rerank
        candidates = self.search_with_scores(query, top_k=top_k * 2)
        
        # Simple keyword boosting
        query_terms = set(query.lower().split())
        
        for doc in candidates:
            text = doc['text'].lower()
            
            # Calculate term overlap score
            term_matches = sum(1 for term in query_terms if term in text)
            keyword_score = term_matches / len(query_terms) if query_terms else 0
            
            # Boost score: Weighted average of semantic (0.7) and keyword (0.3)
            # This is a basic heuristic; could be tuned
            doc['original_score'] = doc['score']
            doc['score'] = (doc['score'] * 0.7) + (keyword_score * 0.3)
            
        # Re-sort stats
        candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return candidates[:top_k]

    def delete_collection(self) -> bool:
        """
        Remove all documents from collection.
        """
        try:
            if not self.client:
                return True

            self.client.delete_collection(self.collection_name)
            # Re-create mostly empty collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' deleted and re-created.")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return stats about the collection.
        """
        try:
            if not self.collection:
                return {'error': 'Mock Store', 'total_chunks': 0}

            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name,
                'embedding_model': self.embedding_model_name
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}

    def _calculate_similarity_score(self, distance: float) -> float:
        """
        Convert distance to similarity score (0-1).
        Chroma default cosine distance is 0 (identical) to 2 (opposite).
        Actually chroma hnsw:space='cosine' returns cosine distance: 1 - cosine_similarity.
        Range is 0 to 2.
        We want a score where 1 is best, 0 is worst.
        """
        # Cosine Distance: 0 = exact match
        # Let's just do max(0, 1 - distance) for simple normalized behavior 
        # assuming vectors are normalized and distance is usually < 1 for similar items
        return max(0.0, 1.0 - distance)

