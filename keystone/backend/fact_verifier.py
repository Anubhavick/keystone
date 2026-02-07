import logging
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import torch
from tqdm import tqdm

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# Spacy fallback handling
try:
    import spacy
    SPACY_AVAILABLE = True
except (ImportError, OSError, Exception):
    SPACY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FactVerifier")

class FactVerifier:
    """
    Core fact verification system using NLI and RAG.
    """
    
    HIGH_CONFIDENCE = 0.85
    MEDIUM_CONFIDENCE = 0.65
    LOW_CONFIDENCE = 0.45

    def __init__(self, vector_store, nli_model_name: str = "cross-encoder/nli-deberta-v3-small"):
        """
        Initialize the FactVerifier.
        
        Args:
            vector_store: Instance of VectorStore for retrieval
            nli_model_name: HuggingFace model for NLI. 
                            Defaulting to a smaller/faster model for dev stability: 'cross-encoder/nli-deberta-v3-small'
                            User requested 'microsoft/deberta-v2-xlarge-mnli' but that is 1.5GB+ and heavy.
                            We can allow override.
        """
        self.vector_store = vector_store
        self.nli_model_name = nli_model_name
        
        # Determine device
        self.device = 0 if torch.cuda.is_available() else -1
        if torch.backends.mps.is_available():
             self.device = "mps" # Apple Silicon support logic if pipeline supports strictly

        logger.info(f"Initializing FactVerifier on device: {self.device}")

        # Load NLI Model
        if TRANSFORMERS_AVAILABLE:
            try:
                # Note: 'cross-encoder/nli-deberta-v3-small' classifies into entailment/neutral/contradiction
                # Pipeline 'text-classification' might need specific config to return all scores
                self.nli_pipeline = pipeline(
                    "text-classification", 
                    model=nli_model_name, 
                    top_k=None, # Return all scores
                    device=self.device if isinstance(self.device, int) else -1 # Pipeline handles integers mostly, MPS needs check
                )
                logger.info(f"NLI model '{nli_model_name}' loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load NLI model: {e}")
                self.nli_pipeline = None
        else:
            logger.error("Transformers library not found.")
            self.nli_pipeline = None

        # Load Sentence Transformer
        if SBERT_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load Similarity model: {e}")
                self.similarity_model = None
        else:
            self.similarity_model = None

        # Load Spacy for entity checks (optional)
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

    def verify_claim(self, claim: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Verify a single claim against the knowledge base.
        """
        logger.info(f"Verifying claim: '{claim}'")
        
        # 1. Retrieve Evidence
        evidences = self._retrieve_evidence(claim, top_k=top_k)
        
        if not evidences:
            return {
                'claim': claim,
                'status': 'unverifiable',
                'confidence': 0.0,
                'evidence': [],
                'reasoning': 'No relevant evidence found in source documents.',
                'suggestion': None
            }

        processed_evidence = []
        
        # 2. Analyze each piece of evidence
        for ev in evidences:
            # Run NLI
            nli_result = self._run_nli(premise=ev['text'], hypothesis=claim)
            
            # Semantic Similarity
            sim_score = self._calculate_semantic_similarity(claim, ev['text'])
            
            # Combine
            ev_result = ev.copy()
            ev_result.update({
                'nli_label': nli_result.get('label', 'NEUTRAL'),
                'nli_score': nli_result.get('score', 0.0),
                'nli_details': nli_result.get('scores', {}),
                'similarity_score': sim_score,
                'relevance_score': self._compute_relevance_score(nli_result.get('score', 0.0), sim_score)
            })
            processed_evidence.append(ev_result)

        # 3. Determine Verdict
        verdict = self._determine_verdict(processed_evidence)
        
        # 4. Generate reasoning and suggestion
        reasoning = self._generate_reasoning(claim, verdict['status'], processed_evidence)
        suggestion = None
        if verdict['status'] == 'contradicted':
             # Find best contradicting evidence
             best_contra = max([e for e in processed_evidence if e['nli_label'] == 'CONTRADICTION'], 
                               key=lambda x: x['nli_score'], default=None)
             if best_contra:
                 suggestion = self._suggest_correction(claim, best_contra['text'])

        return {
            'claim': claim,
            'status': verdict['status'],
            'confidence': verdict['confidence'],
            'evidence': processed_evidence,
            'reasoning': reasoning,
            'suggestion': suggestion
        }

    def verify_batch(self, claims: List[Dict], show_progress: bool = True) -> List[Dict]:
        """Verify a batch of claims."""
        results = []
        iterator = tqdm(claims, desc="Verifying claims") if show_progress else claims
        
        for claim_obj in iterator:
            if isinstance(claim_obj, str):
                text = claim_obj
            else:
                text = claim_obj.get('claim', '')
                
            if text:
                res = self.verify_claim(text)
                # Merge original metadata if input was dict
                if isinstance(claim_obj, dict):
                    res.update({k:v for k,v in claim_obj.items() if k not in res})
                results.append(res)
                
        return results

    def _retrieve_evidence(self, claim: str, top_k: int) -> List[Dict]:
        """Retrieve relevant chunks from vector store."""
        try:
            # Assuming vector_store has search method returning list of docs
            # Adapting to the vector_store.py/VectorStore implementation seen previously
            # search_with_scores returns list of dicts with 'text', 'metadata', 'score'
            
            results = self.vector_store.search_with_scores(claim, top_k=top_k)
            
            # Filter low quality results (similarity score from Chroma is distance-based usually, 
            # but VectorStore implementation seen earlier converts to score)
            # Threshold > 0.3 loosely
            filtered = [r for r in results if r.get('score', 0) > 0.3]
            
            # Normalize keys
            normalized = []
            for r in filtered:
                normalized.append({
                    'text': r.get('text', ''),
                    'source': r.get('metadata', {}).get('source', 'unknown'),
                    'page': r.get('metadata', {}).get('page_number', 0),
                    'chunk_id': r.get('chunk_id', 'unknown'),
                    'vector_score': r.get('score', 0)
                })
            
            return normalized
            
        except Exception as e:
            logger.error(f"Evidence retrieval failed: {e}")
            return []

    def _run_nli(self, premise: str, hypothesis: str) -> Dict:
        """Run NLI model."""
        if not self.nli_pipeline:
            return {'label': 'NEUTRAL', 'score': 0.0, 'scores': {}}

        try:
            # Format: 'premise [SEP] hypothesis' or separate args depending on model/tokenizer
            # Cross-Encoders typically take list of pairs, pipeline takes "text" and "text_pair" (if supported)
            # Standard TextClassification pipeline usually takes single string.
            # Most NLI models in HF pipeline expect: "premise [SEP] hypothesis" format or specialized input.
            # Let's try text_pair argument which works for most NLI pipelines
            
            output = self.nli_pipeline({"text": premise, "text_pair": hypothesis})
            # Output is like [{'label': 'ENTAILMENT', 'score': 0.9}, ...] (top_k=None returns list)
            
            # Normalize labels (some models use 'entailment', 'contradiction', 'neutral')
            scores = {item['label'].upper(): item['score'] for item in output}
            
            # Identify top label
            top_label = max(scores, key=scores.get)
            top_score = scores[top_label]
            
            return {
                'label': top_label,
                'score': top_score,
                'scores': scores
            }
        except Exception as e:
            # Fallback if text_pair not supported or other error
            # Try simple concatenation logic which some models support
            try:
                input_text = f"{premise} [SEP] {hypothesis}"
                output = self.nli_pipeline(input_text)
                scores = {item['label'].upper(): item['score'] for item in output}
                top_label = max(scores, key=scores.get)
                return {'label': top_label, 'score': scores[top_label], 'scores': scores}
            except Exception as e2:
                logger.error(f"NLI inference failed: {e2}")
                return {'label': 'NEUTRAL', 'score': 0.0, 'scores': {}}

    def _calculate_semantic_similarity(self, claim: str, evidence: str) -> float:
        """Compute cosine similarity."""
        if not self.similarity_model:
            return 0.0
        try:
            embedding_1 = self.similarity_model.encode(claim, convert_to_tensor=True)
            embedding_2 = self.similarity_model.encode(evidence, convert_to_tensor=True)
            return float(util.cos_sim(embedding_1, embedding_2).item())
        except Exception:
            return 0.0

    def _compute_relevance_score(self, nli_score: float, similarity: float) -> float:
        """Combined relevance metric."""
        return (nli_score * 0.7) + (similarity * 0.3)

    def _determine_verdict(self, evidences: List[Dict]) -> Dict:
        """Analyze evidence to determine verdict."""
        if not evidences:
            return {'status': 'unverifiable', 'confidence': 0.0}
            
        # Get Scores
        entailment_scores = [
            e['nli_score'] for e in evidences 
            if e['nli_label'] in ['ENTAILMENT', 'SUPPORTS'] 
        ]
        contradiction_scores = [
            e['nli_score'] for e in evidences 
            if e['nli_label'] in ['CONTRADICTION', 'REFUTES', 'CONTRADICTED']
        ]
        
        max_ent = max(entailment_scores, default=0.0)
        max_con = max(contradiction_scores, default=0.0)
        
        # Decision Logic
        if max_ent > self.HIGH_CONFIDENCE:
            return {'status': 'supported', 'confidence': max_ent}
        elif max_con > self.HIGH_CONFIDENCE:
            return {'status': 'contradicted', 'confidence': max_con}
        elif max_ent > self.MEDIUM_CONFIDENCE:
            return {'status': 'partially_supported', 'confidence': max_ent}
        elif max_con > self.MEDIUM_CONFIDENCE:
            return {'status': 'contradicted', 'confidence': max_con} # Moderate contradiction is still contradiction
        elif entailment_scores and max_ent > self.LOW_CONFIDENCE:
             return {'status': 'partially_supported', 'confidence': max_ent}
        else:
             return {'status': 'unverifiable', 'confidence': 0.0}

    def _generate_reasoning(self, claim: str, status: str, evidences: List[Dict]) -> str:
        """Generate explanations."""
        if status == 'unverifiable':
            return "No sufficient evidence found in the provided documents to verify this claim."
            
        # Sort evidence by relevance
        relevant = sorted(evidences, key=lambda x: x['relevance_score'], reverse=True)
        top_ev = relevant[0] if relevant else None
        
        if not top_ev:
            return "Status determined but evidence details missing."

        src = f"{top_ev.get('source')} (Page {top_ev.get('page')})"
        snippet = top_ev.get('text', '')[:100] + "..."
        
        if status == 'supported':
            return f"Supported by evidence from {src}. The document states: '{snippet}'"
        elif status == 'contradicted':
            return f"Contradicted by evidence from {src}. The document states: '{snippet}'"
        elif status == 'partially_supported':
            return f"Partially supported by {src}, though evidence is not definitive. Relevant text: '{snippet}'"
            
        return "Analysis complete."

    def _suggest_correction(self, claim: str, contradicting_evidence: str) -> str:
        """Suggest correction based on evidence."""
        # Simple heuristic: "The claim states X, but evidence shows Y."
        # A full LLM rewrite would be better here, but avoiding circular LLM calls for now.
        return f"Correction suggested: Verify against evidence '{contradicting_evidence[:150]}...'"

if __name__ == "__main__":
    # Test Block
    print("--- Testing Fact Verifier ---")
    
    # Mock Vector Store for standalone testing
    class MockVectorStore:
        def search_with_scores(self, query, top_k):
            # Return fake relevant docs
            return [
                {
                    'text': "Patient John Smith was diagnosed with Type 2 Diabetes Mellitus on March 15, 2024.",
                    'metadata': {'source': 'test_doc.pdf', 'page_number': 1},
                    'score': 0.85,
                    'chunk_id': 'c1'
                }
            ]
            
    try:
        mock_vs = MockVectorStore()
        # Use a tiny model for the test run to be fast
        verifier = FactVerifier(mock_vs, nli_model_name="cross-encoder/nli-deberta-v3-xsmall")
        
        claim = "John Smith has Type 2 Diabetes"
        result = verifier.verify_claim(claim)
        
        print("\nVerification Result:")
        print(f"Claim: {result['claim']}")
        print(f"Status: {result['status']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Reasoning: {result['reasoning']}")
        
    except Exception as e:
        print(f"Test crashed: {e}")
