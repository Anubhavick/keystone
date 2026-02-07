import os
import json
import re
import hashlib
import time
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

# Third-party imports
try:
    import spacy
    SPACY_AVAILABLE = True
except (ImportError, Exception):
    # Handle Python 3.14 incompatibility where spacy crashes on import due to pydantic
    SPACY_AVAILABLE = False
    spacy = None

from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# LLM Clients (using direct imports as requested, or langchain wrappers if preferred for consistency. 
# User asked for 'import anthropic', so let's try to accommodate direct usage or robust handling)
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ClaimExtractor")

load_dotenv()

class ClaimExtractor:
    """
    AI-powered claim extraction system that decomposes text into atomic factual claims.
    Supports LLM-based extraction with rule-based fallback and semantic deduplication.
    """

    def __init__(self, api_key: str = None, model_provider: str = "openai", model_name: str = "gpt-4-turbo"):
        """
        Initialize the ClaimExtractor.
        
        Args:
            api_key: API key for the LLM provider (optional, can be read from env)
            model_provider: 'openai' or 'anthropic'
            model_name: Specific model identifier
        """
        self.model_provider = model_provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{model_provider.upper()}_API_KEY")
        
        # Initialize Spacy with fallback
        try:
            if SPACY_AVAILABLE:
                self.nlp = spacy.load("en_core_web_sm")
                self.use_spacy = True
            else:
                self.nlp = None
                self.use_spacy = False
                logger.warning("Spacy not available (import failed). Using regex fallback.")
        except (OSError, ImportError, Exception) as e:
            logger.warning(f"Spacy failed to load: {e}. Using regex fallback.")
            self.nlp = None
            self.use_spacy = False

        # Initialize Sentence Transformer for deduplication
        # Using a lightweight model for speed
        logger.info("Loading sentence-transformer model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Cache
        self.claim_cache: Dict[str, List[Dict]] = {}
        
        logger.info(f"ClaimExtractor initialized (Provider: {model_provider}, Model: {model_name})")

    def _initialize_llm(self):
        """Initializes LangChain LLM wrapper."""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available. LLM extraction will be disabled.")
            return None

        try:
            # Auto-detect GitHub Models credential
            if self.api_key and self.api_key.startswith("github_pat_"):
                logger.info("Detected GitHub Personal Access Token. Configuring for GitHub Models.")
                return ChatOpenAI(
                    model="gpt-4o", # Default to GPT-4o for GitHub Models
                    temperature=0,
                    openai_api_key=self.api_key,
                    openai_api_base="https://models.inference.ai.azure.com",
                    max_retries=2
                )

            if self.model_provider == "openai":
                return ChatOpenAI(model=self.model_name, temperature=0, openai_api_key=self.api_key)
            elif self.model_provider == "anthropic":
                return ChatAnthropic(model=self.model_name, temperature=0, anthropic_api_key=self.api_key)
            else:
                logger.warning(f"Unknown provider {self.model_provider}, defaulting to generic OpenAI compatible or failing.")
                return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return None

    def extract_atomic_claims(self, text: str, use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        Main method to extract claims from text.
        
        Args:
            text: Input text
            use_llm: Whether to use LLM (True) or fallback to rule-based (False)
            
        Returns:
            List of claim dictionaries
        """
        if not text or not text.strip():
            return []

        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.claim_cache:
            logger.info("Returning cached claims.")
            return self.claim_cache[text_hash]

        claims = []
        try:
            if use_llm and self.llm:
                claims = self.extract_with_llm(text)
            
            # If LLM failed or disabled, use fallback
            if not claims:
                if use_llm:
                    logger.warning("LLM extraction returned no results. Falling back to rule-based.")
                claims = self.extract_rule_based(text)

            # Post-processing: Merge similar claims
            claims = self.merge_similar_claims(claims)
            
            # Cache results
            self.claim_cache[text_hash] = claims
            return claims

        except Exception as e:
            logger.error(f"Error in extract_atomic_claims: {e}")
            return []

    def extract_with_llm(self, text: str) -> List[Dict[str, Any]]:
        """
        Use LLM API to extract claims.
        """
        if not self.llm:
            return []

        prompt_template = """You are a fact-checking assistant. Break down the following text into atomic factual claims.
  
  Rules:
  1. Each claim must be a single, verifiable statement.
  2. Split compound sentences into multiple claims.
  3. Preserve exact entities (names, numbers, dates).
  4. Label claim type: 'factual', 'numerical', 'temporal', or 'entity'.
  5. Extract key entities mentioned in each claim.
  6. Return a JSON array of objects.
  
  Text: {text}
  
  Format Instructions:
  Return ONLY valid JSON.
  [
    {{
      "claim": "exact claim text",
      "claim_type": "factual|numerical|temporal|entity",
      "entities": ["entity1", "entity2"]
    }}
  ]
"""
        try:
            chain = ChatPromptTemplate.from_template(prompt_template) | self.llm | JsonOutputParser()
            response = chain.invoke({"text": text})
            
            # Normalize and validate
            valid_claims = []
            for i, item in enumerate(response):
                if self._validate_claim(item):
                    item['claim_id'] = f"claim_llm_{i}_{int(time.time())}"
                    item['confidence'] = 0.95 # Placeholder confidence for LLM
                    item['source'] = 'llm'
                    item['claim'] = self._normalize_claim(item['claim'])
                    valid_claims.append(item)
            
            return valid_claims

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    def extract_rule_based(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback method using spaCy or simple regex for sentence segmentation.
        """
        claims = []
        sentences = []
        
        if self.use_spacy and self.nlp:
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
        else:
            # Regex fallback for sentence splitting
            sentences = re.split(r'(?<=[.!?]) +', text)

        claim_idx = 0
        for sent_text in sentences:
            sent_text = sent_text.strip()
            if len(sent_text) < 10: # Skip very short snippets
                continue
                
            claim_type = self.classify_claim_type(sent_text)
            entities = self.extract_entities(sent_text)
            
            contact = {
                'claim': sent_text,
                'claim_id': f"claim_rb_{claim_idx}_{int(time.time())}",
                'sentence_id': 0, # Placeholder
                'claim_type': claim_type,
                'entities': entities,
                'confidence': 0.60, # Lower confidence for rule-based
                'source': 'rule_based'
            }
            claims.append(contact)
            claim_idx += 1
            
        return claims

    def merge_similar_claims(self, claims: List[Dict], threshold: float = 0.85) -> List[Dict]:
        """
        Remove duplicate/similar claims using embeddings.
        """
        if len(claims) < 2:
            return claims
            
        final_claims = []
        claim_texts = [c['claim'] for c in claims]
        
        # Compute embeddings
        embeddings = self.embedding_model.encode(claim_texts, convert_to_tensor=True)
        
        # Compute cosine similarity matrix
        cosine_scores = util.cos_sim(embeddings, embeddings)
        
        # Greedy clustering / deduplication
        kept_indices = set()
        sorted_indices = list(range(len(claims))) # Could sort by length or confidence
        
        # Mark duplicates
        ignored_indices = set()
        for i in range(len(claims)):
            if i in ignored_indices:
                continue
            
            # Keep i
            kept_indices.add(i)
            
            # Check for similars
            for j in range(i + 1, len(claims)):
                if j in ignored_indices:
                    continue
                    
                score = cosine_scores[i][j]
                if score >= threshold:
                    # j is a duplicate of i
                    # Logic: keep the one with higher confidence or longer text?
                    # For now simplicity: keep i, drop j
                    ignored_indices.add(j)
        
        for i in sorted(list(kept_indices)):
            final_claims.append(claims[i])
            
        return final_claims

    def classify_claim_type(self, claim: str) -> str:
        """Classify claim type based on content."""
        # Temporal
        if re.search(r'\b(19|20)\d{2}\b|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', claim.lower()):
            return 'temporal'
        # Numerical
        if re.search(r'\d+(\.\d+)?%?|\b(one|two|three|ten|hundred|thousand)\b', claim.lower()):
            return 'numerical'
        # Entity (heuristic: multiple capital words together)
        if len(self.extract_entities(claim)) > 0:
            # If purely entity focused? Hard to say strictly.
            pass
            
        return 'factual' # Default

    def extract_entities(self, claim: str) -> List[str]:
        """Extract named entities using Spacy or Regex fallback."""
        if self.use_spacy and self.nlp:
            doc = self.nlp(claim)
            return [ent.text for ent in doc.ents]
        
        # Simple Regex Fallback for entities (Capitalized phrases, Numbers)
        entities = []
        # Extract Capitalized Words (potential names/places)
        caps = re.findall(r'\b[A-Z][a-z]+\b(?: \b[A-Z][a-z]+\b)*', claim)
        entities.extend(caps)
        # Extract Numbers
        nums = re.findall(r'\b\d+(?:[\.,]\d+)?%?\b', claim)
        entities.extend(nums)
        
        return list(set(entities))

    def _validate_claim(self, claim: Dict) -> bool:
        """Validate claim structure."""
        if not isinstance(claim, dict):
            return False
        if 'claim' not in claim or not claim['claim']:
            return False
        if len(claim['claim'].split()) < 3: # Min 3 words roughly
            return False
        return True

    def _normalize_claim(self, claim_text: str) -> str:
        """Clean up claim text."""
        text = claim_text.strip()
        text = text[0].upper() + text[1:]
        if not text.endswith('.'):
            text += '.'
        return text

if __name__ == "__main__":
    # Test Block
    print("--- Testing ClaimExtractor ---")
    
    # Mocking environment for test if not set (or rely on cached models)
    # If no API key, it might default to rule-based.
    
    extractor = ClaimExtractor(model_provider="openai", model_name="gpt-3.5-turbo")
    
    test_text = "Patient Jane Doe, age 45, was admitted on January 10, 2024 with acute chest pain. ECG showed ST elevation. Troponin levels were 0.8 ng/mL (elevated)."
    
    print(f"\nEncoding test text: {test_text}")
    
    # Force use_llm=False if no API key to demonstrate functionality, 
    # but the prompt asks to test 'extract_atomic_claims' which defaults use_llm=True.
    # We'll try default.
    try:
        claims = extractor.extract_atomic_claims(test_text)
        
        print(f"\nExtracted {len(claims)} claims:")
        for i, c in enumerate(claims):
            print(f" {i+1}. [{c['claim_type'].upper()}] {c['claim']}")
            print(f"    Entities: {c.get('entities', [])}")
            print(f"    Confidence: {c.get('confidence')}")
            print(f"    Source: {c.get('source')}")
            
    except Exception as e:
        print(f"Test failed: {e}")
