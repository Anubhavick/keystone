import anthropic
import json
import os
import re
import logging
from typing import List, Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Try importing transformers/torch, but don't crash if missing
try:
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    LOCAL_ML_AVAILABLE = True
except ImportError:
    LOCAL_ML_AVAILABLE = False
    logger.warning("Transformers/Torch not found. Local model extraction will be disabled.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI library not found. OpenAI provider will be disabled.")


class ClaimExtractor:
    def __init__(self, api_key: str = None, provider: str = "anthropic", local_model_path: str = None):
        """
        Initialize ClaimExtractor with optional local model, Anthropic, or OpenAI client.
        Args:
            api_key: Optional API key override.
            provider: 'anthropic' or 'openai' (default: 'anthropic').
            local_model_path: Path to the local fine-tuned model directory. 
        """
        self.local_model = None
        self.tokenizer = None
        self.use_local = False
        self.provider = provider
        self.client = None
        
        # Determine strict path to model if not provided
        if not local_model_path:
            # Look in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            local_model_path = os.path.join(current_dir, "saved_claim_extractor_model")

        # 1. Try Loading Local Model
        if LOCAL_ML_AVAILABLE and os.path.exists(local_model_path):
            try:
                logger.info(f"Loading local model from {local_model_path}...")
                self.tokenizer = T5Tokenizer.from_pretrained(local_model_path)
                self.local_model = T5ForConditionalGeneration.from_pretrained(local_model_path)
                
                # Move to device if available
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
                self.local_model.to(device)
                
                self.use_local = True
                logger.info(f"Local claim extraction model loaded successfully on {device}.")
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
        elif LOCAL_ML_AVAILABLE and not os.path.exists(local_model_path):
             logger.info(f"Local model not found at {local_model_path}. Using API fallback.")
        
        # 2. Setup API Client (Anthropic or OpenAI)
        if not self.use_local:
            if self.provider == "anthropic":
                key = api_key or os.getenv("ANTHROPIC_API_KEY")
                if not key:
                    logger.warning("ANTHROPIC_API_KEY not found. Extraction will fail.")
                else:
                    self.client = anthropic.Anthropic(api_key=key)
            elif self.provider == "openai":
                if not OPENAI_AVAILABLE:
                    logger.error("OpenAI library not installed. Please run `pip install openai`.")
                else:
                    key = api_key or os.getenv("OPENAI_API_KEY")
                    if not key:
                        logger.warning("OPENAI_API_KEY not found. Extraction will fail.")
                    else:
                        self.client = OpenAI(api_key=key)

    def extract_atomic_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract claims using Local Model (preferred) or API (fallback)"""
        if not text or not text.strip():
            return []

        if self.use_local and self.local_model and self.tokenizer:
            return self._extract_with_local_model(text)
        elif self.client:
            if self.provider == "anthropic":
                return self._extract_with_claude(text)
            elif self.provider == "openai":
                return self._extract_with_openai(text)
        
        logger.error("No extraction method available (No local model and no valid API client).")
        return []

    def _extract_with_local_model(self, text: str) -> List[Dict[str, Any]]:
        """Run inference using the local T5 model"""
        try:
            input_text = f"extract claims: {text}"
            
            # Prepare inputs
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.local_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.local_model.generate(
                    **inputs, 
                    max_length=512,
                    num_beams=4, # Use beam search for better quality
                    early_stopping=True
                )
            
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse JSON
            try:
                # Sometimes models output single quotes or minor syntax errors
                # Try basic cleanup
                cleaned = decoded_output.replace("'", '"')
                claims = json.loads(cleaned)
                
                # Normalize output format
                if isinstance(claims, list):
                    for i, c in enumerate(claims):
                        if isinstance(c, dict):
                            if 'claim_id' not in c: c['claim_id'] = f'local_claim_{i}'
                            if 'confidence' not in c: c['confidence'] = 0.85
                            if 'type' not in c: c['type'] = 'factual'
                        else:
                            # Unexpected format
                            continue
                    return claims
                else:
                    return []
                    
            except json.JSONDecodeError:
                logger.warning(f"Local model output was not valid JSON: {decoded_output}")
                return []

        except Exception as e:
            logger.error(f"Local model inference error: {e}")
            if self.client:
                logger.info(f"Falling back to {self.provider} due to local error.")
                if self.provider == "anthropic":
                    return self._extract_with_claude(text)
                elif self.provider == "openai":
                    return self._extract_with_openai(text)
            return []

    def _extract_with_claude(self, text: str) -> List[Dict[str, Any]]:
        """Extraction logic using Claude"""
        try:
            prompt = f"""Extract 3-7 atomic factual claims from this text. Return ONLY valid JSON array:
[{{"claim": "exact text", "type": "factual"}}]

Text: {text}"""
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            content = response.content[0].text
            clean_content = re.sub(r'```json\n?|\n?```', '', content).strip()
            
            claims = json.loads(clean_content)
            
            for i, c in enumerate(claims):
                c['claim_id'] = f'claim_{i}'
                c['confidence'] = 0.9
                if 'type' not in c: c['type'] = 'factual'
            
            return claims

        except Exception as e:
            logger.error(f"Claude Extraction Error: {e}")
            return []

    def _extract_with_openai(self, text: str) -> List[Dict[str, Any]]:
        """Extraction logic using OpenAI GPT-4o"""
        try:
            prompt = f"""You are a precise fact-checking assistant. Extract 3-7 atomic factual claims from the text below.
Return ONLY a valid JSON array of objects.
Format:
[
  {{
    "claim": "The exact claim text here",
    "type": "factual"
  }}
]

Text: {text}"""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that outputs only JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content
            # OpenAI with json_object usually returns a dict like {"claims": [...] } or just the array if prompted well
            # But specific instruction "Return ONLY a valid JSON array" usually works, 
            # however response_format={"type": "json_object"} requires the output to be a valid JSON object, NOT a list.
            # So the prompt should ask for { "claims": [...] }
            
            # Let's adjust parsing to handle both key-based or list-based if possible, 
            # but strict json mode requires an object.
            # actually better to just parse whatever valid json comes back
            
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    claims = data
                elif isinstance(data, dict):
                     # Look for likely keys
                     for k in ['claims', 'result', 'output']:
                         if k in data and isinstance(data[k], list):
                             claims = data[k]
                             break
                     else:
                         # Fallback if no list found
                         claims = []
                else:
                    claims = []
            except json.JSONDecodeError:
                # Try regex cleanup if json mode failed or wasn't used
                clean_content = re.sub(r'```json\n?|\n?```', '', content).strip()
                claims = json.loads(clean_content)

            # Post-process
            for i, c in enumerate(claims):
                if 'claim_id' not in c: c['claim_id'] = f'openai_claim_{i}'
                if 'confidence' not in c: c['confidence'] = 0.9
                if 'type' not in c: c['type'] = 'factual'
            
            return claims

        except Exception as e:
            logger.error(f"OpenAI Extraction Error: {e}")
            return []


