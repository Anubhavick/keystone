import os
import logging
from typing import Optional, Dict, Any
import anthropic
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CorrectionEngine")

class CorrectionEngine:
    """
    AI-powered engine to rewrite contradicted claims to be factually accurate 
    based on the retrieved evidence.
    """

    def __init__(self, api_key: str = None, provider: str = "anthropic"):
        """
        Initialize CorrectionEngine.
        
        Args:
            api_key: API key for the provider.
            provider: 'anthropic' or 'openai'. Defaults to 'anthropic' to match ClaimExtractor preference.
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") # fallback default
        self.client = None
        
        if self.provider == "anthropic":
            if not self.api_key:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                logger.warning("Anthropic API Key missing.")
        
        elif self.provider == "openai":
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                 logger.warning("OPENAI_API_KEY missing.")
            else:
                if OPENAI_AVAILABLE:
                    self.client = OpenAI(api_key=key)
                else:
                    logger.error("OpenAI library not installed.")
            
    def generate_correction(self, claim: str, evidence: str) -> Optional[str]:
        """
        Generate a corrected version of the claim based on evidence.
        """
        if not claim or not evidence:
            return None
            
        prompt = f"""You are a helpful fact-checking assistant. 
The following claim has been found to be FALSE based on the provided evidence.

Claim: "{claim}"
Evidence: "{evidence}"

Task: Rewrite the claim to be factually ACCURATE and supported by the evidence. 
Keep the rewriting concise and professional. Do not add conversational filler.
If the evidence is insufficient to correct it, return "Insufficient evidence to correct."

Corrected Claim:"""

        try:
            if self.provider == "anthropic" and self.client:
                # Use the user's preferred model
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20240620", # Using standard recent stable model
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
            elif self.provider == "openai" and self.client:
                # OpenAI Implementation
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                return response.choices[0].message.content.strip()
                
            else:
                return "AI provider not available for correction."
                
        except Exception as e:
            logger.error(f"Error generating correction: {e}")
            return None

if __name__ == "__main__":
    # Test
    print("Testing Correction Engine...")
    engine = CorrectionEngine()
    c = "The earth is flat."
    e = "The earth is an oblate spheroid."
    print(f"Original: {c}")
    print(f"Evidence: {e}")
    # print(f"Correction: {engine.generate_correction(c, e)}")
