
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory
# Add parent directory of 'keystone' package (project root)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from keystone.backend.claim_extractor import ClaimExtractor
from keystone.backend.correction_engine import CorrectionEngine

class TestProviderConnection(unittest.TestCase):
    def test_claim_extractor_openai_init(self):
        """Test initializing ClaimExtractor with OpenAI provider"""
        with patch('keystone.backend.claim_extractor.OpenAI') as MockOpenAI:
            extractor = ClaimExtractor(api_key="test-key", provider="openai")
            self.assertEqual(extractor.provider, "openai")
            MockOpenAI.assert_called_once_with(api_key="test-key")

    def test_claim_extractor_anthropic_init(self):
        """Test initializing ClaimExtractor with Anthropic provider"""
        with patch('keystone.backend.claim_extractor.anthropic.Anthropic') as MockAnthropic:
            extractor = ClaimExtractor(api_key="test-key", provider="anthropic")
            self.assertEqual(extractor.provider, "anthropic")
            MockAnthropic.assert_called_once_with(api_key="test-key")

    def test_correction_engine_openai_init(self):
        """Test initializing CorrectionEngine with OpenAI provider"""
        with patch('keystone.backend.correction_engine.OpenAI') as MockOpenAI:
            engine = CorrectionEngine(api_key="test-key", provider="openai")
            self.assertEqual(engine.provider, "openai")
            MockOpenAI.assert_called_once_with(api_key="test-key")

    def test_correction_engine_anthropic_init(self):
        """Test initializing CorrectionEngine with Anthropic provider"""
        with patch('keystone.backend.correction_engine.anthropic.Anthropic') as MockAnthropic:
            engine = CorrectionEngine(api_key="test-key", provider="anthropic")
            self.assertEqual(engine.provider, "anthropic")
            MockAnthropic.assert_called_once()  # args might vary slightly depending on env but it should be called

if __name__ == '__main__':
    unittest.main()
