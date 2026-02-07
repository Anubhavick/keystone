import logging
import os
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    Handles downloading, processing, and caching of fact-checking datasets (HaluEval, FEVER).
    """

    def __init__(self, cache_dir: str = "data/cache", processed_dir: str = "data/processed"):
        """
        Initialize the DatasetLoader.
        
        Args:
            cache_dir: Directory to cache downloaded datasets
            processed_dir: Directory to store processed JSON files
        """
        self.cache_dir = Path(cache_dir)
        self.processed_dir = Path(processed_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets: Dict[str, List[Dict]] = {}
        logger.info(f"DatasetLoader initialized. Cache: {cache_dir}, Processed: {processed_dir}")

    def load_halueval(self, task: str = "summarization", split: str = "data") -> List[Dict[str, Any]]:
        """
        Load HaluEval dataset (hallucination detection).
        
        Args:
           task: 'summarization', 'qa', or 'dialogue'
           split: 'train', 'data' (HaluEval typically has 'data' for the main set on HF)
           
        Returns:
           List of processed examples
        """
        logger.info(f"Loading HaluEval dataset (Task: {task}, Split: {split})...")
        
        try:
            # HaluEval structure on HF is 'pminervini/HaluEval'
            # Tasks are often separate configs or subsets. 
            # Note: Checking HF, 'pminervini/HaluEval' structure usually requires specifying configuration.
            # Using 'summarization' as default.
            
            # Attempt to load from HF
            dataset_name = "pminervini/HaluEval"
            
            # Provide cache_dir so we don't redownload unnecessarily
            hf_dataset = load_dataset(dataset_name, task, split=split, cache_dir=str(self.cache_dir))
            
            processed_data = []
            
            logger.info(f"Processing {len(hf_dataset)} examples from HaluEval...")
            
            for i, example in tqdm(enumerate(hf_dataset), total=len(hf_dataset)):
                # Map fields to our schema
                # HaluEval Summarization fields usually: document, right_summary, hallucinated_summary
                
                # Faithful example
                faithful_entry = {
                    "id": f"halueval_{task}_{i}_faithful",
                    "source_document": example.get("document", ""),
                    "generated_text": example.get("right_summary", ""), # The correct summary
                    "label": "faithful",
                    "hallucination_type": "none",
                    "metadata": {
                        "source": "halueval",
                        "task": task
                    }
                }
                
                # Hallucinated example
                hallucinated_entry = {
                    "id": f"halueval_{task}_{i}_hallucinated",
                    "source_document": example.get("document", ""),
                    "generated_text": example.get("hallucinated_summary", ""),
                    "label": "hallucinated",
                    "hallucination_type": "generated_hallucination", 
                    "metadata": {
                        "source": "halueval",
                        "task": task
                    }
                }
                
                processed_data.append(faithful_entry)
                processed_data.append(hallucinated_entry)
            
            self.datasets[f"halueval_{task}"] = processed_data
            logger.info(f"Successfully processed {len(processed_data)} HaluEval examples.")
            return processed_data

        except Exception as e:
            logger.error(f"Error loading HaluEval: {e}")
            return []

    def load_fever(self, split: str = "train", max_samples: int = 1000) -> List[Dict[str, Any]]:
        """
        Load FEVER dataset (Fact Extraction and VERification).
        
        Args:
            split: 'train', 'dev', or 'test'
            max_samples: Limit number of samples as FEVER is huge
        
        Returns:
            List of processed examples
        """
        logger.info(f"Loading FEVER dataset (Split: {split}, Max: {max_samples})...")
        
        try:
            # FEVER v1.0
            hf_dataset = load_dataset("fever", "v1.0", split=split, cache_dir=str(self.cache_dir), streaming=True)
            
            processed_data = []
            count = 0
            
            logger.info("Processing FEVER examples...")
            
            # Since we used streaming=True (assuming large), we iterate until max_samples
            for i, example in tqdm(enumerate(hf_dataset)):
                if count >= max_samples:
                    break
                
                # FEVER Labels: 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'
                label_map = {
                    "SUPPORTS": "faithful",
                    "REFUTES": "hallucinated", # Contradiction -> Hallucination in our context
                    "NOT ENOUGH INFO": "unverifiable"
                }
                
                original_label = example.get("label", "")
                mapped_label = label_map.get(original_label)
                
                if not mapped_label:
                    continue
                    
                entry = {
                    "id": f"fever_{split}_{i}",
                    "source_document": "", # FEVER implies Wikipedia, but just has evidence. We might put claim here effectively.
                    # Wait, FEVER is claim verification. 'evidence_wiki_url', 'evidence_sentence_id' etc.
                    # The 'claim' is the generated text to check. 'evidence' needs resolving.
                    # For simplicity in this loader, we might skip full evidence resolution if complex,
                    # or just map 'claim' -> 'generated_text'.
                    "generated_text": example.get("claim", ""),
                    "label": mapped_label,
                    "hallucination_type": "contradictory" if mapped_label == "hallucinated" else "none",
                    "metadata": {
                        "source": "fever",
                        "original_label": original_label
                    }
                }
                
                processed_data.append(entry)
                count += 1
                
            self.datasets["fever"] = processed_data
            logger.info(f"Successfully processed {len(processed_data)} FEVER examples.")
            return processed_data

        except Exception as e:
            logger.error(f"Error loading FEVER: {e}")
            return []

    def create_train_test_split(self, data: List[Dict], test_size: float = 0.2) -> Tuple[List[Dict], List[Dict]]:
        """
        Split data into train and test sets, stratified by label.
        """
        if not data:
            return [], []
            
        try:
            labels = [d.get("label", "unknown") for d in data]
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=42, 
                stratify=labels
            )
            logger.info(f"Split data: Train ({len(train_data)}), Test ({len(test_data)})")
            return train_data, test_data
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            # Fallback to simple split if stratification fails (e.g. only 1 class)
            pivot = int(len(data) * (1 - test_size))
            return data[:pivot], data[pivot:]

    def save_to_json(self, data: List[Dict], filepath: str) -> None:
        """
        Save processed data to JSON file.
        """
        try:
            path = self.processed_dir / Path(filepath).name # Force into processed dir for safety or just use path
            # If absolute or relative path provided, respect it but ensure dir exists
            target_path = Path(filepath)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(data)} records to {target_path}")
        except Exception as e:
            logger.error(f"Error saving to JSON {filepath}: {e}")

    def get_sample_batch(self, n: int = 5) -> List[Dict]:
        """
        Return n random examples for quick testing.
        """
        all_data = []
        for d in self.datasets.values():
            all_data.extend(d)
            
        if not all_data:
            logger.warning("No data loaded to sample from.")
            return []
            
        n = min(n, len(all_data))
        return random.sample(all_data, n)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return dataset statistics.
        """
        all_data = []
        for d in self.datasets.values():
            all_data.extend(d)
        
        if not all_data:
            return {"total": 0}
            
        stats = {
            "total_examples": len(all_data),
            "hallucinated_count": sum(1 for d in all_data if d["label"] == "hallucinated"),
            "faithful_count": sum(1 for d in all_data if d["label"] == "faithful"),
            "unverifiable_count": sum(1 for d in all_data if d["label"] == "unverifiable"),
            "sources": list(self.datasets.keys())
        }
        
        # Calculate avg length
        lengths = [len(d["generated_text"].split()) for d in all_data if "generated_text" in d]
        stats["avg_word_count"] = sum(lengths) / len(lengths) if lengths else 0
        
        return stats

if __name__ == "__main__":
    # Automatic execution logic
    loader = DatasetLoader()
    
    # 1. Load HaluEval
    halueval_data = loader.load_halueval(task="summarization", split="data")
    
    # 2. Statistics
    stats = loader.get_statistics()
    print("\n=== Dataset Statistics ===")
    print(json.dumps(stats, indent=2))
    
    # 3. Save Processed
    loader.save_to_json(halueval_data, "data/processed/halueval_summarization.json")
    
    # 4. Save Samples
    samples = loader.get_sample_batch(n=5)
    loader.save_to_json(samples, "data/processed/samples.json")
    
    print("\n✅ Dataset setup complete.")
