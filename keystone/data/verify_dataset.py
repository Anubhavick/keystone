import json
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DatasetVerifier")

DATA_DIR = Path("keystone/data/processed")

def verify_dataset():
    print(f"\n=== 1. Directory Structure: {DATA_DIR} ===")
    if not DATA_DIR.exists():
        print(f"❌ Directory not found: {DATA_DIR}")
        return

    files = list(DATA_DIR.glob("*"))
    for f in files:
        print(f" - {f.name} ({f.stat().st_size / 1024:.2f} KB)")

    # Load main dataset
    dataset_path = DATA_DIR / "halueval_summarization.json"
    if not dataset_path.exists():
        print(f"❌ Main dataset file missing: {dataset_path}")
        return

    print(f"\n=== Loading {dataset_path.name} ===")
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # 2. Statistics
    total = len(data)
    hallucinated = sum(1 for d in data if d["label"] == "hallucinated")
    faithful = sum(1 for d in data if d["label"] == "faithful")
    
    # Avg length
    lengths = [len(d.get("generated_text", "").split()) for d in data]
    avg_len = sum(lengths) / len(lengths) if lengths else 0

    print("\n=== 2. Statistics ===")
    print(f" - Total examples loaded: {total}")
    print(f" - Hallucinated: {hallucinated} ({hallucinated/total*100:.1f}%)")
    print(f" - Faithful: {faithful} ({faithful/total*100:.1f}%)")
    print(f" - Average text length: {avg_len:.2f} words")

    # 3. Display Samples
    print("\n=== 3. Samples ===")
    
    # Find one of each
    hal_sample = next((d for d in data if d["label"] == "hallucinated"), None)
    faith_sample = next((d for d in data if d["label"] == "faithful"), None)

    print("\n--- [Sample 1: Hallucinated] ---")
    if hal_sample:
        print(f"ID: {hal_sample.get('id')}")
        print(f"Generated Text: {hal_sample.get('generated_text')[:200]}...")
        print(f"Label: {hal_sample.get('label')}")
    else:
        print("None found.")

    print("\n--- [Sample 2: Faithful] ---")
    if faith_sample:
        print(f"ID: {faith_sample.get('id')}")
        print(f"Generated Text: {faith_sample.get('generated_text')[:200]}...")
        print(f"Label: {faith_sample.get('label')}")
    else:
        print("None found.")

    # 4. Confirm Files (Train/Test)
    # The previous step might not have created train/test json files specifically, only the main one.
    # Let's check and create them if missing to satisfy the user request.
    print("\n=== 4. Verifying Train/Test Split Files ===")
    train_path = DATA_DIR / "train.json"
    test_path = DATA_DIR / "test.json"
    samples_path = DATA_DIR / "samples.json"

    required_files = {
        "train.json": train_path,
        "test.json": test_path,
        "samples.json": samples_path
    }

    missing_files = []
    for name, path in required_files.items():
        if path.exists():
             print(f" ✅ {name} exists.")
        else:
             print(f" ⚠️ {name} missing.")
             missing_files.append(name)
    
    if "train.json" in missing_files or "test.json" in missing_files:
        print("\n Generating missing train/test split files...")
        # Simple split logic here to avoid importing complex Loader dependencies if easy
        # But let's verify import works
        try:
             # Just split the raw list manually to be quick and robust
             import random
             random.seed(42)
             random.shuffle(data)
             split_idx = int(total * 0.8)
             train_data = data[:split_idx]
             test_data = data[split_idx:]
             
             with open(train_path, "w") as f:
                 json.dump(train_data, f, indent=2)
             with open(test_path, "w") as f:
                 json.dump(test_data, f, indent=2)
             
             print(f" ✅ Created train.json ({len(train_data)} items)")
             print(f" ✅ Created test.json ({len(test_data)} items)")

        except Exception as e:
            print(f" ❌ Failed to create split: {e}")

    # 5. JSON Structure
    print("\n=== 5. JSON Structure Example ===")
    if hal_sample:
        print(json.dumps(hal_sample, indent=2))

if __name__ == "__main__":
    verify_dataset()
