#!/usr/bin/env python3
"""
OpenThoughts Math Dataset Downloader
Filters for math domain problems only from the metadata subset
"""

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_NAME = "open-thoughts/OpenThoughts-114k"
METADATA_SUBSET = "metadata"
TARGET_DOMAIN = "math"
NUM_SAMPLES = 50000
CACHE_DIR = "/local2/salman/cache"
SAVE_DIR = "/home/salman/reward-signal-analysis/LLaMA-Factory/mid_train_sft_data/open_thoughts/math_50k"

# ============================================================================
# SCRIPT START
# ============================================================================
from datasets import load_dataset, Dataset
import os
import sys

print("=" * 80)
print("OpenThoughts Math Dataset Downloader")
print("=" * 80)
print(f"Dataset: {DATASET_NAME}")
print(f"Target domain: {TARGET_DOMAIN}")
print(f"Samples to collect: {NUM_SAMPLES}")
print(f"Cache directory: {CACHE_DIR}")
print(f"Save directory: {SAVE_DIR}")
print("=" * 80)

# Step 1: Load metadata to get math problems
print("\n[STEP 1/4] Loading metadata subset to identify math problems...")
try:
    metadata_ds = load_dataset(
        DATASET_NAME,
        METADATA_SUBSET,
        split='train',
        cache_dir=CACHE_DIR
    )
    print(f"[SUCCESS] Loaded {len(metadata_ds)} metadata entries")
except Exception as e:
    print(f"[ERROR] Failed to load metadata: {e}")
    sys.exit(1)

# Step 2: Count math domain entries
print(f"\n[STEP 2/4] Counting math domain entries...")
math_count = sum(1 for entry in metadata_ds if entry.get('domain', '') == TARGET_DOMAIN)

print(f"[SUCCESS] Found {math_count} math domain examples in metadata")

if math_count == 0:
    print("[ERROR] No math examples found in metadata")
    sys.exit(1)

# Step 3: Use metadata directly (it already has system and conversations)
print(f"\n[STEP 3/4] Filtering metadata for math examples...")
print(f"Target: {NUM_SAMPLES} samples")

# Collect math samples directly from metadata
collected_samples = []
count = 0

for entry in metadata_ds:
    domain = entry.get('domain', '')
    
    if domain == TARGET_DOMAIN:
        # This entry is math domain, collect it
        # Only keep system and conversations fields for training
        sample = {
            'system': entry.get('system', ''),
            'conversations': entry.get('conversations', [])
        }
        collected_samples.append(sample)
        count += 1
        
        if count % 5000 == 0:
            print(f"  Progress: {count}/{NUM_SAMPLES} math samples collected")
        
        if count >= NUM_SAMPLES:
            break

print(f"[SUCCESS] Collected {len(collected_samples)} math examples")

if len(collected_samples) == 0:
    print("[ERROR] No math examples found")
    sys.exit(1)

# Step 4: Save dataset
print(f"\n[STEP 4/4] Saving dataset to disk...")
try:
    ds_sampled = Dataset.from_list(collected_samples)
    os.makedirs(SAVE_DIR, exist_ok=True)
    ds_sampled.save_to_disk(SAVE_DIR)
    print(f"[SUCCESS] Dataset saved to: {SAVE_DIR}")
except Exception as e:
    print(f"[ERROR] Failed to save dataset: {e}")
    sys.exit(1)

# Print summary
print("\n" + "=" * 80)
print("DATASET DOWNLOAD COMPLETE")
print("=" * 80)
print(f"Total math examples collected: {len(collected_samples)}")
print(f"Total metadata entries: {len(metadata_ds)}")
print(f"Math examples ratio: {(len(collected_samples)/len(metadata_ds))*100:.2f}%")
print(f"Dataset location: {SAVE_DIR}")
print("\n" + "=" * 80)
print("NEXT STEP")
print("=" * 80)
print("The dataset is saved in Arrow format (raw, no formatting applied)")
print("You can now format it for LLaMA Factory if needed")
print("=" * 80)

