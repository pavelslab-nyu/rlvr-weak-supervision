#!/usr/bin/env python3

# Configuration
DATASET_NAME = "leonli66/nemotron-sft-math"
CONFIG_NAME = "thinking"
SAVE_DIR = "/local2/salman/mid_train_sft_data/leon_nemotron_thinking"
NUM_SAMPLES = 50000
CACHE_DIR = "/local2/salman/cache"  # Cache downloads to /local2 (has space!)

from datasets import load_dataset, Dataset
import os

# Load dataset in STREAMING mode (only downloads what we need!)
print(f"Loading {DATASET_NAME} ({CONFIG_NAME}) in streaming mode...")
ds_stream = load_dataset(DATASET_NAME, CONFIG_NAME, split='train', streaming=True, cache_dir=CACHE_DIR)

# Filter and collect samples
print(f"Filtering for is_multiturn = False and collecting {NUM_SAMPLES} samples...")
filtered_samples = []
count = 0

for example in ds_stream:
    if example.get('is_multiturn') == False:
        filtered_samples.append(example)
        count += 1
        if count % 5000 == 0:
            print(f"  Collected {count}/{NUM_SAMPLES} samples...")
        if count >= NUM_SAMPLES:
            break

print(f"Collected {len(filtered_samples)} samples")

# Convert to Dataset
print("Converting to Dataset format...")
ds_sampled = Dataset.from_list(filtered_samples)

# Save to disk
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Saving to {SAVE_DIR}...")
ds_sampled.save_to_disk(SAVE_DIR)

print(f"✅ Done! Saved {len(ds_sampled)} examples to {SAVE_DIR}")
