#!/usr/bin/env python3
"""
OpenThoughts Dataset Preparation for LLaMA Factory
Downloads and prepares OpenThoughts-114k dataset in ShareGPT format
"""

# ============================================================================
# CONFIGURATION - Modify these variables as needed
# ============================================================================
DATASET_NAME = "open-thoughts/OpenThoughts-114k"
NUM_SAMPLES = 50000
FORMATTING = "sharegpt"  # ShareGPT format (OpenAI variant)

# Directories
CACHE_DIR = "/local2/salman/cache"  # HuggingFace cache (has space!)
SAVE_DIR = "/local2/salman/mid_train_sft_data/open_thought_50k"
OUTPUT_JSON_DIR = "/home/salman/reward-signal-analysis/LLaMA-Factory/mid_train_sft_data/open_thoughts/50k"
OUTPUT_JSON_FILE = "open_thought_50k_data.json"

# Optional: Custom system prompt (set to None to keep original)
CUSTOM_SYSTEM_PROMPT = None
# Example: "You are a helpful AI assistant that thinks step-by-step before answering."

# ============================================================================
# SCRIPT START
# ============================================================================
from datasets import load_dataset, Dataset
import json
import os
import sys

print("=" * 80)
print("OpenThoughts-114k Dataset Preparation for LLaMA Factory")
print("=" * 80)
print(f"Dataset: {DATASET_NAME}")
print(f"Samples to collect: {NUM_SAMPLES}")
print(f"Format: {FORMATTING}")
print(f"Cache directory: {CACHE_DIR}")
print(f"Save directory: {SAVE_DIR}")
print(f"Output JSON: {OUTPUT_JSON_DIR}/{OUTPUT_JSON_FILE}")
print("=" * 80)

# Step 1: Download dataset in streaming mode
print("\n[STEP 1/4] Loading dataset in streaming mode...")
try:
    ds_stream = load_dataset(
        DATASET_NAME,
        split='train',
        streaming=True,
        cache_dir=CACHE_DIR
    )
    print("[SUCCESS] Dataset loaded successfully in streaming mode")
except Exception as e:
    print(f"[ERROR] Failed to load dataset: {e}")
    sys.exit(1)

# Step 2: Collect samples
print(f"\n[STEP 2/4] Collecting {NUM_SAMPLES} samples...")
collected_samples = []
count = 0

try:
    for example in ds_stream:
        collected_samples.append(example)
        count += 1
        
        if count % 5000 == 0:
            print(f"  Progress: {count}/{NUM_SAMPLES} samples collected")
        
        if count >= NUM_SAMPLES:
            break
    
    print(f"[SUCCESS] Collected {len(collected_samples)} samples")
except Exception as e:
    print(f"[ERROR] Failed to collect samples: {e}")
    sys.exit(1)

# Step 3: Convert to Dataset and save raw data
print(f"\n[STEP 3/4] Saving raw dataset to disk...")
try:
    ds_sampled = Dataset.from_list(collected_samples)
    os.makedirs(SAVE_DIR, exist_ok=True)
    ds_sampled.save_to_disk(SAVE_DIR)
    print(f"[SUCCESS] Raw dataset saved to: {SAVE_DIR}")
except Exception as e:
    print(f"[ERROR] Failed to save raw dataset: {e}")
    sys.exit(1)

# Step 4: Prepare data in LLaMA Factory ShareGPT format
print(f"\n[STEP 4/4] Preparing data in ShareGPT format for LLaMA Factory...")

# Inspect first example to understand structure
print("\n[INFO] Inspecting first example structure:")
first_example = collected_samples[0]
print(f"  Keys in example: {list(first_example.keys())}")
if 'conversations' in first_example:
    print(f"  Number of conversations: {len(first_example['conversations'])}")
    if first_example['conversations']:
        print(f"  First conversation keys: {list(first_example['conversations'][0].keys())}")
        print(f"  First conversation sample: {first_example['conversations'][0]}")
    if 'system' in first_example:
        print(f"  System prompt present: {bool(first_example['system'])}")
else:
    print("[WARNING] 'conversations' field not found in first example")

# Role mapping from ShareGPT format to OpenAI format
ROLE_MAPPING = {
    'human': 'user',
    'gpt': 'assistant',
    'system': 'system'
}

# Prepare data for LLaMA Factory
sft_data = []
skipped_count = 0

for idx, example in enumerate(collected_samples):
    try:
        # Check if 'conversations' field exists
        if 'conversations' not in example:
            if idx < 10:  # Only print first 10 warnings to avoid spam
                print(f"[WARNING] Example {idx} has no 'conversations' field, skipping")
            skipped_count += 1
            continue
        
        conversations = example['conversations']
        
        # Validate conversations structure
        if not isinstance(conversations, list) or len(conversations) == 0:
            if idx < 10:
                print(f"[WARNING] Example {idx} has invalid conversations structure, skipping")
            skipped_count += 1
            continue
        
        # Build messages list
        messages = []
        
        # Add system prompt if exists
        system_prompt = example.get('system', '')
        if system_prompt:
            # Use custom system prompt if specified, otherwise use original
            if CUSTOM_SYSTEM_PROMPT is not None:
                system_prompt = CUSTOM_SYSTEM_PROMPT
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        elif CUSTOM_SYSTEM_PROMPT is not None:
            # Add custom system prompt even if original doesn't have one
            messages.append({
                "role": "system",
                "content": CUSTOM_SYSTEM_PROMPT
            })
        
        # Convert conversations from ShareGPT format (from/value) to OpenAI format (role/content)
        for conv in conversations:
            from_role = conv.get('from', '')
            content = conv.get('value', '')
            
            # Map role from ShareGPT format to OpenAI format
            role = ROLE_MAPPING.get(from_role, from_role)
            
            messages.append({
                "role": role,
                "content": content
            })
        
        # Add to dataset
        sft_data.append({
            "messages": messages
        })
        
    except Exception as e:
        if idx < 10:  # Only print first 10 errors to avoid spam
            print(f"[WARNING] Error processing example {idx}: {e}, skipping")
        skipped_count += 1
        continue

print(f"[SUCCESS] Processed {len(sft_data)} examples successfully")
if skipped_count > 0:
    print(f"[WARNING] Skipped {skipped_count} examples due to errors or missing fields")

# Validate that we have data
if len(sft_data) == 0:
    print("[ERROR] No valid data collected. Please check dataset structure.")
    sys.exit(1)

# Save as JSON for LLaMA Factory
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_JSON_DIR, OUTPUT_JSON_FILE)

try:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
    print(f"[SUCCESS] JSON data saved to: {output_path}")
except Exception as e:
    print(f"[ERROR] Failed to save JSON: {e}")
    sys.exit(1)

# Print summary
print("\n" + "=" * 80)
print("DATASET PREPARATION COMPLETE")
print("=" * 80)
print(f"Total samples processed: {len(sft_data)}")
print(f"Samples skipped: {skipped_count}")
print(f"Format: ShareGPT (OpenAI variant)")
print(f"Raw dataset location: {SAVE_DIR}")
print(f"JSON file location: {output_path}")
print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
print("\n" + "=" * 80)
print("NEXT STEPS")
print("=" * 80)
print("1. Add this entry to LLaMA-Factory/data/dataset_info.json:")
print("-" * 80)
print(f'''{{
  "open_thoughts_50k": {{
    "file_name": "mid_train_sft_data/open_thoughts/50k/{OUTPUT_JSON_FILE}",
    "formatting": "sharegpt",
    "columns": {{
      "messages": "messages"
    }},
    "tags": {{
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant",
      "system_tag": "system"
    }}
  }}
}}''')
print("-" * 80)
print("2. Use 'dataset: open_thoughts_50k' in your training YAML configuration")
print("=" * 80)

