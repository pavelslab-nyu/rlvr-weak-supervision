#!/usr/bin/env python3

# Configuration
INPUT_DIR = "/local2/salman/mid_train_sft_data/leon_nemotron_thinking"
OUTPUT_DIR = "/home/salman/reward-signal-analysis/LLaMA-Factory/mid_train_sft_data/leon_nemotron_thinking/50k"
OUTPUT_FILE_NAME = "data_sft_50k_leon_nemotron_thinking.json"
SYSTEM_PROMPT = "You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}."

from datasets import load_from_disk
import json
import os

# Load dataset
print(f"Loading dataset from {INPUT_DIR}...")
ds = load_from_disk(INPUT_DIR)
print(f"Loaded {len(ds)} examples")

# Convert to ShareGPT format for LLaMA Factory
sft_data = []
for example in ds:
    # Get the prompt (user question) and target (assistant response with thinking)
    user_content = example['prompt']
    assistant_content = example['target']
    
    # No prefix removal needed for thinking version - data is clean
    # Just use the content as-is
    
    # Build conversation with system prompt
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_content
        },
        {
            "role": "assistant",
            "content": assistant_content
        }
    ]
    
    sft_data.append({
        "messages": messages
    })

# Save as JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(sft_data)} examples to {output_file}")
