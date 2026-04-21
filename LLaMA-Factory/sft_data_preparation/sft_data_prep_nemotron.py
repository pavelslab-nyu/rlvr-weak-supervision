#!/usr/bin/env python3

# Configuration
INPUT_DIR = "/local2/salman/mid_train_sft_data/leon_nemotron"
OUTPUT_DIR = "/home/salman/reward-signal-analysis/LLaMA-Factory/mid_train_sft_data/leon_nemotron/50k"
SYSTEM_PROMPT = "You are a helpful AI Assistant, designed to provide well-reasoned and detailed responses. You FIRST think about the reasoning process step by step and then provide the user with the answer. Please enclose your final answer in the box: \\boxed{Your Answer}."

from datasets import load_from_disk
import json
import os

# Load dataset
ds = load_from_disk(INPUT_DIR)

# Convert to ShareGPT format for LLaMA Factory
sft_data = []
for example in ds:
    # Build conversation with system prompt
    conversations = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]
    
    # Add user messages from prompt (remove "input: " prefix)
    for msg in example['prompt']:
        content = msg['content']
        # Remove "input: " prefix if present
        if content.startswith("input: "):
            content = content[7:]  # len("input: ") = 7
        conversations.append({
            "role": msg['role'],
            "content": content
        })
    
    # Add assistant response (remove "output: " prefix)
    target_content = example['target']
    if target_content.startswith("output: "):
        target_content = target_content[8:]  # len("output: ") = 8
    
    conversations.append({
        "role": "assistant",
        "content": target_content
    })
    
    sft_data.append({
        "messages": conversations
    })

# Save as JSON
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, "data.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sft_data, f, ensure_ascii=False, indent=2)

print(f"Saved {len(sft_data)} examples to {output_file}")
