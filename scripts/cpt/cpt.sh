#!/bin/bash
# Continual pre-training (CPT) for Llama-3.2-3B on math reasoning data.
#
# Produces: Llama-3.2-3B-CPT-Math  (pavelslab-nyu/Llama-3.2-3B-CPT-Math)
#
# Usage:
#   LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory bash scripts/cpt.sh
#
# Requirements: pip install llamafactory

LLAMA_FACTORY_DIR=${LLAMA_FACTORY_DIR:-"LLaMA-Factory"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU settings
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# Cache
export HF_HOME=${HF_HOME:-"$HOME/.cache/huggingface"}
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HF_HOME}/torch"
export TOKENIZERS_PARALLELISM=false
export FORCE_TORCHRUN=1

# NCCL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

OUTPUT_DIR=${OUTPUT_DIR:-"cpt_output"}
mkdir -p "${OUTPUT_DIR}/logs"

cd "${LLAMA_FACTORY_DIR}"

llamafactory-cli train "${SCRIPT_DIR}/cpt_config.yaml" \
    2>&1 | tee "${OUTPUT_DIR}/logs/cpt_$(date +%Y%m%d_%H%M%S).log"
