#!/bin/bash
# Supervised fine-tuning (SFT) on thinking or non-thinking traces.
#
# THINK=true  (default) — thinking format, 43k open-thoughts traces
#             Produces: Llama-3.2-3B-CPT-Math-ThinkSFT  (pavelslab-nyu/Llama-3.2-3B-CPT-Math-ThinkSFT)
#             or        Llama-3.2-3B-ThinkSFT            (pavelslab-nyu/Llama-3.2-3B-ThinkSFT)
#             depending on whether the input model is CPT or base.
#
# THINK=false — non-thinking format, 50k Nemotron math traces (cutoff 2048)
#
# Usage:
#   LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory bash scripts/sft.sh
#   THINK=false LLAMA_FACTORY_DIR=/path/to/LLaMA-Factory bash scripts/sft.sh
#
# Requirements: pip install llamafactory

THINK=${THINK:-"true"}
LLAMA_FACTORY_DIR=${LLAMA_FACTORY_DIR:-"LLaMA-Factory"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# GPU settings
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# Cache
export HF_HOME=${HF_HOME:-"/local2/salman/reward_signal_data/hf_cache"}
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TORCH_HOME="${HF_HOME}/torch"
export TOKENIZERS_PARALLELISM=false
export FORCE_TORCHRUN=1

# NCCL
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

if [ "${THINK}" = "true" ]; then
    CONFIG="${SCRIPT_DIR}/sft_think_config.yaml"
    OUTPUT_DIR=${OUTPUT_DIR:-"/local2/salman/model/sft_model/llama_3b_think_sft"}
else
    CONFIG="${SCRIPT_DIR}/sft_non_think_config.yaml"
    OUTPUT_DIR=${OUTPUT_DIR:-"/local2/salman/model/sft_model/llama_3b_non_think_sft"}
fi

mkdir -p "${OUTPUT_DIR}/logs"

cd "${LLAMA_FACTORY_DIR}"

llamafactory-cli train "${CONFIG}" \
    2>&1 | tee "${OUTPUT_DIR}/logs/sft_$(date +%Y%m%d_%H%M%S).log"
