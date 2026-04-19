#!/bin/bash
# =============================================================================
# VERL Evaluation Script
# =============================================================================
# Evaluates a checkpoint on specified datasets using verl's val_only mode.
#
# Usage:
#   bash verl_eval.sh                                    # Use defaults
#   MODEL_PATH=/path/to/ckpt bash verl_eval.sh           # Different checkpoint
#   N_GPUS=2 GPUS="4,5" bash verl_eval.sh                # Use 2 GPUs
#   N_GPUS=4 GPUS="0,1,2,3" bash verl_eval.sh            # Use 4 GPUs
#   RES_LENGTH=32768 N_SAMPLES=8 bash verl_eval.sh       # Different generation settings
#   EVAL_DATASETS="math500" bash verl_eval.sh            # Single dataset
#
# Dataset sizes (for reference):
#   - math500: 500 examples
#   - amc_test: 83 examples
#   - scp_test_difficult_1: 50 examples
#
# =============================================================================

set -e

# =============================================================================
#  GPU SETTINGS
# =============================================================================

# Prevent Ray from connecting to an existing cluster
export RAY_ADDRESS=""

# NCCL timeout: 3 hours to accommodate long 32k-token generation
# Read by verl/workers/fsdp_workers.py at init_process_group
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800}

export CUDA_VISIBLE_DEVICES="${GPUS:-0,1,2,3}"
N_GPUS=${N_GPUS:-4}
GPU_MEMORY=${GPU_MEMORY:-0.8}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}

echo "Using GPUs: $CUDA_VISIBLE_DEVICES ($N_GPUS)"

# =============================================================================
#  CONFIGURATION
# =============================================================================

# HuggingFace-format checkpoint path
MODEL_PATH=${MODEL_PATH:-"/path/to/checkpoints_hf_format/global_step_50"}

RES_LENGTH=${RES_LENGTH:-8192}           # Response length: 8192, 16384, 32768
TEMPERATURE=${TEMPERATURE:-1}            # 1 = sampling enabled
N_SAMPLES=${N_SAMPLES:-16}               # Number of samples per prompt

# Available datasets: aime2024, aime2025, amc_test, math500, minerva_math, olympiad_bench,
#   scibench_test, scp_test_difficult_1, stem__gpqa_diamond_198, stem__mmlu_sci_college_346,
#   super_gpqa_in_domain_319, super_gpqa_out_domain_250
EVAL_DATASETS=${EVAL_DATASETS:-"aime2024,aime2025,amc_test,math500,minerva_math,olympiad_bench,scibench_test,scp_test_difficult_1,stem__gpqa_diamond_198,stem__mmlu_sci_college_346,super_gpqa_in_domain_319,super_gpqa_out_domain_250"}
REWARD_TYPE=${REWARD_TYPE:-"RULE_BASED"}
export REWARD_MODEL_TYPE=$REWARD_TYPE

# =============================================================================
#  AUTO-DERIVED PATHS
# =============================================================================

STEP_NUM=$(basename "$MODEL_PATH" | sed 's/global_step_//')
EXPERIMENT_BASE=$(dirname "$(dirname "$MODEL_PATH")")
OUTPUT_DIR=${OUTPUT_DIR:-"$EXPERIMENT_BASE/eval_results"}
# Format: step_50_8192len_t0.6_n8
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"step_${STEP_NUM}_${RES_LENGTH}len_t${TEMPERATURE}_n${N_SAMPLES}"}

# =============================================================================
#  BUILD DATASET PATHS
# =============================================================================

# Use script's location to find workspace root (not pwd which can vary)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
# data/eval/think   for reasoning (thinking-format) models
# data/eval/non-think  for non-reasoning models
EVAL_DATA_DIR=${EVAL_DATA_DIR:-"$WORKSPACE_DIR/data/eval/think"}
REWARD_FUNCTION="$WORKSPACE_DIR/reward_function.py"

build_eval_files() {
    local files=""
    IFS=',' read -ra DATASETS <<< "$EVAL_DATASETS"
    for dataset in "${DATASETS[@]}"; do
        dataset=$(echo "$dataset" | xargs)
        local path="$EVAL_DATA_DIR/${dataset}.parquet"
        if [ -f "$path" ]; then
            [ -z "$files" ] && files="'$path'" || files="$files,'$path'"
        else
            echo "WARNING: Dataset not found: $path" >&2
        fi
    done
    echo "[$files]"
}

eval_files=$(build_eval_files)

# =============================================================================
#  SETUP
# =============================================================================

VALIDATION_DIR="$OUTPUT_DIR/$EXPERIMENT_NAME/generations"
LOG_FILE="$OUTPUT_DIR/$EXPERIMENT_NAME/eval.log"

mkdir -p "$VALIDATION_DIR"

# Use V1 engine (faster than XFORMERS fallback to V0)
unset VLLM_ATTENTION_BACKEND

# =============================================================================
#  PRINT CONFIG
# =============================================================================

echo ""
echo "=============================================="
echo "  VERL Evaluation"
echo "=============================================="
echo "Model:        $MODEL_PATH"
echo "Step:         $STEP_NUM"
echo "Output:       $OUTPUT_DIR/$EXPERIMENT_NAME"
echo ""
echo "Generation:   ${RES_LENGTH} tokens, temp=${TEMPERATURE}, n=${N_SAMPLES}"
echo "Datasets:     $EVAL_DATASETS"
echo "Eval Data:    $EVAL_DATA_DIR"
echo "Eval Files:   $eval_files"
echo "GPUs:         $CUDA_VISIBLE_DEVICES ($N_GPUS)"
echo "GPU Memory:   $GPU_MEMORY"
echo "Micro Batch:  $MICRO_BATCH_SIZE"
echo "NCCL Timeout: ${NCCL_TIMEOUT}s ($((NCCL_TIMEOUT / 3600))h)"
echo "Reward:       $REWARD_TYPE"
echo "=============================================="
echo ""

# =============================================================================
#  RUN EVALUATION
# =============================================================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$eval_files \
    data.val_files=$eval_files \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=$RES_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY \
    actor_rollout_ref.rollout.max_model_len=$((2048 + RES_LENGTH)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((2048 + RES_LENGTH)) \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.val_kwargs.n=$N_SAMPLES \
    actor_rollout_ref.rollout.val_kwargs.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.val_kwargs.do_sample=$([ "$TEMPERATURE" != "0" ] && echo "True" || echo "False") \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path=$REWARD_FUNCTION \
    custom_reward_function.name=compute_score_batch \
    trainer.val_only=True \
    trainer.val_before_train=True \
    trainer.resume_mode=disable \
    trainer.total_epochs=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.logger='["console"]' \
    trainer.project_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.validation_data_dir=$VALIDATION_DIR \
    2>&1 | tee "$LOG_FILE"

# Capture the exit code from the python command (not tee)
EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Done! Results: $VALIDATION_DIR"
else
    echo "FAILED with exit code $EXIT_CODE"
    echo "Check log: $LOG_FILE"
fi
echo ""

exit $EXIT_CODE
