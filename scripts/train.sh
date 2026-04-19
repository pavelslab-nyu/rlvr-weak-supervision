#!/bin/bash
set -euo pipefail
# GRPO training script for "When Can LLMs Learn to Reason with Weak Supervision?"
#
# Usage:
#   bash scripts/train.sh
#   DEBUG=True bash scripts/train.sh          # 1 GPU, small batches — quick sanity check
#
# Any variable in the configuration block below can be overridden from the shell:
#   BASE_MODEL=... TRAIN_DATA=... bash scripts/train.sh

# =============================================================================
#  EXPERIMENT CONFIGURATION — modify only this section
#
#  REWARD_TYPE options:
#    RULE_BASED                    standard verifiable reward (MATH, SCIENCE domains)
#    RULE_BASED_THINKING_FORMAT    same but enforces <think>...</think> format
#                                  (use for Thinking SFT models in Section 4)
#    MAJORITY_VOTE                 self-supervised majority vote proxy reward
#    MAJORITY_VOTE_FORMAT_PENALTY  majority vote + <think>...</think> format enforcement
#                                  (use for Thinking SFT models with majority vote in Section 4)
#    SELF_CERTAINTY                self-supervised self-certainty proxy reward
#    VERIFIER_BASED                for GRAPH domain (Reasoning Gym verifier)
#    VERIFIER_BASED_MAJORITY_VOTE  majority vote on GRAPH domain
#
#  RES_LENGTH:
#    2048  for base and instruct models (Sections 3.1, 3.2, 3.3)
#    8192  for Thinking SFT models (Section 4)
#
#  TOTAL_EPOCHS guide — keeps total training steps ~500 regardless of dataset size:
#    N=8: 3968 | N=64: 496 | N=512: 62 | N=1024: 31 | N=2048: 15
#
#  Paper experiment commands:
#
#  Section 3.1 — Scarce Data (replace N with: 8, 64, 512, 1024, 2048)
#    BASE_MODEL=Qwen/Qwen2.5-Math-1.5B        TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_N.parquet  TOTAL_EPOCHS=31  bash scripts/train.sh
#    BASE_MODEL=Qwen/Qwen2.5-1.5B             TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_N.parquet  TOTAL_EPOCHS=31  bash scripts/train.sh
#    BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct  TRAIN_DATA=data/math/train/llama-3b/sky_math_N.parquet   TOTAL_EPOCHS=31  bash scripts/train.sh
#
#  Section 3.2 — Noisy Rewards (noise pre-applied in data, reward type stays RULE_BASED)
#    BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct  TRAIN_DATA=data/math/noisy/llama-3b-think/sky_math_2048_gamma0.70.parquet  TOTAL_EPOCHS=15  bash scripts/train.sh
#
#  Section 3.3 — Proxy Rewards
#    BASE_MODEL=Qwen/Qwen2.5-Math-1.5B  REWARD_TYPE=MAJORITY_VOTE   TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_1024.parquet  TOTAL_EPOCHS=31  bash scripts/train.sh
#    BASE_MODEL=Qwen/Qwen2.5-Math-1.5B  REWARD_TYPE=SELF_CERTAINTY  TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_1024.parquet  TOTAL_EPOCHS=31  bash scripts/train.sh
#
#  Section 4 — Pre-RL Intervention: CPT + Thinking SFT (RES_LENGTH=8192)
#  (Download Llama-3B-CPT-ThinkSFT from HuggingFace — see README)
#    BASE_MODEL=[your-org]/Llama-3B-CPT-ThinkSFT  REWARD_TYPE=RULE_BASED_THINKING_FORMAT    RES_LENGTH=8192  TRAIN_DATA=data/math/train/llama-3b-think/sky_math_8.parquet              TOTAL_EPOCHS=3968  bash scripts/train.sh
#    BASE_MODEL=[your-org]/Llama-3B-CPT-ThinkSFT  REWARD_TYPE=MAJORITY_VOTE_FORMAT_PENALTY  RES_LENGTH=8192  TRAIN_DATA=data/math/train/llama-3b-think/sky_math_1024.parquet           TOTAL_EPOCHS=31    bash scripts/train.sh
#    BASE_MODEL=[your-org]/Llama-3B-CPT-ThinkSFT  REWARD_TYPE=RULE_BASED_THINKING_FORMAT    RES_LENGTH=8192  TRAIN_DATA=data/math/noisy/llama-3b-think/sky_math_2048_gamma0.70.parquet  TOTAL_EPOCHS=15    bash scripts/train.sh
# =============================================================================

export DEBUG=${DEBUG:-False}

# --- Model ---
export BASE_MODEL=${BASE_MODEL:-"Qwen/Qwen2.5-Math-1.5B"}

# --- Data ---
export TRAIN_DATA=${TRAIN_DATA:-"data/math/train/qwen-math-1.5b/sky_math_1024.parquet"}
# Use eval/non-think for base/instruct models (Sections 3.1-3.3)
# Use eval/think for Thinking SFT models (Section 4)
export EVAL_DATASETS=${EVAL_DATASETS:-"data/eval/non-think/aime2024.parquet,data/eval/non-think/aime2025.parquet,data/eval/non-think/math500.parquet,data/eval/non-think/amc_test.parquet"}

# --- Reward ---
# See REWARD_TYPE options in the header above.
export REWARD_TYPE=${REWARD_TYPE:-"RULE_BASED"}

# --- Response length ---
# 2048 for base/instruct models; 8192 for Thinking SFT models (Section 4)
export RES_LENGTH=${RES_LENGTH:-2048}

# --- Training duration ---
# See TOTAL_EPOCHS guide in the header above.
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-31}

# --- Output ---
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"experiment"}
export SAVE_DIR=${SAVE_DIR:-"outputs"}

# --- GPUs ---
# Set to a comma-separated list of GPU indices, e.g. "0,1,2,3" or "4,5,6,7"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# =============================================================================
#  DERIVED SETTINGS — do not modify below this line
# =============================================================================

export REWARD_MODEL_TYPE=$REWARD_TYPE

if [ "$DEBUG" = "True" ]; then
    unset VLLM_ATTENTION_BACKEND
    CUDA_VISIBLE_DEVICES="0"
    N_GPUS_PER_NODE=1
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH=4
    MAX_PROMPT_LENGTH=512
    RES_LENGTH=512
    GROUP_SIZE=4
    PPO_MICRO_BATCH_SIZE_PER_GPU=1
    LOG_PROB_MICRO_BATCH_SIZE=1
    GPU_MEMORY_UTILIZATION=0.6
    MAX_NUM_BATCHED_TOKENS=4096
else
    export VLLM_ATTENTION_BACKEND=XFORMERS
    N_GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    MAX_PROMPT_LENGTH=2048
    GPU_MEMORY_UTILIZATION=0.7

    if [ "$RES_LENGTH" -ge 8192 ]; then
        # Long-form thinking responses (Section 4: Thinking SFT models)
        TRAIN_BATCH_SIZE=64
        PPO_MINI_BATCH=32
        PPO_MICRO_BATCH_SIZE_PER_GPU=4
        LOG_PROB_MICRO_BATCH_SIZE=4
        MAX_NUM_BATCHED_TOKENS=12288
    else
        # Standard responses (Sections 3.1, 3.2, 3.3)
        TRAIN_BATCH_SIZE=64
        PPO_MINI_BATCH=64
        PPO_MICRO_BATCH_SIZE_PER_GPU=8
        LOG_PROB_MICRO_BATCH_SIZE=16
        MAX_NUM_BATCHED_TOKENS=4096
    fi

    GROUP_SIZE=8
fi

# Majority vote requires TTRL parameters and adjusted batch sizes
if [ "$REWARD_TYPE" = "MAJORITY_VOTE" ] || [ "$REWARD_TYPE" = "MAJORITY_VOTE_FORMAT_PENALTY" ] || [ "$REWARD_TYPE" = "VERIFIER_BASED_MAJORITY_VOTE" ]; then
    if [ "$DEBUG" = "True" ]; then
        export TTRL_N_VOTES_PER_PROMPT=4
        export TTRL_N_SAMPLES_PER_PROMPT=4
    else
        export TTRL_N_VOTES_PER_PROMPT=16
        export TTRL_N_SAMPLES_PER_PROMPT=8
        PPO_MINI_BATCH=32
    fi
    GROUP_SIZE=$TTRL_N_SAMPLES_PER_PROMPT
    OPTIMIZER_OFFLOAD=True
else
    OPTIMIZER_OFFLOAD=False
fi

ACTOR_LR=1e-6
ENTROPY_COEFF=0.0
USE_KL_LOSS=True
KL_LOSS_COEF=0.001
KL_LOSS_TYPE=low_var_kl
TENSOR_MODEL_PARALLEL_SIZE=1
PARAM_OFFLOAD=False
REF_PARAM_OFFLOAD=True
USE_REMOVE_PADDING=True
ENABLE_GRADIENT_CHECKPOINTING=True
SAVE_FREQ=50
TEST_FREQ=25

CUSTOM_REWARD_PATH="$(pwd)/reward_function.py"
CHECKPOINT_DIR="$SAVE_DIR/$EXPERIMENT_NAME/checkpoints"
LOG_FILE="$SAVE_DIR/$EXPERIMENT_NAME/logs/train.log"
MLFLOW_DIR="$SAVE_DIR/$EXPERIMENT_NAME/mlflow"
ROLLOUT_DIR="$SAVE_DIR/$EXPERIMENT_NAME/rollouts/training"
VALIDATION_DIR="$SAVE_DIR/$EXPERIMENT_NAME/rollouts/validation"
export MLFLOW_TRACKING_URI=file://$MLFLOW_DIR

# =============================================================================
#  VALIDATE
# =============================================================================

if [ -z "$BASE_MODEL" ]; then
    echo "Error: BASE_MODEL is not set."
    exit 1
fi

if [ -z "$TRAIN_DATA" ]; then
    echo "Error: TRAIN_DATA is not set."
    exit 1
fi

# Build val_files list from comma-separated EVAL_DATASETS
IFS=',' read -ra _eval_arr <<< "$EVAL_DATASETS"
test_files="["
for i in "${!_eval_arr[@]}"; do
    [ $i -gt 0 ] && test_files="${test_files},"
    test_files="${test_files}'${_eval_arr[$i]}'"
done
test_files="${test_files}]"

mkdir -p "$CHECKPOINT_DIR" "$(dirname "$LOG_FILE")" "$MLFLOW_DIR" "$ROLLOUT_DIR" "$VALIDATION_DIR"

# =============================================================================
#  TRAIN
# =============================================================================

echo "Base model:   $BASE_MODEL"
echo "Train data:   $TRAIN_DATA"
echo "Reward type:  $REWARD_TYPE"
echo "Res length:   $RES_LENGTH"
echo "Epochs:       $TOTAL_EPOCHS"
echo "GPUs:         $CUDA_VISIBLE_DEVICES ($N_GPUS_PER_NODE)"
echo "Output:       $SAVE_DIR/$EXPERIMENT_NAME"
echo ""

python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files="$TRAIN_DATA" \
data.val_files="$test_files" \
data.train_batch_size=$TRAIN_BATCH_SIZE \
data.max_prompt_length=$MAX_PROMPT_LENGTH \
data.max_response_length=$RES_LENGTH \
data.filter_overlong_prompts=True \
data.truncation='error' \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
actor_rollout_ref.model.use_remove_padding=$USE_REMOVE_PADDING \
actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFF \
actor_rollout_ref.model.enable_gradient_checkpointing=$ENABLE_GRADIENT_CHECKPOINTING \
actor_rollout_ref.actor.fsdp_config.param_offload=$PARAM_OFFLOAD \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OPTIMIZER_OFFLOAD \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
actor_rollout_ref.rollout.n=$GROUP_SIZE \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
algorithm.use_kl_in_reward=False \
reward_model.reward_manager=batch \
custom_reward_function.path=$CUSTOM_REWARD_PATH \
custom_reward_function.name=compute_score_batch \
trainer.critic_warmup=0 \
trainer.default_hdfs_dir=null \
trainer.default_local_dir=$CHECKPOINT_DIR \
trainer.resume_mode=auto \
trainer.logger='["console","mlflow"]' \
trainer.project_name=$EXPERIMENT_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
trainer.nnodes=1 \
trainer.save_freq=$SAVE_FREQ \
trainer.test_freq=$TEST_FREQ \
trainer.total_epochs=$TOTAL_EPOCHS \
trainer.rollout_data_dir=$ROLLOUT_DIR \
trainer.validation_data_dir=$VALIDATION_DIR 2>&1 | tee "$LOG_FILE"
