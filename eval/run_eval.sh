#!/bin/bash
# =============================================================================
# Run evaluation across saved checkpoints
# =============================================================================
#
# Usage:
#   CHECKPOINT_BASE=outputs/my_experiment/checkpoints_hf_format bash eval/run_eval.sh
#   STEPS="0 200 400" EVAL_DATASETS="aime2024,math500" bash eval/run_eval.sh
#   THINK=false bash eval/run_eval.sh     # Non-reasoning (non-thinking) model
#
# CHECKPOINT_BASE  path to the HuggingFace-format checkpoint directory
# STEPS            which checkpoints to evaluate (space-separated step numbers)
# EVAL_DATASETS    comma-separated list of dataset names (must exist in EVAL_DATA_DIR)
# RES_LENGTH       max response length (match what was used during training)
# N_SAMPLES        number of samples per prompt for avg@N evaluation (default: 16)
# GPUS             comma-separated GPU indices to use
# THINK            true (default) for reasoning/thinking models, false for non-reasoning
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

export CHECKPOINT_BASE=${CHECKPOINT_BASE:-""}

THINK=${THINK:-true}  # true = thinking/reasoning model, false = non-reasoning

if [ "$THINK" = "true" ]; then
    export RES_LENGTH=${RES_LENGTH:-8192}
    export EVAL_DATA_DIR=${EVAL_DATA_DIR:-"$WORKSPACE_DIR/data/eval/think"}
    export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}   # smaller: 8k responses use more KV-cache
    MODE_LABEL="REASONING (thinking)"
else
    export RES_LENGTH=${RES_LENGTH:-2048}
    export EVAL_DATA_DIR=${EVAL_DATA_DIR:-"$WORKSPACE_DIR/data/eval/non-think"}
    export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}   # larger: 2k responses use less KV-cache
    MODE_LABEL="NON-REASONING (non-thinking)"
fi

export TEMPERATURE=${TEMPERATURE:-1.0}
export N_SAMPLES=${N_SAMPLES:-16}

EXPERIMENT_BASE=$(dirname "$CHECKPOINT_BASE")
RES_LEN_SHORT=$((RES_LENGTH / 1024))k
TEMP_STR=$(echo "$TEMPERATURE" | sed 's/\./_/')
export OUTPUT_DIR=${OUTPUT_DIR:-"$EXPERIMENT_BASE/eval_results_${RES_LEN_SHORT}_t${TEMP_STR}_n${N_SAMPLES}"}

STEPS=${STEPS:-"0"}
export EVAL_DATASETS=${EVAL_DATASETS:-"aime2024,aime2025,amc_test,math500,scibench_test,scp_test_difficult_1"}
export GPUS=${GPUS:-"0,1,2,3"}
export N_GPUS=${N_GPUS:-4}
export GPU_MEMORY=${GPU_MEMORY:-0.8}

echo "=============================================="
echo "  Running Evaluation Across Checkpoints"
echo "  Mode: $MODE_LABEL"
echo "=============================================="
echo "Checkpoints:  $CHECKPOINT_BASE"
echo "Steps:        $STEPS"
echo "Datasets:     $EVAL_DATASETS"
echo "Eval Data:    $EVAL_DATA_DIR"
echo "Output Dir:   $OUTPUT_DIR"
echo "Generation:   ${RES_LENGTH} tokens, temp=${TEMPERATURE}, n=${N_SAMPLES}"
echo "GPUs:         $GPUS ($N_GPUS)"
echo "Micro Batch:  $MICRO_BATCH_SIZE"
echo "=============================================="
echo ""

TOTAL=$(echo "$STEPS" | wc -w)
CURRENT=0
FAILED_STEPS=""
SUCCEEDED_STEPS=""

for STEP in $STEPS; do
    CURRENT=$((CURRENT + 1))
    MODEL_PATH="$CHECKPOINT_BASE/global_step_$STEP"

    echo ""
    echo "=============================================="
    echo "  [$CURRENT/$TOTAL] Evaluating: global_step_$STEP"
    echo "=============================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Checkpoint not found: $MODEL_PATH"
        echo "Skipping..."
        FAILED_STEPS="$FAILED_STEPS $STEP(not_found)"
        continue
    fi

    export MODEL_PATH
    if bash "$SCRIPT_DIR/verl_eval.sh"; then
        echo ""
        echo "Completed: global_step_$STEP"
        SUCCEEDED_STEPS="$SUCCEEDED_STEPS $STEP"
    else
        echo ""
        echo "FAILED: global_step_$STEP"
        FAILED_STEPS="$FAILED_STEPS $STEP"
    fi

    echo "Waiting 10s for GPU cleanup..."
    sleep 10
    echo ""
done

echo ""
echo "=============================================="
echo "  All evaluations complete!"
echo "=============================================="
echo ""
echo "Succeeded:$SUCCEEDED_STEPS"
if [ -n "$FAILED_STEPS" ]; then
    echo "Failed:$FAILED_STEPS"
fi
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
