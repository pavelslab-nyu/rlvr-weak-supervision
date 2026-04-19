# When Can LLMs Learn to Reason with Weak Supervision?

**Salman Rahman\*, Jingyan Shen\*, Anna Mordvina, Hamid Palangi, Saadia Gabriel, Pavel Izmailov**

[\[Paper\]](https://arxiv.org/abs/XXXX.XXXXX) &nbsp; [\[Project Page\]](https://link/rlvr-weak-supervision)

We study when RLVR generalizes under weak supervision (scarce data, noisy rewards, proxy rewards) across Qwen and Llama models on Math, Science, and Graph domains. We find that generalization is governed by **saturation dynamics** — models with extended pre-saturation phases generalize from as few as 8 examples, while rapidly saturating models fail. The root cause of failure is **unfaithful reasoning**, not lack of diversity. **The fix:** continual pre-training + supervised fine-tuning on explicit reasoning traces before RL recovers generalization across all three settings.

---

## Installation

```bash
git clone https://github.com/pavelslab-nyu/rlvr-weak-supervision.git
cd rlvr-weak-supervision
pip install -e ".[vllm]"
pip install flash-attn --no-build-isolation
pip install math-verify reasoning-gym mlflow
```

---

## Models

We release the pre-RL intervention checkpoint used in Section 4:

| Model | Description | HuggingFace |
|---|---|---|
| Llama-3B-CPT-ThinkSFT | Llama-3.2-3B-Base + continual pre-training + thinking SFT | `[your-org]/Llama-3B-CPT-ThinkSFT` |

Base models (Qwen2.5-Math-1.5B, Qwen2.5-1.5B, Llama-3.2-3B-Instruct) are available directly from HuggingFace.

---

## Training

All experiments are run with a single script. Set the variables for your experiment and run:

```bash
bash scripts/train.sh
```

**Section 3.1 — Scarce Data** (vary `N` over 8, 64, 512, 1024, 2048)

```bash
# Qwen2.5-Math-1.5B on MATH
BASE_MODEL=Qwen/Qwen2.5-Math-1.5B \
TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/train.sh

# Llama-3.2-3B-Instruct on MATH
BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
TRAIN_DATA=data/math/train/llama-3b/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/train.sh
```

**Section 3.2 — Noisy Rewards** (noise is pre-applied in the data files; `gamma` controls label noise level)

```bash
BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
TRAIN_DATA=data/math/noisy/llama-3b-think/sky_math_2048_gamma0.70.parquet \
TOTAL_EPOCHS=15 \
bash scripts/train.sh
```

**Section 3.3 — Proxy Rewards**

```bash
# Majority vote
BASE_MODEL=Qwen/Qwen2.5-Math-1.5B \
REWARD_TYPE=MAJORITY_VOTE \
TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/train.sh

# Self-certainty
BASE_MODEL=Qwen/Qwen2.5-Math-1.5B \
REWARD_TYPE=SELF_CERTAINTY \
TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/train.sh
```

**Section 4 — Pre-RL Intervention (CPT + Thinking SFT)**

```bash
# Scarce data with thinking format reward
BASE_MODEL=[your-org]/Llama-3B-CPT-ThinkSFT \
REWARD_TYPE=RULE_BASED_THINKING_FORMAT \
RES_LENGTH=8192 \
TRAIN_DATA=data/math/train/llama-3b-think/sky_math_8.parquet \
TOTAL_EPOCHS=3968 \
bash scripts/train.sh

# Noisy reward with thinking format reward
BASE_MODEL=[your-org]/Llama-3B-CPT-ThinkSFT \
REWARD_TYPE=RULE_BASED_THINKING_FORMAT \
RES_LENGTH=8192 \
TRAIN_DATA=data/math/noisy/llama-3b-think/sky_math_2048_gamma0.70.parquet \
TOTAL_EPOCHS=15 \
bash scripts/train.sh

# Majority vote proxy reward with thinking format
BASE_MODEL=[your-org]/Llama-3B-CPT-ThinkSFT \
REWARD_TYPE=MAJORITY_VOTE_FORMAT_PENALTY \
RES_LENGTH=8192 \
TRAIN_DATA=data/math/train/llama-3b-think/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/train.sh
```

See `scripts/train.sh` for the full list of configurable options and `REWARD_TYPE` descriptions.

---

## Evaluation

Evaluate a trained checkpoint across multiple datasets:

```bash
# Convert verl checkpoints to HuggingFace format first (see verl docs)

# Reasoning/thinking model (8k response)
CHECKPOINT_BASE=/path/to/checkpoints_hf_format \
STEPS="100 200 300" \
bash eval/run_eval.sh

# Non-reasoning model (2k response)
CHECKPOINT_BASE=/path/to/checkpoints_hf_format \
STEPS="100 200 300" \
THINK=false \
bash eval/run_eval.sh
```

This generates a `generations/` folder under each step directory. To compute pass@k metrics:

```bash
python3 eval/pass_k_eval.py \
    /path/to/eval_results \
    data/eval/think \
    --benchmarks aime2024 aime2025 amc_test math500
```

---

## Faithfulness Analysis

We measure reasoning faithfulness — whether the model's chain-of-thought genuinely drives its final answer — following the perturbation-based protocol described in Section 4 of the paper.

Code and instructions coming soon in `faithfulness_analysis/`.

---

## Citation

```bibtex
@article{rahman2026rlvr,
  title     = {When Can LLMs Learn to Reason with Weak Supervision?},
  author    = {Rahman, Salman and Shen, Jingyan and Mordvina, Anna and Palangi, Hamid and Gabriel, Saadia and Izmailov, Pavel},
  journal   = {arXiv preprint},
  year      = {2026}
}
```
