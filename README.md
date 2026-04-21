# When Can LLMs Learn to Reason with Weak Supervision?

<p align="center">
<b>Salman Rahman*, Jingyan Shen*, Anna Mordvina, Hamid Palangi, Saadia Gabriel, Pavel Izmailov</b>
</p>

<p align="center">
<a href="https://arxiv.org/abs/2604.18574" target="_blank">[Paper]</a> &nbsp;&nbsp;
<a href="https://salmanrahman.net/rlvr-weak-supervision" target="_blank">[Project Page]</a> &nbsp;&nbsp;
<a href="https://huggingface.co/collections/pavelslab-nyu/rlvr-weak-supervision" target="_blank">[Models]</a>
</p>

We study when RLVR generalizes under weak supervision (scarce data, noisy rewards, proxy rewards) across Qwen and Llama models on Math, Science, and Graph domains. We find that generalization is governed by **saturation dynamics**. Models with extended pre-saturation phases generalize from as few as **8 examples**, tolerating noisy rewards and even proxy rewards, while rapidly saturating models fail. The root cause of failure is **unfaithful reasoning**, not lack of diversity. **The fix:** continual pre-training combined with supervised fine-tuning on explicit reasoning traces before RL recovers generalization across all three weak supervision settings.

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

We release three pre-RL intervention checkpoints used in Section 4 on [HuggingFace](https://huggingface.co/collections/pavelslab-nyu/rlvr-weak-supervision):

| Model | Description | HuggingFace |
|---|---|---|
| Llama-3.2-3B-CPT-Math-ThinkSFT | Llama-3.2-3B + continual pre-training (52B math tokens) + thinking SFT | [pavelslab-nyu/Llama-3.2-3B-CPT-Math-ThinkSFT](https://huggingface.co/pavelslab-nyu/Llama-3.2-3B-CPT-Math-ThinkSFT) |
| Llama-3.2-3B-CPT-Math | Llama-3.2-3B + continual pre-training (52B math tokens) | [pavelslab-nyu/Llama-3.2-3B-CPT-Math](https://huggingface.co/pavelslab-nyu/Llama-3.2-3B-CPT-Math) |
| Llama-3.2-3B-ThinkSFT | Llama-3.2-3B + thinking SFT (no CPT) | [pavelslab-nyu/Llama-3.2-3B-ThinkSFT](https://huggingface.co/pavelslab-nyu/Llama-3.2-3B-ThinkSFT) |

---

## Training

All experiments use a single script with environment variable overrides:

```bash
bash scripts/rl/train.sh
```

**Section 3.1 — Scarce Data**

Experiments vary the model, domain (Math / Science / Graph), and training set size `N` ∈ {8, 64, 512, 1024, 2048}. Epoch guide to keep total steps ~500: N=8 → 496 (8× upsampled), N=64 → 496, N=512 → 62, N=1024 → 31, N=2048 → 15.

Example run (Science, N=1024):

```bash
BASE_MODEL=Qwen/Qwen2.5-Math-1.5B \
TRAIN_DATA=data/science/train/qwen-math-1.5b/scp_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/rl/train.sh
```

**Section 3.2 — Noisy Rewards**

Noise is pre-applied in the data files. Swap in the desired `gamma` level; `REWARD_TYPE` stays `RULE_BASED`.

```bash
BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
TRAIN_DATA=data/math/noisy/llama-3b/sky_math_2048_gamma0.70.parquet \
TOTAL_EPOCHS=15 \
bash scripts/rl/train.sh
```

**Section 3.3 — Proxy Rewards**

```bash
# Majority vote
BASE_MODEL=Qwen/Qwen2.5-Math-1.5B \
REWARD_TYPE=MAJORITY_VOTE \
TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/rl/train.sh

# Self-certainty
BASE_MODEL=Qwen/Qwen2.5-Math-1.5B \
REWARD_TYPE=SELF_CERTAINTY \
TRAIN_DATA=data/math/train/qwen-math-1.5b/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/rl/train.sh
```

**Section 4 — Pre-RL Intervention**

Section 4 sweeps across model initializations (Base, ThinkSFT, CPT, CPT+ThinkSFT) and SFT types (thinking vs. non-thinking).

**Option A — Use our released checkpoints** (recommended): download from [HuggingFace](https://huggingface.co/collections/pavelslab-nyu/rlvr-weak-supervision) and set `BASE_MODEL` accordingly.

**Option B — Reproduce from scratch**:
1. Register your datasets in `LLaMA-Factory/data/dataset_info.json` (CPT data for Step 1, SFT traces for Step 2)
2. Update `model_name_or_path` and `output_dir` in the config yamls under `scripts/cpt/` and `scripts/sft/`

```bash
# Step 1 — Continual pre-training (produces Llama-3.2-3B-CPT-Math)
bash scripts/cpt/cpt.sh

# Step 2 — Thinking SFT (produces Llama-3.2-3B-CPT-Math-ThinkSFT or Llama-3.2-3B-ThinkSFT)
bash scripts/sft/sft.sh
```

Then run RL. Thinking models require `RES_LENGTH=8192` and `REWARD_TYPE=RULE_BASED_THINKING_FORMAT`. Example commands for CPT+ThinkSFT:

```bash
# Scarce data
BASE_MODEL=pavelslab-nyu/Llama-3.2-3B-CPT-Math-ThinkSFT \
REWARD_TYPE=RULE_BASED_THINKING_FORMAT \
RES_LENGTH=8192 \
TRAIN_DATA=data/math/train/llama-3b-think/sky_math_8.parquet \
TOTAL_EPOCHS=496 \
bash scripts/rl/train.sh

# Noisy reward
BASE_MODEL=pavelslab-nyu/Llama-3.2-3B-CPT-Math-ThinkSFT \
REWARD_TYPE=RULE_BASED_THINKING_FORMAT \
RES_LENGTH=8192 \
TRAIN_DATA=data/math/noisy/llama-3b-think/sky_math_2048_gamma0.70.parquet \
TOTAL_EPOCHS=15 \
bash scripts/rl/train.sh

# Majority vote proxy reward
BASE_MODEL=pavelslab-nyu/Llama-3.2-3B-CPT-Math-ThinkSFT \
REWARD_TYPE=MAJORITY_VOTE_FORMAT_PENALTY \
RES_LENGTH=8192 \
TRAIN_DATA=data/math/train/llama-3b-think/sky_math_1024.parquet \
TOTAL_EPOCHS=31 \
bash scripts/rl/train.sh
```

See `scripts/rl/train.sh` for the full list of configurable options and `REWARD_TYPE` descriptions.

---

## Evaluation

Evaluate a trained checkpoint across multiple datasets:

```bash
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

We measure reasoning faithfulness, whether the model's chain-of-thought logically supports its final answer, using an LLM-as-a-judge framework (details in Section 3.4 of the paper). Code and instructions coming soon in `faithfulness_analysis/`.

---

## Citation

```bibtex
@misc{rahman2026llmslearnreasonweak,
      title={When Can LLMs Learn to Reason with Weak Supervision?}, 
      author={Salman Rahman and Jingyan Shen and Anna Mordvina and Hamid Palangi and Saadia Gabriel and Pavel Izmailov},
      year={2026},
      eprint={2604.18574},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.18574}, 
}
```
