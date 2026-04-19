# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

from collections import defaultdict
from enum import Enum
import os

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

# BOUNDARY_METHOD flag - controlled by environment variable
BOUNDARY_METHOD = os.environ.get('BOUNDARY_METHOD', 'False').lower() == 'true'

POLICY_LOSS_REGISTRY = {}


def register_policy_loss(name):
    def decorator(func):
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


ADV_ESTIMATOR_REGISTRY = {}


def register_adv_est(name_or_enum):
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    INTUITOR = "intuitor"
    TANGO_GRPO = "tango_grpo"
    SELFDISTILL = "selfdistill"
    CONSTANT = "constant"
    NEGATIVE_SELFDISTILL = "negative_selfdistill"
    POSTIVE_GRPO = "positive_grpo"
    NEGATIVE_GRPO = "negative_grpo"


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        # Original method: same advantage for all tokens
        scores = scores.unsqueeze(-1) * response_mask
        # scores = scores.unsqueeze(-1) * response_mask
    return scores, scores

@register_adv_est(AdvantageEstimator.SELFDISTILL)  # or: @register_adv_est("selfdistill")
def compute_selfdistill_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
):
    """
    SelfDistill: use group-average as advantage (no centering, no std-scaling).
    Advantage_i = mean score within its group; Returns_i = outcome score.
    """
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    # outcome per sample
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # group means
    id2sum = defaultdict(lambda: torch.zeros((), device=device, dtype=dtype))
    id2cnt = defaultdict(int)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2sum[idx] = id2sum[idx] + scores[i]
            id2cnt[idx] += 1
        id2mean = {k: (id2sum[k] / max(1, id2cnt[k])) for k in id2sum.keys()}

    adv_scalar = torch.empty_like(scores)
    with torch.no_grad():
        for i in range(scores.shape[0]):
            adv_scalar[i] = id2mean[index[i]]

    advantages = adv_scalar.unsqueeze(-1) * response_mask
    returns = scores.unsqueeze(-1) * response_mask  # keep returns as outcome

    return advantages, returns

from collections import defaultdict
import numpy as np
import torch

@register_adv_est(AdvantageEstimator.POSTIVE_GRPO)  
def compute_positive_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # was `str`; make it bool
):
    """
    GRPO on outcome-only reward (one scalar per response):
    - Sum token rewards only on valid tokens.
    """
    ### DEBUG ###
    tlr = token_level_rewards
    if tlr.is_floating_point():
        close0 = torch.isclose(tlr, torch.zeros_like(tlr), rtol=0.0, atol=1e-6)
        close1 = torch.isclose(tlr, torch.ones_like(tlr),  rtol=0.0, atol=1e-6)
    else:
        close0 = (tlr == 0)
        close1 = (tlr == 1)
    non_binary = ~(close0 | close1)
    if non_binary.any():
        idxs = non_binary.nonzero(as_tuple=False)
        vals = tlr[non_binary]
        total = tlr.numel()
        bad = vals.numel()
        print(f"[WARN] token_level_rewards contains non-binary values: {bad}/{total}")
        max_show = 10
        for j in range(min(max_show, idxs.size(0))):
            i0, i1 = idxs[j].tolist()
            v = float(vals[j].detach().cpu())
            print(f"  at [{i0},{i1}] -> {v}")
            
    scores = token_level_rewards.sum(dim=-1)
    zero_mask = torch.isclose(scores, torch.zeros_like(scores), rtol=0.0, atol=1e-8)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if zero_mask[i]:
                scores[i] = 0.0
            elif norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
                
    advantages = scores.unsqueeze(-1) * response_mask
    returns    = scores.unsqueeze(-1) * response_mask  
    
    # Ensure all advantages are positive
    assert (advantages == advantages.abs()).all(), f"Advantages is {advantages} are not positive."

    return advantages, returns

@register_adv_est(AdvantageEstimator.NEGATIVE_GRPO)  
def compute_negative_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,  # was `str`; make it bool
):
    """
    GRPO on outcome-only reward (one scalar per response):
    - Sum token rewards only on valid tokens.
    """
    scores = token_level_rewards.sum(dim=-1)
    ones_mask = torch.isclose(scores, torch.ones_like(scores), rtol=0.0, atol=1e-8)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if ones_mask[i]:
                scores[i] = 0.0
            elif norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
    advantages = scores.unsqueeze(-1) * response_mask
    returns    = scores.unsqueeze(-1) * response_mask 
    
    assert (advantages == -advantages.abs()).all(), "Advantages are not negative"

    return advantages, returns
@register_adv_est(AdvantageEstimator.CONSTANT)  # or: @register_adv_est("constant")
def compute_constant_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
):
    """
    SelfDistill: use group-average as advantage (no centering, no std-scaling).
    Advantage_i = mean score within its group; Returns_i = outcome score.
    """
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    # outcome per sample
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    advantages = torch.ones_like(scores).unsqueeze(-1) * response_mask
    returns = scores.unsqueeze(-1) * response_mask  # keep returns as outcome

    return advantages, returns

@register_adv_est(AdvantageEstimator.NEGATIVE_SELFDISTILL)  # or: @register_adv_est("constant")
def compute_negative_selfdistill_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
):
    """
    SelfDistill: use group-average as advantage (no centering, no std-scaling).
    Advantage_i = mean score within its group; Returns_i = outcome score.
    """
    device = token_level_rewards.device
    dtype = token_level_rewards.dtype

    # outcome per sample
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # group means
    id2sum = defaultdict(lambda: torch.zeros((), device=device, dtype=dtype))
    id2cnt = defaultdict(int)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2sum[idx] = id2sum[idx] + scores[i]
            id2cnt[idx] += 1
        id2mean = {k: (id2sum[k] / max(1, id2cnt[k])) for k in id2sum.keys()}

    adv_scalar = torch.empty_like(scores)
    with torch.no_grad():
        for i in range(scores.shape[0]):
            adv_scalar[i] = -id2mean[index[i]]

    advantages = adv_scalar.unsqueeze(-1) * response_mask
    returns = scores.unsqueeze(-1) * response_mask  # keep returns as outcome

    return advantages, returns

@register_adv_est(AdvantageEstimator.TANGO_GRPO)  # RL Tango style GRPO with outcome + step rewards
def compute_tango_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    outcome_weight: float = 0.6,
    process_weight: float = 0.4,
    alpha: float = 0.2,  # Fixed 20% step weight as requested
    reward_extra_info=None,
    **kwargs,
):
    """
    Compute RL Tango style GRPO advantages combining outcome and step rewards.
    
    This follows the exact RL Tango approach:
    1. Outcome rewards: GRPO on final rewards (80% weight)
    2. Step rewards: Normalized step rewards with cumulative returns (20% weight)
    3. Combined: (1-α) * outcome_adv + α * step_adv where α=0.2
    
    Args:
        token_level_rewards: (B, L) - outcome rewards (typically at EOS tokens)
        response_mask: (B, L) - mask for valid response tokens
        index: (B,) - prompt group IDs for GRPO comparison
        epsilon: Small constant for numerical stability
        norm_adv_by_std_in_grpo: Whether to normalize by group standard deviation
        outcome_weight: Weight for outcome rewards (default 0.6)
        process_weight: Weight for step rewards (default 0.4) 
        alpha: Fixed weight for step rewards (default 0.2 = 20%)
        reward_extra_info: Dict containing step reward information
        
    Returns:
        advantages: (B, L) - combined advantages
        returns: (B, L) - combined returns
    """
    bsz, seq_len = token_level_rewards.shape
    device = token_level_rewards.device
    
    # Debug flag - controlled by environment variable DEBUG
    import os
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # 1. Compute outcome advantages (standard GRPO)
    outcome_scores = token_level_rewards.sum(dim=-1)  # (B,)
    
    if DEBUG:
        print(f"\nTANGO GRPO DEBUG - STEP 1: OUTCOME REWARDS")
        print(f"   Raw outcome scores: {outcome_scores.tolist()}")
        print(f"   Batch size: {bsz}, Sequence length: {seq_len}")
        print(f"   Group indices (UIDs): {index.tolist()}")
    
    id2score = defaultdict(list)
    id2mean, id2std = {}, {}
    
    with torch.no_grad():
        # Group outcome scores by prompt index
        for i in range(bsz):
            id2score[index[i]].append(outcome_scores[i])
        
        # Compute group statistics for outcome rewards
        for gid, vals in id2score.items():
            vals_tensor = torch.stack(vals)
            if vals_tensor.numel() > 1:
                id2mean[gid] = vals_tensor.mean()
                id2std[gid] = vals_tensor.std()
            else:
                id2mean[gid] = torch.tensor(0.0, device=device)
                id2std[gid] = torch.tensor(1.0, device=device)
        
        # Normalize outcome scores
        outcome_normalized = torch.zeros_like(outcome_scores)
        for i in range(bsz):
            gid = index[i]
            if norm_adv_by_std_in_grpo:
                outcome_normalized[i] = (outcome_scores[i] - id2mean[gid]) / (id2std[gid] + epsilon)
            else:
                outcome_normalized[i] = outcome_scores[i] - id2mean[gid]
        
        if DEBUG:
            print(f"\nTANGO GRPO DEBUG - STEP 2: OUTCOME NORMALIZATION")
            for gid in id2mean.keys():
                group_indices = [i for i in range(bsz) if index[i] == gid]
                group_scores = [outcome_scores[i].item() for i in group_indices]
                group_normalized = [outcome_normalized[i].item() for i in group_indices]
                print(f"   Group {gid}: raw_scores={group_scores}, mean={id2mean[gid]:.4f}, std={id2std[gid]:.4f}")
                print(f"   Group {gid}: normalized={group_normalized}")
            print(f"   Final outcome_normalized: {outcome_normalized.tolist()}")
        
        # Place outcome advantages at EOS tokens and compute cumulative returns
        outcome_tensor = torch.zeros_like(token_level_rewards)
        last_token_idx = response_mask.sum(dim=-1) - 1  # Find EOS position
        outcome_tensor[torch.arange(bsz, device=device), last_token_idx] = outcome_normalized
        
        # Cumulative returns for outcome (RL Tango style)
        outcome_returns = outcome_tensor.flip(-1).cumsum(-1).flip(-1)
        outcome_advantages = outcome_returns.clone()
        
        # Check for differential advantage mode (PRM_COT_HYBRID_OUTCOME_DIFF_ADV variants)
        REWARD_MODEL_TYPE = os.environ.get('REWARD_MODEL_TYPE', '').upper()
        use_differential_advantage = REWARD_MODEL_TYPE in ['PRM_COT_HYBRID_OUTCOME_DIFF_ADV', 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT']
        use_half_discount = REWARD_MODEL_TYPE == 'PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT'
        
        if DEBUG and use_differential_advantage:
            mode_name = "PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT" if use_half_discount else "PRM_COT_HYBRID_OUTCOME_DIFF_ADV"
            print(f"\nDIFFERENTIAL ADVANTAGE MODE: {mode_name}")
        
        # 2. Extract and process step rewards if available
        if reward_extra_info and "step_rewards" in reward_extra_info and "step_rewards_mask" in reward_extra_info:
            step_rewards = reward_extra_info["step_rewards"]  # (B, L)
            step_rewards_mask = reward_extra_info["step_rewards_mask"]  # (B, L)
            
            
            if DEBUG:
                print(f"\nTANGO GRPO DEBUG - STEP 3: STEP REWARDS")
                for i in range(min(2, bsz)):  # Show first 2 responses
                    step_positions = torch.nonzero(step_rewards_mask[i], as_tuple=True)[0]
                    step_vals = step_rewards[i, step_positions] if len(step_positions) > 0 else torch.tensor([])
                    print(f"   Response {i}: step_positions={step_positions.tolist()}, step_rewards={step_vals.tolist()}")
            
            # Collect all step rewards per prompt group for normalization (RL Tango approach)
            id2mean_step, id2std_step = {}, {}
            
            for gid in id2score.keys():
                group_mask = torch.tensor([index[i] == gid for i in range(bsz)], device=device)
                group_step_rewards = step_rewards[group_mask]  # (group_size, L)
                group_step_mask = step_rewards_mask[group_mask]  # (group_size, L)
                
                # Get all step rewards for this prompt group
                valid_step_rewards = group_step_rewards[group_step_mask]
                
                if valid_step_rewards.numel() <= 1:
                    id2mean_step[gid] = torch.tensor(0.0, device=device)
                    id2std_step[gid] = torch.tensor(1.0, device=device)
                else:
                    id2mean_step[gid] = valid_step_rewards.mean()
                    id2std_step[gid] = valid_step_rewards.std() + epsilon
            
            # Normalize step rewards within each prompt group
            step_tensor = torch.zeros_like(step_rewards)
            for i in range(bsz):
                gid = index[i]
                step_mask = step_rewards_mask[i]  # (L,)
                
                if step_mask.any():
                    if norm_adv_by_std_in_grpo:
                        step_tensor[i, step_mask] = (
                            step_rewards[i, step_mask] - id2mean_step[gid]
                        ) / id2std_step[gid]
                    else:
                        step_tensor[i, step_mask] = step_rewards[i, step_mask] - id2mean_step[gid]
            
            
            # Compute cumulative returns for step rewards (RL Tango key insight)
            # Each token gets credit for all future step rewards
            step_returns = step_tensor.flip(-1).cumsum(-1).flip(-1)  # (B, L)
            step_advantages = step_returns.clone()
            
            # Apply response mask
            step_advantages = step_advantages * response_mask
            step_returns = step_returns * response_mask
            
            if DEBUG:
                print(f"\nTANGO GRPO DEBUG - STEP 4: STEP NORMALIZATION & CUMULATIVE")
                for gid in id2mean_step.keys():
                    print(f"   Group {gid}: step_mean={id2mean_step[gid]:.4f}, step_std={id2std_step[gid]:.4f}")
                # Show first response step advantages
                if bsz > 0:
                    valid_mask = response_mask[0]
                    valid_step_advs = step_advantages[0, valid_mask]
                    print(f"   First response step_advantages (valid tokens): {valid_step_advs.tolist()[:10]}...")  # First 10
            
        else:
            # No step rewards available, use zero step advantages
            step_advantages = torch.zeros_like(outcome_advantages)
            step_returns = torch.zeros_like(outcome_returns)
        
        # 3. Combine outcome and step advantages
        if use_differential_advantage and reward_extra_info and "raw_step_rewards" in reward_extra_info and "step_rewards_mask" in reward_extra_info:
            # PRM_COT_HYBRID_OUTCOME_DIFF_ADV: Start with 100% outcome, then zero out step ranges
            combined_advantages = outcome_advantages.clone()
            combined_returns = outcome_returns.clone()
            
            raw_step_rewards_list = reward_extra_info["raw_step_rewards"]
            step_rewards_mask = reward_extra_info["step_rewards_mask"]  # (B, L)
            
            if DEBUG:
                print(f"\nPRM_COT_HYBRID_OUTCOME_DIFF_ADV DEBUG - STEP RANGE ZEROING")
                print(f"   Starting with 100% outcome advantage for all tokens")
                print(f"   Then zero out step ranges based on correctness")
            
            # Apply step range zeroing logic
            for i in range(bsz):
                # Get the outcome advantage for this response
                first_valid_pos = torch.nonzero(response_mask[i], as_tuple=True)[0]
                if len(first_valid_pos) > 0:
                    outcome_adv = combined_advantages[i, first_valid_pos[0]].item()
                    
                    if i < len(raw_step_rewards_list):
                        raw_rewards = raw_step_rewards_list[i]  # [1.0, -1.0, 1.0, ...]
                        
                        # Find step boundary positions: [154, 207, 260]
                        step_positions = torch.nonzero(step_rewards_mask[i], as_tuple=True)[0]
                        
                        if DEBUG:
                            print(f"   Response {i}: outcome_adv={outcome_adv:.4f}")
                            print(f"   Response {i}: raw_rewards={raw_rewards}")
                            print(f"   Response {i}: step_positions={step_positions.tolist()}")
           
                        # Use available steps (handle length mismatch gracefully)
                        num_steps = min(len(step_positions), len(raw_rewards))
                        
                        if DEBUG and i < 2:
                            print(f"   Response {i}: Using {num_steps} steps (positions={len(step_positions)}, rewards={len(raw_rewards)})")
                        
                        for step_idx in range(num_steps):
                            raw_reward = raw_rewards[step_idx]
                            
                            # Define step range: BEFORE current step boundary to current step boundary (inclusive)
                            if step_idx == 0:
                                # First step: from token 3 (skip initial tokens) to first boundary (inclusive)
                                start_pos = 3  # Skip initial tokens that don't belong to any step
                                end_pos = step_positions[step_idx].item() + 1  # Include boundary token
                            else:
                                # Other steps: from after previous boundary to current boundary (inclusive)
                                start_pos = step_positions[step_idx - 1].item() + 1  # Start after previous boundary
                                end_pos = step_positions[step_idx].item() + 1  # Include current boundary token
                            
                            # Simple differential logic
                            should_zero = False
                            if outcome_adv >= 0:
                                # Positive outcome: zero incorrect steps
                                should_zero = (raw_reward < 0)
                            else:
                                # Negative outcome: zero correct steps
                                should_zero = (raw_reward > 0)
                            
                            if should_zero:
                                # Apply zeroing or half discount based on mode
                                tokens_modified = 0
                                for pos in range(start_pos, min(end_pos, combined_advantages.shape[1])):
                                    if response_mask[i, pos]:
                                        if use_half_discount:
                                            # Half discount: divide by 2 instead of zeroing
                                            combined_advantages[i, pos] = combined_advantages[i, pos] / 2.0
                                        else:
                                            # Original: zero out completely
                                            combined_advantages[i, pos] = 0.0
                                        tokens_modified += 1
                                
                                if DEBUG and i < 2:
                                    step_type = "INCORRECT" if raw_reward < 0 else "CORRECT"
                                    action = "half-discounted" if use_half_discount else "zeroed"
                                    print(f"     Step {step_idx}: {step_type} (raw={raw_reward}), {action} {tokens_modified} tokens in range [{start_pos}:{end_pos})")
                            else:
                                if DEBUG and i < 2:
                                    step_type = "CORRECT" if raw_reward > 0 else "INCORRECT"
                                    print(f"     Step {step_idx}: {step_type} (raw={raw_reward}), kept range [{start_pos}:{end_pos})")
            
            # Update returns to match advantages
            combined_returns = combined_advantages.clone()
            
        elif use_differential_advantage:
            # PRM_COT_HYBRID_OUTCOME_DIFF_ADV: Pure 100% outcome (fallback)
            combined_advantages = outcome_advantages.clone()
            combined_returns = outcome_returns.clone()
            
            if DEBUG:
                mode_name = "PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT" if use_half_discount else "PRM_COT_HYBRID_OUTCOME_DIFF_ADV"
                print(f"\n{mode_name} DEBUG - PURE OUTCOME (NO STEP INFO)")
                print(f"   Using 100% outcome advantage for all tokens")
            
        else:
            # Regular TANGO mode: weighted combination (80% outcome + 20% step)
            combined_advantages = (1 - alpha) * outcome_advantages + alpha * step_advantages
            combined_returns = (1 - alpha) * outcome_returns + alpha * step_returns
        if DEBUG:
            if use_differential_advantage:
                mode_name = "PRM_COT_HYBRID_OUTCOME_DIFF_ADV_HALF_DISCOUNT" if use_half_discount else "PRM_COT_HYBRID_OUTCOME_DIFF_ADV"
                action_name = "half-discounting" if use_half_discount else "zeroing"
                print(f"\n{mode_name} DEBUG - FINAL RESULT")
                print(f"   Mode: Step range {action_name} applied")
                print(f"   Incorrect/correct step ranges {action_name.replace('ing', 'ed')} based on outcome sign")
            else:
                print(f"\nTANGO GRPO DEBUG - STEP 5: FINAL COMBINATION")
                print(f"   Alpha (step weight): {alpha}")
                print(f"   Outcome weight: {1-alpha}")
                print(f"   Combined = {1-alpha} * outcome + {alpha} * step")
        
        # Apply response mask to final results
        combined_advantages = combined_advantages * response_mask
        combined_returns = combined_returns * response_mask
        
        if DEBUG:
            # Show first response details
            if bsz > 0:
                valid_mask = response_mask[0]
                valid_outcome_advs = outcome_advantages[0, valid_mask]
                valid_step_advs = step_advantages[0, valid_mask]
                valid_combined_advs = combined_advantages[0, valid_mask]
                print(f"   First response (first 5 valid tokens):")
                print(f"     outcome_advantages: {valid_outcome_advs.tolist()[:5]}")
                print(f"     step_advantages: {valid_step_advs.tolist()[:5]}")
                print(f"     combined_advantages: {valid_combined_advs.tolist()[:5]}")
                print(f"   First token combined advantage: {valid_combined_advs[0].item():.6f}")
            print(f"ADVANTAGE COMPUTATION DEBUG COMPLETE\n")
        
    
    return combined_advantages, combined_returns


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (dict) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est("step_wise_grpo")  # New step-wise GRPO estimator
def compute_step_wise_grpo_advantages(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    reward_extra_info=None,
):
    """
    Compute SEMANTIC step-wise GRPO advantages by comparing the SAME reasoning step across rollouts.
    
    Key Innovation: Instead of comparing token positions, we compare Step 1 vs Step 1, Step 2 vs Step 2, etc.
    
    Example:
        3 rollouts for same prompt (index=[0,0,0]):
        
        Rollout 1: Step 1="Follow PEMDAS"(1.0), Step 2="3×4=12"(1.0), Step 3="2+12=14"(1.0)
        Rollout 2: Step 1="Left to right"(0.0), Step 2="2+3=5"(0.0), Step 3="5×4=20"(0.0)  
        Rollout 3: Step 1="Follow PEMDAS"(1.0), Step 2="2+3=5"(0.0), Step 3="5×4=20"(0.0)
        
        Semantic Comparison:
        Step 1: [1.0, 0.0, 1.0] → mean=0.67, advantages=[+0.70, -1.43, +0.70]
        Step 2: [1.0, 0.0, 0.0] → mean=0.33, advantages=[+1.43, -0.70, -0.70]
        Step 3: [1.0, 0.0, 0.0] → mean=0.33, advantages=[+1.43, -0.70, -0.70]
        
        Result: Each rollout gets step-specific feedback based on reasoning quality!
    
    Args:
        token_level_rewards: Dense reward tensor [bs, seq_len] (for backward compatibility)
        response_mask: Mask indicating valid tokens [bs, seq_len]  
        index: Group indices for GRPO comparison [bs] - responses with same index compared
        epsilon: Small constant for numerical stability
        norm_adv_by_std_in_grpo: Whether to normalize by group standard deviation
        reward_extra_info: Contains step_level_info with step rewards and boundaries
        
    Returns:
        advantages: Semantic step-wise advantages [bs, seq_len] 
        returns: Same as advantages for step-wise GRPO
    """
    bs, seq_len = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
    
    # Extract step-level information from reward manager
    if reward_extra_info is None or "step_level_info" not in reward_extra_info:
        print("[WARNING] No step_level_info found, falling back to token-position comparison")
        # Fallback to original token-position method
        return compute_step_wise_grpo_advantages_fallback(
            token_level_rewards, response_mask, index, epsilon, norm_adv_by_std_in_grpo
        )
    
    step_level_info = reward_extra_info["step_level_info"]
    step_rewards_per_rollout = step_level_info["step_rewards_per_rollout"]
    step_boundaries_per_rollout = step_level_info["step_boundaries_per_rollout"]
    
    with torch.no_grad():
        # Group rollouts by prompt index
        id2rollout_indices = defaultdict(list)
        for i in range(bs):
            prompt_id = index[i]
            id2rollout_indices[prompt_id].append(i)
        
        # Process each group of rollouts (same prompt)
        for prompt_id, rollout_indices in id2rollout_indices.items():
            if len(rollout_indices) < 2:
                # Single rollout - no comparison possible, advantages remain zero
                continue
                
            # Extract step information for this group
            group_step_rewards = []
            group_step_boundaries = []
            
            for rollout_idx in rollout_indices:
                step_rewards = step_rewards_per_rollout[rollout_idx]
                step_boundaries = step_boundaries_per_rollout[rollout_idx]
                group_step_rewards.append(step_rewards)
                group_step_boundaries.append(step_boundaries)
            
            # Find maximum number of steps in this group
            max_steps = max(len(step_rewards) for step_rewards in group_step_rewards)
            
            if max_steps == 0:
                continue  # No steps found for this group
            
            # SEMANTIC COMPARISON: Compare same step across rollouts
            for step_num in range(max_steps):
                # Collect rewards for this step number across all rollouts in group
                step_rewards_for_comparison = []
                rollout_indices_with_step = []
                
                for i, rollout_idx in enumerate(rollout_indices):
                    step_rewards = group_step_rewards[i]
                    if step_num < len(step_rewards):
                        # This rollout has this step
                        step_rewards_for_comparison.append(step_rewards[step_num])
                        rollout_indices_with_step.append(rollout_idx)
                    # If rollout doesn't have this step, skip it for this step comparison
                
                if len(step_rewards_for_comparison) < 2:
                    continue  # Need at least 2 rollouts for comparison
                
                # Apply GRPO formula to this step across rollouts
                step_tensor = torch.tensor(step_rewards_for_comparison, device=advantages.device, dtype=torch.float32)
                step_mean = torch.mean(step_tensor)
                step_std = torch.std(step_tensor)
                
                # Calculate step-level advantages
                for i, rollout_idx in enumerate(rollout_indices_with_step):
                    step_reward = step_rewards_for_comparison[i]
                    
                    if norm_adv_by_std_in_grpo:
                        step_advantage = (step_reward - step_mean) / (step_std + epsilon)
                    else:
                        step_advantage = step_reward - step_mean
                    
                    # Map step advantage to token positions for this rollout
                    step_boundaries = group_step_boundaries[rollout_indices.index(rollout_idx)]
                    if step_num < len(step_boundaries):
                        start_token, end_token = step_boundaries[step_num]
                        # Apply step advantage to all tokens in this step
                        start_token = max(0, min(start_token, seq_len))
                        end_token = max(0, min(end_token, seq_len))
                        if start_token < end_token:
                            advantages[rollout_idx, start_token:end_token] = step_advantage
            
            # Debug print for first group
            if prompt_id == list(id2rollout_indices.keys())[0]:
                print(f"[SEMANTIC STEP GRPO] Group {prompt_id}: {len(rollout_indices)} rollouts")
                for i, rollout_idx in enumerate(rollout_indices):
                    step_rewards = group_step_rewards[i]
                    print(f"  Rollout {rollout_idx}: {len(step_rewards)} steps, rewards={step_rewards}")
                print(f"  Max steps in group: {max_steps}")
    
    # Apply response mask to ensure advantages are zero for invalid positions
    advantages = advantages * response_mask
    
    return advantages, advantages


def compute_step_wise_grpo_advantages_fallback(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Fallback to token-position comparison if step-level info is not available.
    This is the original implementation for backward compatibility.
    """
    bs, seq_len = token_level_rewards.shape
    advantages = torch.zeros_like(token_level_rewards, dtype=torch.float32)
    
    with torch.no_grad():
        # For each token position, compute group-relative advantages independently
        for pos in range(seq_len):
            # Get rewards at this specific position across all sequences  
            pos_rewards = token_level_rewards[:, pos]  # [bs]
            pos_mask = response_mask[:, pos]           # [bs] 
            
            # Group rewards by prompt index (only for valid token positions)
            id2rewards = defaultdict(list)
            id2indices = defaultdict(list)
            
            for i in range(bs):
                if pos_mask[i] > 0:  # Only consider valid (non-padded) token positions
                    prompt_id = index[i]
                    id2rewards[prompt_id].append(pos_rewards[i])
                    id2indices[prompt_id].append(i)
            
            # Calculate group statistics and apply GRPO formula for this position
            for prompt_id in id2rewards:
                group_rewards = id2rewards[prompt_id]
                group_indices = id2indices[prompt_id]
                
                if len(group_rewards) == 1:
                    # Single response in group - no comparison possible, set to zero
                    group_mean = torch.tensor(0.0, device=token_level_rewards.device, dtype=torch.float32)
                    group_std = torch.tensor(1.0, device=token_level_rewards.device, dtype=torch.float32)
                elif len(group_rewards) > 1:
                    # Multiple responses - compute group statistics for this position
                    group_tensor = torch.tensor(group_rewards, device=token_level_rewards.device, dtype=torch.float32)
                    group_mean = torch.mean(group_tensor)
                    group_std = torch.std(group_tensor)
                else:
                    continue  # No valid rewards for this group at this position
                
                # Apply GRPO advantage formula: (reward - group_mean) / group_std
                for idx, batch_idx in enumerate(group_indices):
                    reward = group_rewards[idx]
                    if norm_adv_by_std_in_grpo:
                        advantages[batch_idx, pos] = (reward - group_mean) / (group_std + epsilon)
                    else:
                        advantages[batch_idx, pos] = reward - group_mean
    
    # Apply response mask to ensure advantages are zero for invalid positions
    advantages = advantages * response_mask
    
    return advantages, advantages


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config=None,
    **kwargs,
):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config=None, **kwargs
):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config=None,
    **kwargs,
):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, torch.tensor(0.0)


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    loss_agg_mode="token-mean",
    config=None,
):
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, torch.tensor(0.0), ppo_kl_abs, torch.tensor(0.0)


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
