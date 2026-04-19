# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import os
import re
import random

DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

if DEBUG:
    print(f"REWARD_MODEL_TYPE: {os.environ.get('REWARD_MODEL_TYPE', 'NOT_SET')}")

REWARD_MODEL_TYPE = os.environ['REWARD_MODEL_TYPE'].upper()


# =============================================================================
# FORMAT VALIDATION
# =============================================================================

def format_validity(solution_str):
    """Returns 1.0 if the solution contains exactly one \\boxed{} expression, else 0.0."""
    try:
        boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']
        total_boxed = sum(len(re.findall(pattern, solution_str)) for pattern in boxed_patterns)
        return 1.0 if total_boxed == 1 else 0.0
    except Exception:
        return 0.0


def thinking_format_validity(solution_str):
    """
    Returns 1.0 if the solution contains <think>...</think> tags and at least one
    \\boxed{} after </think>, else 0.0.
    """
    try:
        if '<think>' not in solution_str or '</think>' not in solution_str:
            if DEBUG:
                print("THINKING_FORMAT INVALID: Missing <think>...</think> tags")
            return 0.0

        think_end_pos = solution_str.find('</think>')
        content_after_think = solution_str[think_end_pos + len('</think>'):]

        boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']
        total_boxed = sum(len(re.findall(pattern, content_after_think)) for pattern in boxed_patterns)

        if total_boxed == 0:
            if DEBUG:
                print("THINKING_FORMAT INVALID: No \\boxed{} found after </think>")
            return 0.0

        if DEBUG:
            print(f"THINKING_FORMAT VALID: Found <think>...</think> tags and {total_boxed} \\boxed{{}} after </think>")

        return 1.0

    except Exception as e:
        if DEBUG:
            print(f"THINKING_FORMAT ERROR: {e}")
        return 0.0


def deepseek_thinking_format_validity(solution_str):
    """
    Returns 1.0 if the solution contains </think> (the opening tag is in the prompt)
    and at least one \\boxed{} after </think>, else 0.0.
    """
    try:
        # <think> is provided in the prompt, so only </think> is checked here
        if '</think>' not in solution_str:
            if DEBUG:
                print("DEEPSEEK_THINKING_FORMAT INVALID: Missing </think> tag")
            return 0.0

        think_end_pos = solution_str.find('</think>')
        content_after_think = solution_str[think_end_pos + len('</think>'):]

        boxed_patterns = [r'\\\\boxed\{.*?\}', r'\\boxed\{.*?\}']
        total_boxed = sum(len(re.findall(pattern, content_after_think)) for pattern in boxed_patterns)

        if total_boxed == 0:
            if DEBUG:
                print("DEEPSEEK_THINKING_FORMAT INVALID: No \\boxed{} found after </think>")
            return 0.0

        if DEBUG:
            print(f"DEEPSEEK_THINKING_FORMAT VALID: Found </think> tag and {total_boxed} \\boxed{{}} after </think>")

        return 1.0

    except Exception as e:
        if DEBUG:
            print(f"DEEPSEEK_THINKING_FORMAT ERROR: {e}")
        return 0.0


def find_balanced_boxed(text):
    """
    Returns a list of (start, end) positions for all \\boxed{...} expressions,
    matching braces correctly.
    """
    boxed_positions = []
    patterns = [r'\\\\boxed\{', r'\\boxed\{']

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            start = match.start()
            brace_start = match.end() - 1

            brace_count = 1
            pos = brace_start + 1

            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1

            if brace_count == 0:
                boxed_positions.append((start, pos))

    return boxed_positions


def rlvr_format_validity(solution_str):
    """
    Returns 1.0 if the solution has exactly one \\boxed{} expression with no trailing
    content, else 0.0.
    """
    try:
        boxed_positions = find_balanced_boxed(solution_str)

        if len(boxed_positions) != 1:
            if DEBUG:
                print(f"RLVR_FORMAT INVALID: Found {len(boxed_positions)} boxed expressions (expected 1)")
            return 0.0

        start, end = boxed_positions[0]
        content_after_boxed = solution_str[end:].strip()

        if content_after_boxed:
            if DEBUG:
                preview = content_after_boxed[:200] + "..." if len(content_after_boxed) > 200 else content_after_boxed
                print(f"RLVR_FORMAT INVALID: Extra content after boxed answer: '{preview}'")
            return 0.0

        if DEBUG:
            print(f"RLVR_FORMAT VALID: '{solution_str[start:end]}'")
        return 1.0

    except Exception as e:
        if DEBUG:
            print(f"RLVR_FORMAT ERROR: {e}")
        return 0.0


# =============================================================================
# RANDOM REWARD SCORING
# =============================================================================

def compute_random_reward_score(solution_str, ground_truth):
    """Returns 0.0 if format is invalid, otherwise a random 0.0 or 1.0."""
    try:
        if format_validity(solution_str) == 0.0:
            if DEBUG:
                print("RANDOM_REWARD: Format invalid -> 0.0")
            return 0.0

        random_score = random.choice([0.0, 1.0])

        if DEBUG:
            print(f"RANDOM_REWARD: Format valid -> {random_score}")
            print(f"   Ground truth: {ground_truth} (ignored)")

        return random_score

    except Exception as e:
        if DEBUG:
            print(f"RANDOM_REWARD ERROR: {e}")
        return 0.0


# =============================================================================
# RULE-BASED SCORING
# =============================================================================

def compute_rule_based_score(solution_str, ground_truth):
    """Score a solution against ground truth using math_verify."""
    try:
        from verl.utils.reward_score import math_verify

        if isinstance(ground_truth, list) and len(ground_truth) > 0:
            processed_ground_truth = str(ground_truth[0])
        else:
            processed_ground_truth = str(ground_truth)

        score = math_verify.compute_score(solution_str, processed_ground_truth)

        if DEBUG:
            print("\n" + "="*80)
            print("RULE_BASED DEBUG")
            print("="*80)
            print(f"RESPONSE:\n{solution_str}")
            print(f"\nGROUND TRUTH: {ground_truth}")
            print(f"\nREWARD: {score}")
            print("="*80 + "\n")

        return score
    except Exception as e:
        if DEBUG:
            print(f"RULE_BASED ERROR: {e}")
        return 0.0


def compute_verifier_based_score(solution_str, ground_truth, extra_info):
    """Score a solution using the reasoning-gym verifier (graph domain)."""
    from verl.utils.reward_score.reasoning_gym import compute_score
    return compute_score(solution_str, ground_truth, extra_info)


def _process_ground_truth(ground_truth):
    """Convert ground truth to string, unwrapping single-element lists."""
    if isinstance(ground_truth, list) and len(ground_truth) > 0:
        return str(ground_truth[0])
    return str(ground_truth)


# =============================================================================
# MAIN SCORING FUNCTIONS
# =============================================================================

def compute_score(data_source, solution_str, ground_truth, extra_info):
    """
    Per-sample scoring. Dispatches on REWARD_MODEL_TYPE.

    Supported types: RULE_BASED, RULE_BASED_THINKING_FORMAT,
    DEEPSEEK_RULE_BASED_THINKING_FORMAT, RLVR_FORMAT, RANDOM_REWARD,
    MAJORITY_VOTE_FORMAT_PENALTY, VERIFIER_BASED.
    """
    if REWARD_MODEL_TYPE == 'DEEPSEEK_RULE_BASED_THINKING_FORMAT':
        if deepseek_thinking_format_validity(solution_str) == 0.0:
            if DEBUG:
                print(f"\nDEEPSEEK_RULE_BASED_THINKING_FORMAT - FORMAT PENALTY\nRESPONSE:\n{solution_str}\nGROUND TRUTH: {ground_truth}\nREWARD: 0.0\n")
            return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"}

        reward_score = compute_rule_based_score(solution_str, _process_ground_truth(ground_truth))

        if DEBUG:
            print(f"\nDEEPSEEK_RULE_BASED_THINKING_FORMAT - VALID\nRESPONSE:\n{solution_str}\nGROUND TRUTH: {ground_truth}\nREWARD: {reward_score}\n")

        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"}

    if REWARD_MODEL_TYPE == 'RULE_BASED_THINKING_FORMAT':
        if thinking_format_validity(solution_str) == 0.0:
            if DEBUG:
                print(f"\nRULE_BASED_THINKING_FORMAT - FORMAT PENALTY\nRESPONSE:\n{solution_str}\nGROUND TRUTH: {ground_truth}\nREWARD: 0.0\n")
            return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"}

        reward_score = compute_rule_based_score(solution_str, _process_ground_truth(ground_truth))

        if DEBUG:
            print(f"\nRULE_BASED_THINKING_FORMAT - VALID\nRESPONSE:\n{solution_str}\nGROUND TRUTH: {ground_truth}\nREWARD: {reward_score}\n")

        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"}

    if REWARD_MODEL_TYPE == 'RLVR_FORMAT':
        if rlvr_format_validity(solution_str) == 0.0:
            if DEBUG:
                print(f"\nRLVR_FORMAT - FORMAT PENALTY\nRESPONSE:\n{solution_str}\nGROUND TRUTH: {ground_truth}\nREWARD: 0.0\n")
            return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"}

        reward_score = compute_rule_based_score(solution_str, _process_ground_truth(ground_truth))

        if DEBUG:
            print(f"\nRLVR_FORMAT - VALID\nRESPONSE:\n{solution_str}\nGROUND TRUTH: {ground_truth}\nREWARD: {reward_score}\n")

        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"}

    if REWARD_MODEL_TYPE == 'RULE_BASED':
        reward_score = compute_rule_based_score(solution_str, _process_ground_truth(ground_truth))
        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RULE_BASED"}

    if 'VERIFIER_BASED' in REWARD_MODEL_TYPE:
        processed_gt = _process_ground_truth(ground_truth)
        reward_score = compute_verifier_based_score(solution_str, processed_gt, extra_info)
        return {"score": reward_score, "ground_truth": processed_gt, "reward_method": "VERIFIER_BASED"}

    if REWARD_MODEL_TYPE == 'RANDOM_REWARD':
        processed_gt = _process_ground_truth(ground_truth)
        reward_score = compute_random_reward_score(solution_str, processed_gt)

        if DEBUG:
            print(f"\nRANDOM_REWARD\nRESPONSE:\n{solution_str}\nGROUND TRUTH: {processed_gt} (ignored)\nREWARD: {reward_score}\n")

        return {"score": reward_score, "ground_truth": ground_truth, "reward_method": "RANDOM_REWARD"}

    print(f"WARNING: Unknown REWARD_MODEL_TYPE: {REWARD_MODEL_TYPE}")
    return {"score": 0.0, "ground_truth": ground_truth, "reward_method": "UNKNOWN"}


# =============================================================================
# BATCH SCORING FUNCTION
# =============================================================================

def is_validation_data(data_sources):
    """Return True if the batch appears to be validation data based on source name patterns."""
    validation_patterns = [
        'test-math-aime24', 'test-math-aime25', 'huggingfaceh4/math-500',
        'test-math-', 'test-amc', 'test-scp', 'test-graph-',
        'stem__gpqa', 'stem__supergpqa',
        'logic__arcagi', 'logic__zebra_puzzle',
        'mmlu_sci', 'stem__mmlu',
    ]
    return any(
        any(pattern in str(ds).lower() for pattern in validation_patterns)
        for ds in data_sources
    )


def compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, reward_method_name="VALIDATION"):
    """
    Score a validation batch using math_verify (or the reasoning-gym verifier for
    VERIFIER_BASED types). For MAJORITY_VOTE types, the original ground truth from
    extra_info is used instead of the majority-vote label.
    """
    from verl.utils.reward_score import math_verify

    results = []
    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):
        # Use original ground truth for majority-vote validation so metrics reflect true accuracy
        if "MAJORITY_VOTE_VALIDATION" in reward_method_name and 'original_ground_truth' in extra_info:
            processed_gt = extra_info['original_ground_truth']
        else:
            processed_gt = ground_truth

        processed_ground_truth = _process_ground_truth(processed_gt)

        if 'VERIFIER_BASED' in REWARD_MODEL_TYPE:
            score = compute_verifier_based_score(solution_str, processed_ground_truth, extra_info)
            if "MAJORITY_VOTE" in reward_method_name:
                results.append({
                    "score": score,
                    "majority_vote_gt": ground_truth,
                    "original_gt": processed_ground_truth,
                    "ground_truth": processed_ground_truth,
                    "reward_method": reward_method_name,
                })
            else:
                results.append({"score": score, "ground_truth": processed_ground_truth, "reward_method": reward_method_name})

        else:
            score = math_verify.compute_score(solution_str, processed_ground_truth)
            if "MAJORITY_VOTE" in reward_method_name:
                results.append({
                    "score": score,
                    "majority_vote_gt": ground_truth,
                    "original_gt": processed_gt,
                    "ground_truth": processed_gt,
                    "reward_method": reward_method_name,
                })
            else:
                results.append({"score": score, "ground_truth": processed_gt, "reward_method": reward_method_name})

    if DEBUG and results:
        correct_count = sum(1 for r in results if r["score"] > 0.5)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"\n{reward_method_name} SUMMARY: {correct_count}/{total_count} = {accuracy:.3f}\n")

    return results


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Batch scoring function. Dispatches on REWARD_MODEL_TYPE.

    Supported types: RULE_BASED, RULE_BASED_THINKING_FORMAT,
    DEEPSEEK_RULE_BASED_THINKING_FORMAT, RLVR_FORMAT, RANDOM_REWARD,
    MAJORITY_VOTE, MAJORITY_VOTE_FORMAT_PENALTY, SELF_CERTAINTY,
    VERIFIER_BASED, VERIFIER_BASED_MAJORITY_VOTE.

    For proxy reward types (MAJORITY_VOTE, SELF_CERTAINTY, RANDOM_REWARD),
    validation batches fall back to rule-based scoring.
    """
    if DEBUG:
        print(f"BATCH: {len(data_sources)} samples, REWARD_MODEL_TYPE={REWARD_MODEL_TYPE}")

    if REWARD_MODEL_TYPE == 'DEEPSEEK_RULE_BASED_THINKING_FORMAT':
        if is_validation_data(data_sources):
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "DEEPSEEK_RULE_BASED_THINKING_FORMAT_VALIDATION")

        from verl.utils.reward_score import math_verify
        results = []
        format_penalty_count = 0

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            if deepseek_thinking_format_validity(solution_str) == 0.0:
                results.append({"score": 0.0, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"})
                format_penalty_count += 1
            else:
                score = math_verify.compute_score(solution_str, _process_ground_truth(ground_truth))
                results.append({"score": score, "ground_truth": ground_truth, "reward_method": "DEEPSEEK_RULE_BASED_THINKING_FORMAT"})

        if DEBUG and results:
            total_count = len(results)
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            format_valid_count = total_count - format_penalty_count
            print(f"\nDEEPSEEK_RULE_BASED_THINKING_FORMAT TRAINING SUMMARY:")
            print(f"   Format valid: {format_valid_count}/{total_count} = {format_valid_count/total_count:.3f}")
            print(f"   Format penalties: {format_penalty_count}/{total_count} = {format_penalty_count/total_count:.3f}")
            print(f"   Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.3f}\n")

        return results

    if REWARD_MODEL_TYPE == 'RULE_BASED_THINKING_FORMAT':
        if is_validation_data(data_sources):
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "RULE_BASED_THINKING_FORMAT_VALIDATION")

        from verl.utils.reward_score import math_verify
        results = []
        format_penalty_count = 0

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            if thinking_format_validity(solution_str) == 0.0:
                results.append({"score": 0.0, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"})
                format_penalty_count += 1
            else:
                score = math_verify.compute_score(solution_str, _process_ground_truth(ground_truth))
                results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RULE_BASED_THINKING_FORMAT"})

        if DEBUG and results:
            total_count = len(results)
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            format_valid_count = total_count - format_penalty_count
            print(f"\nRULE_BASED_THINKING_FORMAT TRAINING SUMMARY:")
            print(f"   Format valid: {format_valid_count}/{total_count} = {format_valid_count/total_count:.3f}")
            print(f"   Format penalties: {format_penalty_count}/{total_count} = {format_penalty_count/total_count:.3f}")
            print(f"   Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.3f}\n")

        return results

    if REWARD_MODEL_TYPE == 'SELF_CERTAINTY':
        if is_validation_data(data_sources):
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "SELF_CERTAINTY_VALIDATION")

        # Scores are placeholders; actual self-certainty rewards are applied in ray_trainer.py
        results = [
            {"score": 0.0, "ground_truth": gt, "reward_method": "SELF_CERTAINTY"}
            for gt in ground_truths
        ]

        if DEBUG:
            print(f"\nSELF_CERTAINTY TRAINING: {len(results)} samples marked for processing in ray_trainer.py\n")

        return results

    if REWARD_MODEL_TYPE == 'MAJORITY_VOTE_FORMAT_PENALTY':
        if is_validation_data(data_sources):
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_FORMAT_PENALTY_VALIDATION")

        from verl.utils.reward_score import math_verify
        results = []
        format_penalty_count = 0

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            original_gt = extra_info.get('original_ground_truth', None)
            if isinstance(original_gt, list) and len(original_gt) > 0:
                original_gt = str(original_gt[0])
            elif original_gt is not None:
                original_gt = str(original_gt)

            if thinking_format_validity(solution_str) == 0.0:
                results.append({
                    "score": 0.0,
                    "majority_vote_gt": ground_truth,
                    "original_gt": original_gt,
                    "ground_truth": ground_truth,
                    "reward_method": "MAJORITY_VOTE_FORMAT_PENALTY",
                })
                format_penalty_count += 1
            else:
                score = math_verify.compute_score(solution_str, _process_ground_truth(ground_truth))
                results.append({
                    "score": score,
                    "majority_vote_gt": ground_truth,
                    "original_gt": original_gt,
                    "ground_truth": ground_truth,
                    "reward_method": "MAJORITY_VOTE_FORMAT_PENALTY",
                })

        if DEBUG and results:
            total_count = len(results)
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            format_valid_count = total_count - format_penalty_count
            print(f"\nMAJORITY_VOTE_FORMAT_PENALTY TRAINING SUMMARY:")
            print(f"   Format valid: {format_valid_count}/{total_count} = {format_valid_count/total_count:.3f}")
            print(f"   Format penalties: {format_penalty_count}/{total_count} = {format_penalty_count/total_count:.3f}")
            print(f"   Accuracy vs. majority vote GT: {correct_count}/{total_count} = {correct_count/total_count:.3f}\n")

        return results

    if REWARD_MODEL_TYPE == 'MAJORITY_VOTE':
        if is_validation_data(data_sources):
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_VALIDATION")

        from verl.utils.reward_score import math_verify
        results = []

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            score = math_verify.compute_score(solution_str, _process_ground_truth(ground_truth))

            original_gt = extra_info.get('original_ground_truth', None)
            if isinstance(original_gt, list) and len(original_gt) > 0:
                original_gt = str(original_gt[0])
            elif original_gt is not None:
                original_gt = str(original_gt)

            results.append({
                "score": score,
                "majority_vote_gt": ground_truth,
                "original_gt": original_gt,
                "ground_truth": ground_truth,
                "reward_method": "MAJORITY_VOTE",
            })

        if DEBUG and results:
            total_count = len(results)
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            original_gts = [r.get('original_gt') for r in results if r.get('original_gt') is not None]
            majority_gts = [r.get('majority_vote_gt') for r in results]

            print(f"\nMAJORITY_VOTE TRAINING SUMMARY:")
            print(f"   Accuracy vs. majority vote GT: {correct_count}/{total_count} = {correct_count/total_count:.3f}")
            if original_gts and len(original_gts) == len(majority_gts):
                matches = sum(1 for orig, maj in zip(original_gts, majority_gts) if str(orig) == str(maj))
                print(f"   Majority vote vs. original GT agreement: {matches}/{len(original_gts)} = {matches/len(original_gts):.3f}")
            print()

        return results

    if REWARD_MODEL_TYPE == 'RLVR_FORMAT':
        from verl.utils.reward_score import math_verify
        results = []
        format_penalty_count = 0

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            if rlvr_format_validity(solution_str) == 0.0:
                results.append({"score": 0.0, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"})
                format_penalty_count += 1
            else:
                score = math_verify.compute_score(solution_str, _process_ground_truth(ground_truth))
                results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RLVR_FORMAT"})

        if DEBUG and results:
            total_count = len(results)
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            format_valid_count = total_count - format_penalty_count
            print(f"\nRLVR_FORMAT BATCH SUMMARY:")
            print(f"   Format valid: {format_valid_count}/{total_count} = {format_valid_count/total_count:.3f}")
            print(f"   Format penalties: {format_penalty_count}/{total_count} = {format_penalty_count/total_count:.3f}")
            print(f"   Accuracy: {correct_count}/{total_count} = {correct_count/total_count:.3f}\n")

        return results

    if REWARD_MODEL_TYPE == 'RULE_BASED':
        from verl.utils.reward_score import math_verify
        results = []

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            score = math_verify.compute_score(solution_str, _process_ground_truth(ground_truth))
            results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RULE_BASED"})

        return results

    if REWARD_MODEL_TYPE == 'VERIFIER_BASED':
        results = []

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            processed_gt = _process_ground_truth(ground_truth)
            score = compute_verifier_based_score(solution_str, processed_gt, extra_info)
            results.append({"score": score, "ground_truth": processed_gt, "reward_method": "VERIFIER_BASED"})

        return results

    if REWARD_MODEL_TYPE == 'VERIFIER_BASED_MAJORITY_VOTE':
        if is_validation_data(data_sources):
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "MAJORITY_VOTE_VALIDATION")

        results = []

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            processed_gt = _process_ground_truth(ground_truth)
            score = compute_verifier_based_score(solution_str, processed_gt, extra_info)

            original_gt = extra_info.get('original_ground_truth', None)
            if isinstance(original_gt, list) and len(original_gt) > 0:
                original_gt = str(original_gt[0])
            elif original_gt is not None:
                original_gt = str(original_gt)

            results.append({
                "score": score,
                "majority_vote_gt": ground_truth,
                "original_gt": original_gt,
                "ground_truth": ground_truth,
                "reward_method": "MAJORITY_VOTE",
            })

        if DEBUG and results:
            total_count = len(results)
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            original_gts = [r.get('original_gt') for r in results if r.get('original_gt') is not None]
            majority_gts = [r.get('majority_vote_gt') for r in results]

            print(f"\nVERIFIER_BASED_MAJORITY_VOTE TRAINING SUMMARY:")
            print(f"   Accuracy vs. majority vote GT: {correct_count}/{total_count} = {correct_count/total_count:.3f}")
            if original_gts and len(original_gts) == len(majority_gts):
                matches = sum(1 for orig, maj in zip(original_gts, majority_gts) if str(orig) == str(maj))
                print(f"   Majority vote vs. original GT agreement: {matches}/{len(original_gts)} = {matches/len(original_gts):.3f}")
            print()

        return results

    if REWARD_MODEL_TYPE == 'RANDOM_REWARD':
        if is_validation_data(data_sources):
            return compute_validation_scores(data_sources, solution_strs, ground_truths, extra_infos, "RANDOM_REWARD_VALIDATION")

        results = []

        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            processed_gt = _process_ground_truth(ground_truth)
            score = compute_random_reward_score(solution_str, processed_gt)
            results.append({"score": score, "ground_truth": ground_truth, "reward_method": "RANDOM_REWARD"})

        if DEBUG and results:
            total_count = len(results)
            correct_count = sum(1 for r in results if r["score"] > 0.5)
            print(f"\nRANDOM_REWARD TRAINING: {correct_count}/{total_count} = {correct_count/total_count:.3f} (expected ~0.5)\n")

        return results

    print(f"WARNING: Unknown REWARD_MODEL_TYPE in batch: {REWARD_MODEL_TYPE}")
    return [{"score": 0.0, "ground_truth": gt, "reward_method": "UNKNOWN"} for gt in ground_truths]
