from reasoning_gym import get_score_answer_fn
import json
from ast import literal_eval
import numpy as np
from verl.utils.reward_score import math_verify

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def extract_answer_reasoning_gym(model_output: str) -> str:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    solution = last_boxed_only_string(model_output)
    solution = remove_boxed(solution)
    return solution


import json
from ast import literal_eval

def _safe_parse_one(s):
    # Try JSON first
    try: return json.loads(s)
    except Exception: pass
    # Then Python literal (handles single quotes, True/False/None, tuples, etc.)
    try: return literal_eval(s)
    except Exception: return s  # leave as-is if not parseable

def recover_types(obj):
    if isinstance(obj, str):
        return _safe_parse_one(obj)
    if isinstance(obj, list):
        return [_safe_parse_one(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _safe_parse_one(v) for k, v in obj.items()}
    return obj

def compute_score(solution_str, ground_truth, extra_info=None):
    if isinstance(ground_truth, list) and len(ground_truth) > 0:
        processed_ground_truth = str(ground_truth[0])
    else:
        processed_ground_truth = str(ground_truth)

    if extra_info is not None:
        if isinstance(extra_info, str):
            extra_info = parse_to_dict(extra_info)
        extra_info = recover_types(extra_info)
        source_dataset = extra_info.get('source_dataset', None)
    else:
        source_dataset = None  # Default fallback

    if source_dataset is None:
       score = math_verify.compute_score(solution_str, processed_ground_truth)
    else:
        score_fn = get_score_answer_fn(source_dataset)
    
        entry = {'metadata':extra_info,'answer':ground_truth}
        extracted_answer = extract_answer_reasoning_gym(solution_str)
        if not isinstance(extracted_answer, str):
            extracted_answer = str(extracted_answer)
        if source_dataset in ['shortest_path','self_reference','aiw']: # for numerical answers, we use the math_verify verifier
            formatted_answer = "\\boxed{" + extracted_answer + "}"
            score = math_verify.compute_score(formatted_answer, processed_ground_truth)
        else:
            if source_dataset in ['course_schedule','syllogism']: # for single string answers, we use lower case comparison
                entry['answer'] = entry['answer'].strip().lower()
                extracted_answer = extracted_answer.lower()
            if source_dataset == 'graph_color':
                if not extracted_answer.startswith('{'):
                    extracted_answer = "{"
                elif not extracted_answer.endswith('}'):
                    extracted_answer = extracted_answer + "}"
            score = score_fn(extracted_answer, entry)
            score = 1.0 if float(score) == 1.0 else 0.0
    return score