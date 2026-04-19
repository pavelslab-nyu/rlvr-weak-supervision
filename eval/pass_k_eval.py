#!/usr/bin/env python3
"""
Calculate pass@k for all benchmarks across all checkpoint steps.

Can be used as a standalone script or imported as a module.

Usage:
    python3 pass_k_eval.py EVAL_RESULTS_DIR BENCHMARK_DIR [--benchmarks b1 b2 ...]

Example:
    python3 eval/pass_k_eval.py \\
        /path/to/eval_results_8k_t1_n16 \\
        data/eval/think \\
        --benchmarks aime2024 aime2025 amc_test math500
"""

import re
import json
import duckdb
from math import comb
from pathlib import Path
from collections import defaultdict


def find_step_folders(eval_results_dir: str) -> list:
    """Find all step_X_* folders and return sorted by step number."""
    base = Path(eval_results_dir)
    step_folders = []

    for folder in base.iterdir():
        if folder.is_dir() and folder.name.startswith('step_'):
            match = re.match(r'step_(\d+)_', folder.name)
            if match:
                step_num = int(match.group(1))
                jsonl_path = folder / "generations" / "0.jsonl"
                if jsonl_path.exists():
                    step_folders.append((step_num, jsonl_path))

    step_folders.sort(key=lambda x: x[0])
    return step_folders


def load_benchmark_prompts(parquet_path: str) -> set:
    """Load a benchmark parquet file and return the set of unique user prompts."""
    conn = duckdb.connect()
    result = conn.execute(f"SELECT * FROM '{parquet_path}'").fetchall()
    columns = [desc[0] for desc in conn.description]
    conn.close()

    prompts = set()
    for row in result:
        record = dict(zip(columns, row))
        for msg in record.get('prompt', []):
            if isinstance(msg, dict) and msg.get('role') == 'user':
                prompts.add(msg.get('content', ''))

    return prompts


def load_generations(jsonl_path: str) -> list:
    """Load all generations from a jsonl file."""
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_user_content(input_str: str) -> str:
    """
    Extract the user turn from a formatted model input string.

    Handles two chat template formats:
      - Qwen:  ...\\nuser\\n<content>\\nassistant\\n...
      - Llama: ...user\\n<content>assistant...
    """
    # Qwen format
    parts = input_str.split('\nuser\n')
    if len(parts) > 1:
        return parts[1].split('\nassistant')[0].strip()

    # Llama format
    if 'user\n' in input_str:
        parts = input_str.split('user\n', 1)
        if len(parts) > 1:
            return parts[1].split('assistant')[0].strip()

    return input_str


def match_generations_to_benchmark(generations: list, benchmark_prompts: set) -> dict:
    """Group generations by problem, keeping only those that match the benchmark."""
    problem_results = defaultdict(list)

    for gen in generations:
        user_content = extract_user_content(gen.get('input', ''))
        if user_content in benchmark_prompts:
            problem_results[user_content].append({
                'reward': gen.get('reward', 0.0),
                'ground_truth': gen.get('ground_truth', ''),
            })

    return problem_results


def pass_at_k_for_problem(n: int, c: int, k: int) -> float:
    """
    Compute pass@k for a single problem.

    pass@k = 1 - C(n-c, k) / C(n, k)

    Args:
        n: total number of samples
        c: number of correct samples
        k: the k value

    Returns:
        Probability of at least one correct answer among k samples.
    """
    if c == 0:
        return 0.0
    if c >= n or k > n or n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def calculate_pass_at_k(problem_results: dict, k_values: list = None) -> tuple:
    """Return per-problem results and average pass@k across all problems."""
    if k_values is None:
        k_values = [1, 4, 8, 16]

    results_table = []

    for idx, (problem, runs) in enumerate(problem_results.items(), 1):
        num_runs = len(runs)
        num_correct = sum(1 for r in runs if r['reward'] == 1.0)
        pass_k_scores = {
            k: pass_at_k_for_problem(num_runs, num_correct, k) if k <= num_runs else None
            for k in k_values
        }
        results_table.append({
            'problem_num': idx,
            'num_runs': num_runs,
            'num_correct': num_correct,
            'ground_truth': runs[0]['ground_truth'] if runs else '',
            'pass_k_scores': pass_k_scores,
        })

    avg_pass_k = {}
    for k in k_values:
        valid = [r['pass_k_scores'][k] for r in results_table if r['pass_k_scores'][k] is not None]
        avg_pass_k[k] = sum(valid) / len(valid) if valid else 0.0

    return results_table, len(problem_results), avg_pass_k


def process_single_step(generations: list, benchmark_prompts_cache: dict, k_values: list) -> dict:
    """Return pass@k results for all benchmarks for a single checkpoint step."""
    step_results = {}

    for benchmark, benchmark_prompts in benchmark_prompts_cache.items():
        problem_results = match_generations_to_benchmark(generations, benchmark_prompts)
        if not problem_results:
            continue
        _, _, avg_pass_k = calculate_pass_at_k(problem_results, k_values)
        step_results[benchmark] = avg_pass_k

    return step_results


def compute_passk_eval(eval_results_dir: str, benchmark_dir: str, benchmarks: list, verbose: bool = True) -> dict:
    """
    Compute pass@k for all benchmarks across all checkpoint steps.

    Args:
        eval_results_dir: Directory containing step_*/ folders with generation outputs.
        benchmark_dir: Directory containing benchmark .parquet files.
        benchmarks: List of benchmark names to evaluate.
        verbose: Print progress to stdout.

    Returns:
        {"benchmarks": {name: {step: {"pass@k": ...}}}, "overall_average": {step: {"pass@k": ...}}}
    """
    if verbose:
        print(f"Eval results: {eval_results_dir}")
        print(f"Benchmark dir: {benchmark_dir}")

    step_folders = find_step_folders(eval_results_dir)

    if verbose:
        print(f"Steps found:  {len(step_folders)}")

    if not step_folders:
        return {"benchmarks": {}, "overall_average": {}}

    k_values = [1, 4, 8, 16]

    benchmark_prompts_cache = {}
    for benchmark in benchmarks:
        parquet_path = Path(benchmark_dir) / f"{benchmark}.parquet"
        if parquet_path.exists():
            benchmark_prompts_cache[benchmark] = load_benchmark_prompts(str(parquet_path))

    if verbose:
        print(f"Benchmarks loaded: {len(benchmark_prompts_cache)}")

    all_step_results = {}
    for step_num, jsonl_path in step_folders:
        if verbose:
            print(f"  step {step_num}...")
        generations = load_generations(str(jsonl_path))
        all_step_results[step_num] = process_single_step(generations, benchmark_prompts_cache, k_values)

    json_output = {"benchmarks": {}, "overall_average": {}}

    for benchmark in benchmarks:
        if benchmark not in benchmark_prompts_cache:
            continue
        json_output["benchmarks"][benchmark] = {}
        for step_num in sorted(all_step_results):
            step_results = all_step_results[step_num]
            if benchmark in step_results:
                json_output["benchmarks"][benchmark][f"step_{step_num}"] = {
                    f"pass@{k}": round(step_results[benchmark][k] * 100, 2)
                    for k in k_values
                }

    for step_num in sorted(all_step_results):
        step_results = all_step_results[step_num]
        avg_scores = {k: [] for k in k_values}
        for results in step_results.values():
            for k in k_values:
                avg_scores[k].append(results[k])
        json_output["overall_average"][f"step_{step_num}"] = {
            f"pass@{k}": round(sum(avg_scores[k]) / len(avg_scores[k]) * 100, 2) if avg_scores[k] else 0.0
            for k in k_values
        }

    return json_output


if __name__ == '__main__':
    import argparse

    DEFAULT_BENCHMARKS = [
        "aime2024", "aime2025", "amc_test", "math500",
        "scibench_test", "scp_test_difficult_1",
    ]

    parser = argparse.ArgumentParser(description="Compute pass@k from verl evaluation results.")
    parser.add_argument("eval_results_dir", help="Directory containing step_*/ folders with generation outputs.")
    parser.add_argument("benchmark_dir", help="Directory containing benchmark .parquet files (e.g. data/eval/think).")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=DEFAULT_BENCHMARKS,
        metavar="BENCHMARK",
        help="Benchmark names to evaluate (default: %(default)s).",
    )
    args = parser.parse_args()

    results = compute_passk_eval(args.eval_results_dir, args.benchmark_dir, args.benchmarks)
    print(json.dumps(results, indent=2))
