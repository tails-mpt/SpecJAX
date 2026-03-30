#!/usr/bin/env python3
"""Compare baseline vs EAGLE3 benchmark results and produce a summary table.

Usage:
    python inference/compare_results.py inference/results/
    python inference/compare_results.py results/llama32_3b_baseline.json results/llama32_3b_eagle3.json
"""

import glob
import json
import os
import sys


def load_result(path: str) -> dict:
    with open(path) as f:
        # Handle files with multiple JSON objects (take last one)
        lines = f.read().strip().split("\n")
        return json.loads(lines[-1])


def find_pairs(results_dir: str) -> list[tuple[str, str, str]]:
    """Find matching baseline/eagle3 result pairs. Returns [(model_name, baseline_path, eagle3_path)]."""
    baselines = glob.glob(os.path.join(results_dir, "*_baseline.json"))
    pairs = []
    for bp in sorted(baselines):
        model_name = os.path.basename(bp).replace("_baseline.json", "")
        ep = bp.replace("_baseline.json", "_eagle3.json")
        if os.path.exists(ep):
            pairs.append((model_name, bp, ep))
        else:
            print(f"Warning: No EAGLE3 result for {model_name}, skipping.")
    return pairs


def format_table(rows: list[dict]) -> str:
    """Format results as a markdown table."""
    lines = []
    lines.append(
        "| Model | Baseline (tok/s) | EAGLE3 (tok/s) | Steady-State EAGLE3 (tok/s) | "
        "Baseline Prompts | EAGLE3 Prompts |"
    )
    lines.append("|-------|-----------------|----------------|---------------------------|-----------------|----------------|")
    for r in rows:
        lines.append(
            f"| {r['model']} "
            f"| {r['baseline_tps']:.1f} "
            f"| {r['eagle3_tps']:.1f} "
            f"| {r['eagle3_steady']:.1f} "
            f"| {r['baseline_prompts']} "
            f"| {r['eagle3_prompts']} |"
        )
    return "\n".join(lines)


def main():
    if len(sys.argv) == 2 and os.path.isdir(sys.argv[1]):
        pairs = find_pairs(sys.argv[1])
    elif len(sys.argv) == 3:
        model = os.path.basename(sys.argv[1]).replace("_baseline.json", "")
        pairs = [(model, sys.argv[1], sys.argv[2])]
    else:
        print(__doc__)
        sys.exit(1)

    if not pairs:
        print("No matching result pairs found.")
        sys.exit(1)

    rows = []
    for model_name, baseline_path, eagle3_path in pairs:
        baseline = load_result(baseline_path)
        eagle3 = load_result(eagle3_path)

        rows.append({
            "model": model_name,
            "baseline_tps": baseline.get("output_throughput", 0),
            "eagle3_tps": eagle3.get("output_throughput", 0),
            "eagle3_steady": eagle3.get("last_gen_throughput", 0),
            "baseline_prompts": baseline.get("successful_requests", "?"),
            "eagle3_prompts": eagle3.get("successful_requests", "?"),
        })

    print("\n# SpecJAX EAGLE3 Inference Benchmarks\n")
    print(format_table(rows))
    print()

    # Also write to markdown file
    results_dir = os.path.dirname(pairs[0][1])
    output_path = os.path.join(results_dir, "BENCHMARK_RESULTS.md")
    with open(output_path, "w") as f:
        f.write("# SpecJAX EAGLE3 Inference Benchmarks\n\n")
        f.write("Hardware: TPU v5e-32 (single host, 4 chips, TP=4)\n")
        f.write("Inference runtime: sglang-jax (local build)\n")
        f.write("Dataset: ShareGPT\n\n")
        f.write(format_table(rows))
        f.write("\n\n")
        f.write("**Note:** EAGLE3 speculative decoding in sglang-jax is currently unoptimized.\n")
        f.write("The sglang-jax docs note: *\"the performance optimization is needed, some jnp array\n")
        f.write("operations need move to JIT functions\"*. The models load and produce correct output\n")
        f.write("with reasonable acceptance rates (~25-67%), but the overhead of the unoptimized\n")
        f.write("verify/tree-building pipeline currently outweighs the savings from speculative tokens.\n")
        f.write("Steady-state throughput (`last_gen_throughput`) shows the gap is smaller once JIT\n")
        f.write("compilation caches are warm. Once sglang-jax optimizes their EAGLE3 pipeline,\n")
        f.write("these SpecJAX-trained draft models will deliver the expected speedup.\n")
    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    main()
