"""Run sync, safe async and fully async swap across pressure levels."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


BENCHMARK = Path(__file__).with_name("benchmark.py")
MODES = {
    "sync": ["--mode", "sync"],
    "safe_async": ["--mode", "async"],
    "full_async": ["--mode", "async", "--unsafe-async-swap-out"],
}
ORDERS = [
    ("sync", "safe_async", "full_async"),
    ("full_async", "safe_async", "sync"),
    ("safe_async", "sync", "full_async"),
]
METRICS = (
    "throughput_tokens_per_s",
    "ttft_p95_ms",
    "tpot_p95_ms",
    "swap_in_blocks",
    "swap_out_blocks",
    "pending_only_steps",
    "scheduler_steps",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--gpu-blocks", default="32,24,20")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--cpu-swap-space-gb", type=float, default=4.0)
    parser.add_argument("--max-num-seqs", type=int, default=12)
    parser.add_argument("--max-model-len", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_once(args, blocks, repetition, mode):
    run_dir = args.results_dir / f"blocks-{blocks}" / f"repeat-{repetition}"
    output = run_dir / f"{mode}.json"
    command = [
        sys.executable,
        str(BENCHMARK),
        "--model", args.model,
        *MODES[mode],
        "--num-requests", str(args.num_requests),
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--num-gpu-blocks", str(blocks),
        "--cpu-swap-space-gb", str(args.cpu_swap_space_gb),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-model-len", str(args.max_model_len),
        "--seed", str(args.seed),
        "--output", str(output),
    ]
    print(
        f"[blocks={blocks} repeat={repetition + 1}/{args.repetitions}] {mode}",
        flush=True,
    )
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        print(completed.stdout, file=sys.stderr)
        print(completed.stderr, file=sys.stderr)
        completed.check_returncode()
    return json.loads(output.read_text(encoding="utf-8"))


def mean_and_stdev(values):
    return {
        "mean": round(statistics.mean(values), 6),
        "stdev": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
    }


def aggregate_mode(runs, sync_runs):
    aggregate = {
        metric: mean_and_stdev([run[metric] for run in runs])
        for metric in METRICS
    }
    matching = [
        sum(
            left == right
            for left, right in zip(
                sync["request_output_token_ids"],
                run["request_output_token_ids"],
            )
        )
        for run, sync in zip(runs, sync_runs)
    ]
    aggregate["matching_requests"] = mean_and_stdev(matching)
    aggregate["total_requests"] = runs[0]["num_requests"]
    return aggregate


def percent_change(value, baseline):
    return round((value / baseline - 1) * 100, 3)


def write_markdown(summary, destination):
    lines = [
        "# 三种 Swap 方案实验汇总",
        "",
        "每个单元格为多次独立进程运行的均值；括号内为标准差。",
        "",
        "| GPU blocks | 方案 | 吞吐 tok/s | TTFT P95 ms | TPOT P95 ms | 输出匹配 |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for blocks, pressure in summary["pressures"].items():
        for mode in MODES:
            values = pressure["modes"][mode]
            lines.append(
                f"| {blocks} | {mode} | "
                f"{values['throughput_tokens_per_s']['mean']:.3f} "
                f"({values['throughput_tokens_per_s']['stdev']:.3f}) | "
                f"{values['ttft_p95_ms']['mean']:.3f} "
                f"({values['ttft_p95_ms']['stdev']:.3f}) | "
                f"{values['tpot_p95_ms']['mean']:.3f} "
                f"({values['tpot_p95_ms']['stdev']:.3f}) | "
                f"{values['matching_requests']['mean']:.1f}/"
                f"{values['total_requests']} |"
            )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    blocks_list = [int(value) for value in args.gpu_blocks.split(",")]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": {
            "gpu_blocks": blocks_list,
            "repetitions": args.repetitions,
            "num_requests": args.num_requests,
            "input_len": args.input_len,
            "output_len": args.output_len,
            "seed": args.seed,
        },
        "pressures": {},
    }
    for blocks in blocks_list:
        runs = {mode: [] for mode in MODES}
        for repetition in range(args.repetitions):
            for mode in ORDERS[repetition % len(ORDERS)]:
                runs[mode].append(run_once(args, blocks, repetition, mode))
        sync_runs = runs["sync"]
        modes = {
            mode: aggregate_mode(mode_runs, sync_runs)
            for mode, mode_runs in runs.items()
        }
        baseline = modes["sync"]
        for mode in ("safe_async", "full_async"):
            modes[mode]["vs_sync_percent"] = {
                "throughput": percent_change(
                    modes[mode]["throughput_tokens_per_s"]["mean"],
                    baseline["throughput_tokens_per_s"]["mean"],
                ),
                "ttft_p95": percent_change(
                    modes[mode]["ttft_p95_ms"]["mean"],
                    baseline["ttft_p95_ms"]["mean"],
                ),
                "tpot_p95": percent_change(
                    modes[mode]["tpot_p95_ms"]["mean"],
                    baseline["tpot_p95_ms"]["mean"],
                ),
            }
        summary["pressures"][str(blocks)] = {"modes": modes}
    (args.results_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(summary, args.results_dir / "summary.md")
    print(json.dumps(summary["pressures"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
