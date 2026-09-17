"""Run sync and async benchmarks in isolated processes."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).with_name("benchmark.py")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--num-requests", type=int, default=48)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--num-gpu-blocks", type=int)
    parser.add_argument("--cpu-swap-space-gb", type=float, default=8.0)
    parser.add_argument("--max-num-seqs", type=int, default=48)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def run_mode(args, mode):
    output = args.results_dir / f"{mode}.json"
    command = [
        sys.executable,
        str(SCRIPT),
        "--model", args.model,
        "--mode", mode,
        "--num-requests", str(args.num_requests),
        "--input-len", str(args.input_len),
        "--output-len", str(args.output_len),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--cpu-swap-space-gb", str(args.cpu_swap_space_gb),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-model-len", str(args.max_model_len),
        "--seed", str(args.seed),
        "--output", str(output),
    ]
    if args.num_gpu_blocks is not None:
        command.extend(["--num-gpu-blocks", str(args.num_gpu_blocks)])
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    sync = run_mode(args, "sync")
    asynchronous = run_mode(args, "async")
    comparison = {
        "sync": sync,
        "async": asynchronous,
        "async_vs_sync": {
            "output_digest_matches": (
                asynchronous["output_digest"] == sync["output_digest"]
            ),
            "matching_request_outputs": sum(
                left == right
                for left, right in zip(
                    sync["request_output_digests"],
                    asynchronous["request_output_digests"],
                )
            ),
            "mismatched_request_indices": [
                index
                for index, (left, right) in enumerate(zip(
                    sync["request_output_token_ids"],
                    asynchronous["request_output_token_ids"],
                ))
                if left != right
            ],
            "throughput_change_percent": round(
                (asynchronous["throughput_tokens_per_s"]
                 / sync["throughput_tokens_per_s"] - 1) * 100,
                3,
            ),
            "ttft_p95_change_percent": round(
                (asynchronous["ttft_p95_ms"] / sync["ttft_p95_ms"] - 1) * 100,
                3,
            ),
            "tpot_p95_change_percent": round(
                (asynchronous["tpot_p95_ms"] / sync["tpot_p95_ms"] - 1) * 100,
                3,
            ),
        },
    }
    destination = args.results_dir / "comparison.json"
    destination.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(comparison["async_vs_sync"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
