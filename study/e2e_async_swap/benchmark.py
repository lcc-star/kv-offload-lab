"""Run one real-model KV swap benchmark and emit one JSON result."""

import argparse
import hashlib
import json
from pathlib import Path
from random import Random
import statistics
import sys
import time

import torch
from torch import nn
from transformers import AutoConfig


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def percentile(values, fraction):
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction))
    return ordered[index]


class GreedySampler(nn.Module):
    def forward(self, logits, temperatures):
        return logits.argmax(dim=-1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("sync", "async"), required=True)
    parser.add_argument("--num-requests", type=int, default=48)
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.10)
    parser.add_argument("--num-gpu-blocks", type=int)
    parser.add_argument("--cpu-swap-space-gb", type=float, default=8.0)
    parser.add_argument("--max-num-seqs", type=int, default=48)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--unsafe-async-swap-out", action="store_true")
    parser.add_argument("--profile-active-range", action="store_true")
    parser.add_argument(
        "--synchronize-swap",
        choices=("none", "in", "out", "both"),
        default="none",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_len + args.output_len > args.max_model_len:
        raise ValueError("input_len + output_len must not exceed max_model_len")
    from nanovllm import LLM, SamplingParams

    random = Random(args.seed)
    torch.manual_seed(args.seed)
    vocab_size = AutoConfig.from_pretrained(args.model).vocab_size
    prompts = [
        [random.randrange(vocab_size) for _ in range(args.input_len)]
        for _ in range(args.num_requests)
    ]
    sampling = SamplingParams(
        temperature=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    llm = LLM(
        args.model,
        enforce_eager=True,
        tensor_parallel_size=1,
        async_swap=args.mode == "async",
        unsafe_async_swap_out=args.unsafe_async_swap_out,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_gpu_blocks or -1,
        cpu_swap_space_gb=args.cpu_swap_space_gb,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_model_len,
        max_model_len=args.max_model_len,
    )
    llm.model_runner.sampler = GreedySampler()
    if args.synchronize_swap != "none":
        original_swap_blocks_async = llm.model_runner.swap_blocks_async

        def synchronized_swap_blocks_async(src, dst, mappings):
            event = original_swap_blocks_async(src, dst, mappings)
            direction = "out" if src.is_cuda else "in"
            if event is not None and args.synchronize_swap in (direction, "both"):
                event.synchronize()
            return event

        llm.model_runner.swap_blocks_async = synchronized_swap_blocks_async
    llm.generate([[0] * 8], SamplingParams(max_tokens=1), use_tqdm=False)

    scheduler = llm.scheduler
    swap_counts = {"in_blocks": 0, "out_blocks": 0, "pending_only_steps": 0}
    original_schedule = scheduler.schedule

    def tracked_schedule():
        scheduled, is_prefill, swap_in, swap_out = original_schedule()
        swap_counts["in_blocks"] += len(swap_in)
        swap_counts["out_blocks"] += len(swap_out)
        if not scheduled and (scheduler.swapping_in or scheduler.swapping_out):
            swap_counts["pending_only_steps"] += 1
        return scheduled, is_prefill, swap_in, swap_out

    scheduler.schedule = tracked_schedule
    for prompt in prompts:
        llm.add_request(prompt, sampling)
    sequences = list(scheduler.waiting)

    torch.cuda.reset_peak_memory_stats()
    if args.profile_active_range:
        torch.cuda.profiler.start()
    started = time.perf_counter()
    first_token_at = {}
    finished_at = {}
    step_count = 0
    while not llm.is_finished():
        llm.step()
        step_count += 1
        now = time.perf_counter()
        for seq in sequences:
            if seq.num_completion_tokens and seq.seq_id not in first_token_at:
                first_token_at[seq.seq_id] = now
            if seq.is_finished and seq.seq_id not in finished_at:
                finished_at[seq.seq_id] = now
    torch.cuda.synchronize()
    ended = time.perf_counter()
    if args.profile_active_range:
        torch.cuda.profiler.stop()

    ttft_ms = [(first_token_at[seq.seq_id] - started) * 1000 for seq in sequences]
    tpot_ms = [
        (finished_at[seq.seq_id] - first_token_at[seq.seq_id])
        * 1000 / max(1, seq.num_completion_tokens - 1)
        for seq in sequences
    ]
    total_tokens = sum(seq.num_completion_tokens for seq in sequences)
    digest = hashlib.sha256()
    request_output_digests = []
    for seq in sequences:
        tokens = bytes(str(seq.completion_token_ids), "utf-8")
        digest.update(tokens)
        request_output_digests.append(hashlib.sha256(tokens).hexdigest())

    result = {
        "mode": args.mode,
        "model": str(Path(args.model).resolve()),
        "seed": args.seed,
        "sampling": "argmax",
        "synchronize_swap": args.synchronize_swap,
        "unsafe_async_swap_out": args.unsafe_async_swap_out,
        "num_requests": args.num_requests,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "requested_gpu_blocks": args.num_gpu_blocks,
        "gpu_blocks": len(scheduler.block_manager.blocks),
        "cpu_blocks": scheduler.block_manager.num_cpu_blocks,
        "elapsed_s": round(ended - started, 6),
        "throughput_tokens_per_s": round(total_tokens / (ended - started), 3),
        "ttft_mean_ms": round(statistics.mean(ttft_ms), 3),
        "ttft_p50_ms": round(percentile(ttft_ms, 0.50), 3),
        "ttft_p95_ms": round(percentile(ttft_ms, 0.95), 3),
        "tpot_mean_ms": round(statistics.mean(tpot_ms), 3),
        "tpot_p50_ms": round(percentile(tpot_ms, 0.50), 3),
        "tpot_p95_ms": round(percentile(tpot_ms, 0.95), 3),
        "swap_in_blocks": swap_counts["in_blocks"],
        "swap_out_blocks": swap_counts["out_blocks"],
        "pending_only_steps": swap_counts["pending_only_steps"],
        "scheduler_steps": step_count,
        "peak_gpu_memory_mib": round(
            torch.cuda.max_memory_allocated() / 1024 / 1024, 3
        ),
        "output_digest": digest.hexdigest(),
        "request_output_digests": request_output_digests,
        "request_output_token_ids": [
            seq.completion_token_ids for seq in sequences
        ],
    }
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    print("RESULT_JSON=" + rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
