"""Measure swap bandwidth and statistics."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))

from random import randint, seed
from nanovllm import LLM, SamplingParams

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
seed(0)

llm = LLM(path, enforce_eager=True, tensor_parallel_size=1,
           gpu_memory_utilization=0.10, cpu_swap_space_gb=2.0,
           max_num_seqs=60, max_num_batched_tokens=16384, max_model_len=4096)

bm = llm.scheduler.block_manager
runner = llm.model_runner

# Calculate block size in bytes
hf = llm.scheduler.block_manager.block_size
kv = runner.kv_cache
block_bytes = kv[:, :, 0].numel() * kv.element_size()  # bytes per block

print(f"GPU blocks: {len(bm.blocks)}, CPU blocks: {bm.num_cpu_blocks}")
print(f"Block size: {hf} tokens, {block_bytes / 1024:.1f} KB per block")
print(f"KV cache shape per block: {list(kv[:, :, 0].shape)}, dtype: {kv.dtype}")

# Monkey-patch to measure swap timing
import torch
swap_stats = {"out_calls": 0, "in_calls": 0, "out_blocks": 0, "in_blocks": 0,
              "out_time": 0.0, "in_time": 0.0}

orig_run = runner.run.__func__

def timed_run(self, seqs, is_prefill, swap_in=None, swap_out=None):
    if swap_out:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.swap_blocks(self.kv_cache, self.cpu_kv_cache, swap_out)
        t1 = time.perf_counter()
        swap_stats["out_calls"] += 1
        swap_stats["out_blocks"] += len(swap_out)
        swap_stats["out_time"] += t1 - t0
    if swap_in:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.swap_blocks(self.cpu_kv_cache, self.kv_cache, swap_in)
        t1 = time.perf_counter()
        swap_stats["in_calls"] += 1
        swap_stats["in_blocks"] += len(swap_in)
        swap_stats["in_time"] += t1 - t0
    from nanovllm.utils.context import set_context, get_context, reset_context
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
    reset_context()
    return token_ids

import types
runner.run = types.MethodType(timed_run, runner)

# Run workload
num_seqs = 128
prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, 512))] for _ in range(num_seqs)]
sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 512)) for _ in range(num_seqs)]

llm.generate([[0]*10], SamplingParams(max_tokens=1), use_tqdm=False)  # warmup

t_start = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
t_total = time.time() - t_start

total_tokens = sum(sp.max_tokens for sp in sampling_params)

print(f"\n{'='*60}")
print(f"Workload: {num_seqs} seqs, {total_tokens} tokens, {t_total:.2f}s")
print(f"Throughput: {total_tokens / t_total:.0f} tok/s")
print()

for direction in ["out", "in"]:
    calls = swap_stats[f"{direction}_calls"]
    blocks = swap_stats[f"{direction}_blocks"]
    t = swap_stats[f"{direction}_time"]
    label = "Swap-out (GPU→CPU)" if direction == "out" else "Swap-in  (CPU→GPU)"
    if calls > 0:
        data_mb = blocks * block_bytes / 1024 / 1024
        bw = data_mb / t / 1024 if t > 0 else 0
        print(f"{label}:")
        print(f"  调用次数:     {calls}")
        print(f"  总 blocks:    {blocks} ({data_mb:.1f} MB)")
        print(f"  平均 blocks:  {blocks / calls:.1f} / 次")
        print(f"  总耗时:       {t*1000:.1f} ms")
        print(f"  平均耗时:     {t/calls*1000:.2f} ms / 次")
        print(f"  平均带宽:     {bw:.2f} GB/s")
        print()
    else:
        print(f"{label}: 无")
        print()

swap_total = swap_stats["out_time"] + swap_stats["in_time"]
print(f"Swap 总耗时: {swap_total*1000:.1f} ms ({swap_total/t_total*100:.1f}% of total)")
