"""Benchmark: KV cache swap vs recompute under memory pressure."""
import os
import sys
import time
import json
from random import randint, seed

def run_bench(label, gpu_mem, cpu_swap_gb, num_seqs=128, max_input_len=512, max_output_len=512):
    from nanovllm import LLM, SamplingParams
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    seed(0)
    llm = LLM(path, enforce_eager=True, max_model_len=4096,
              gpu_memory_utilization=gpu_mem, cpu_swap_space_gb=cpu_swap_gb)

    bm = llm.scheduler.block_manager
    gpu_blocks = len(bm.blocks)
    cpu_blocks = bm.num_cpu_blocks

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    # warmup
    llm.generate([[0]*10], SamplingParams(max_tokens=1), use_tqdm=False)

    # track swap events
    swap_out_blocks = 0
    swap_in_blocks = 0
    orig_schedule = llm.scheduler.schedule
    def tracked_schedule():
        nonlocal swap_out_blocks, swap_in_blocks
        result = orig_schedule()
        _, _, swap_in, swap_out = result
        swap_out_blocks += len(swap_out)
        swap_in_blocks += len(swap_in)
        return result
    llm.scheduler.schedule = tracked_schedule

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed
    result = dict(label=label, gpu_blocks=gpu_blocks, cpu_blocks=cpu_blocks,
                  total_tokens=total_tokens, elapsed=round(elapsed, 2),
                  throughput=round(throughput), swap_out=swap_out_blocks, swap_in=swap_in_blocks)
    print(json.dumps(result))

if __name__ == "__main__":
    label = sys.argv[1]
    gpu_mem = float(sys.argv[2])
    cpu_swap = float(sys.argv[3])
    run_bench(label, gpu_mem, cpu_swap)
