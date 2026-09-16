"""Test KV cache swap to CPU memory."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import SequenceStatus
from transformers import AutoTokenizer

path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
tokenizer = AutoTokenizer.from_pretrained(path)

# Use very low gpu_memory_utilization to limit GPU blocks and force swapping
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1,
           gpu_memory_utilization=0.10, cpu_swap_space_gb=2.0,
           max_num_seqs=60, max_num_batched_tokens=16384, max_model_len=4096)

engine = llm
scheduler = engine.scheduler
bm = scheduler.block_manager

print(f"GPU blocks: {len(bm.blocks)}")
print(f"CPU blocks: {bm.num_cpu_blocks}")
print(f"Free GPU blocks: {len(bm.free_block_ids)}")
print(f"Free CPU blocks: {len(bm.free_cpu_block_ids)}")

# Send many prompts to trigger memory pressure with 137 GPU blocks
# Each seq with 512 max_tokens needs ~3 blocks (256 tokens/block), so ~45 seqs should fill up
sampling_params = SamplingParams(temperature=0.6, max_tokens=512)
base_prompts = [
    "Write a very long and detailed essay about the history of artificial intelligence, "
    "covering all major milestones from the 1950s to the present day.",
    "Explain quantum computing in great detail, including qubits, superposition, "
    "entanglement, quantum gates, and practical applications.",
    "Describe the complete process of photosynthesis at the molecular level, "
    "including light reactions and the Calvin cycle.",
    "Write a comprehensive guide to machine learning algorithms, covering "
    "supervised, unsupervised, and reinforcement learning approaches.",
]
# Repeat to create many concurrent requests
prompts = base_prompts * 15  # 60 prompts
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
    )
    for p in prompts
]

# Monkey-patch scheduler to track swap events
swap_out_count = 0
swap_in_count = 0
orig_schedule = scheduler.schedule

def tracked_schedule():
    global swap_out_count, swap_in_count
    result = orig_schedule()
    seqs, is_prefill, swap_in, swap_out = result
    if swap_out:
        swap_out_count += 1
        print(f"  [SWAP-OUT] {len(swap_out)} blocks swapped to CPU")
    if swap_in:
        swap_in_count += 1
        print(f"  [SWAP-IN] {len(swap_in)} blocks swapped from CPU")
    return result

scheduler.schedule = tracked_schedule

outputs = llm.generate(prompts, sampling_params)

print(f"\n--- Results ---")
print(f"Swap-out events: {swap_out_count}")
print(f"Swap-in events: {swap_in_count}")
print(f"Swapped queue size: {len(scheduler.swapped)}")
print(f"Free GPU blocks: {len(bm.free_block_ids)}")
print(f"Free CPU blocks: {len(bm.free_cpu_block_ids)}")

for i, output in enumerate(outputs):
    text = output['text'][:100]
    print(f"Prompt {i}: {len(output['token_ids'])} tokens generated, preview: {text!r}...")

if swap_out_count > 0 or swap_in_count > 0:
    print("\n✓ KV cache swap to CPU memory is working!")
else:
    print("\nNote: No swap events triggered (GPU memory was sufficient)")
