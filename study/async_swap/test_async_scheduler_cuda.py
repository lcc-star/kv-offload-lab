"""Real-CUDA integration tests for Scheduler's asynchronous swap lifecycle."""

from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
package = ModuleType("nanovllm")
package.__path__ = [str(ROOT / "nanovllm")]
sys.modules["nanovllm"] = package

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


SWAP_OPS = load(
    name="kv_offload_lab_async_scheduler_test",
    sources=[str(ROOT / "nanovllm/csrc/swap_kernels.cu")],
    verbose=False,
)


def make_scheduler(gpu_blocks=3):
    config = SimpleNamespace(
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        max_swap_skips=2,
        async_swap=True,
        eos=-1,
        num_kvcache_blocks=gpu_blocks,
        kvcache_block_size=256,
        num_cpu_blocks=8,
    )
    return Scheduler(config)


def wait_for_reclaim(scheduler, seq, expected_status, timeout=5.0):
    deadline = time.monotonic() + timeout
    while seq.status != expected_status:
        scheduler.reclaim_completed_swaps()
        if time.monotonic() >= deadline:
            raise TimeoutError("CUDA swap event did not complete")
        time.sleep(0.001)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AsyncSchedulerCudaTests(unittest.TestCase):
    def setUp(self):
        self.gpu = torch.zeros(
            (2, 2, 3, 4, 1, 2), dtype=torch.int32, device="cuda"
        )
        self.cpu = torch.full(
            (2, 2, 8, 4, 1, 2),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        self.block_bytes = self.gpu[:, :, 0].numel() * self.gpu.element_size()
        self.stream = torch.cuda.Stream()

    def submit(self, src, dst, mappings):
        mapping = torch.tensor(mappings, dtype=torch.int64, device="cpu")
        with torch.cuda.stream(self.stream):
            SWAP_OPS.copy_blocks_2d(src, dst, mapping, self.block_bytes)
            event = torch.cuda.Event()
            event.record()
        return event

    def test_swap_out_does_not_release_gpu_block_before_event(self):
        scheduler = make_scheduler()
        seq = Sequence([10, 11])
        scheduler.block_manager.allocate(seq)
        seq.append_token(12)
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)
        gpu_block = seq.block_table[0]
        self.gpu[:, :, gpu_block].fill_(37)

        scheduler.running.remove(seq)
        mappings = scheduler.preempt(seq)
        self.assertEqual(seq.status, SequenceStatus.SWAPPING_OUT)
        self.assertIn(gpu_block, scheduler.block_manager.used_block_ids)

        event = self.submit(self.gpu, self.cpu, mappings)
        scheduler.bind_swap_events(None, event)
        wait_for_reclaim(scheduler, seq, SequenceStatus.SWAPPED)

        cpu_block = seq.cpu_block_table[0]
        self.assertNotIn(gpu_block, scheduler.block_manager.used_block_ids)
        self.assertTrue(torch.all(self.cpu[:, :, cpu_block] == 37))

    def test_swap_in_becomes_runnable_only_after_copy_event(self):
        scheduler = make_scheduler()
        seq = Sequence([20, 21])
        scheduler.block_manager.allocate(seq)
        seq.append_token(22)
        scheduler.block_manager.swap_out(seq)
        seq.status = SequenceStatus.SWAPPED
        scheduler.swapped.append(seq)
        cpu_block = seq.cpu_block_table[0]
        self.cpu[:, :, cpu_block].fill_(83)

        scheduled, _, mappings, outgoing = scheduler.schedule()
        self.assertFalse(scheduled)
        self.assertFalse(outgoing)
        self.assertEqual(seq.status, SequenceStatus.SWAPPING_IN)

        gpu_block = mappings[0][1]
        event = self.submit(self.cpu, self.gpu, mappings)
        scheduler.bind_swap_events(event, None)
        wait_for_reclaim(scheduler, seq, SequenceStatus.RUNNING)

        self.assertFalse(seq.cpu_block_table)
        self.assertTrue(torch.all(self.gpu[:, :, gpu_block] == 83))


if __name__ == "__main__":
    unittest.main(verbosity=2)
