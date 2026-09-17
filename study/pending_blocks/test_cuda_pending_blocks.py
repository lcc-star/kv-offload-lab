"""GPU integration test for pending blocks and real CUDA events."""

from pathlib import Path
import sys
import time
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pending_block_manager import BlockState, PendingBlockManager, RequestState
from study.swap_overlap.benchmark_overlap import SWAP_OPS


def reclaim_with_timeout(manager, operation, timeout=5.0):
    deadline = time.monotonic() + timeout
    while operation in manager.pending:
        manager.reclaim_completed()
        if operation not in manager.pending:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("CUDA event did not complete")
        time.sleep(0.001)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class CudaPendingBlockTests(unittest.TestCase):
    def test_async_round_trip_commits_only_after_events(self):
        shape = (2, 28, 6, 256, 8, 128)
        gpu_cache = torch.zeros(shape, dtype=torch.float16, device="cuda")
        cpu_cache = torch.full(
            shape,
            -1,
            dtype=torch.float16,
            device="cpu",
            pin_memory=True,
        )
        block_bytes = gpu_cache[:, :, 0].numel() * gpu_cache.element_size()
        gpu_cache[:, :, 0].fill_(11)
        gpu_cache[:, :, 1].fill_(22)
        torch.cuda.synchronize()

        manager = PendingBlockManager(num_gpu_blocks=6, num_cpu_blocks=6)
        manager.add_running(1, [0, 1])
        stream = torch.cuda.Stream()

        out_event = torch.cuda.Event()
        out_operation = manager.reserve_swap_out(1, [2, 3], out_event)
        out_mapping = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64)
        with torch.cuda.stream(stream):
            SWAP_OPS.copy_blocks_2d(
                gpu_cache, cpu_cache, out_mapping, block_bytes
            )
            out_event.record()

        self.assertEqual(manager.gpu_states[0], BlockState.PENDING_FREE)
        self.assertEqual(manager.cpu_states[2], BlockState.PENDING_WRITE)
        self.assertEqual(manager.request_states[1], RequestState.SWAPPING_OUT)
        with self.assertRaisesRegex(ValueError, "not FREE"):
            manager.add_running(2, [0])
        with self.assertRaisesRegex(ValueError, "not FREE"):
            manager.add_swapped(3, [2])

        reclaim_with_timeout(manager, out_operation)
        self.assertEqual(manager.gpu_states[0], BlockState.FREE)
        self.assertEqual(manager.cpu_states[2], BlockState.USED)
        self.assertEqual(manager.request_states[1], RequestState.SWAPPED)
        self.assertTrue(torch.all(cpu_cache[:, :, 2] == 11))
        self.assertTrue(torch.all(cpu_cache[:, :, 3] == 22))
        manager.add_running(2, [0])
        self.assertEqual(manager.gpu_states[0], BlockState.USED)

        gpu_cache[:, :, 4].fill_(-2)
        gpu_cache[:, :, 5].fill_(-2)
        torch.cuda.synchronize()
        in_event = torch.cuda.Event()
        in_operation = manager.reserve_swap_in(1, [4, 5], in_event)
        in_mapping = torch.tensor([[2, 4], [3, 5]], dtype=torch.int64)
        with torch.cuda.stream(stream):
            SWAP_OPS.copy_blocks_2d(
                cpu_cache, gpu_cache, in_mapping, block_bytes
            )
            in_event.record()

        self.assertEqual(manager.cpu_states[2], BlockState.PENDING_FREE)
        self.assertEqual(manager.gpu_states[4], BlockState.PENDING_WRITE)
        self.assertEqual(manager.request_states[1], RequestState.SWAPPING_IN)

        reclaim_with_timeout(manager, in_operation)
        self.assertEqual(manager.cpu_states[2], BlockState.FREE)
        self.assertEqual(manager.gpu_states[4], BlockState.USED)
        self.assertEqual(manager.request_states[1], RequestState.RUNNING)
        self.assertTrue(torch.all(gpu_cache[:, :, 4] == 11))
        self.assertTrue(torch.all(gpu_cache[:, :, 5] == 22))
        manager.add_swapped(3, [2])
        self.assertEqual(manager.cpu_states[2], BlockState.USED)


if __name__ == "__main__":
    unittest.main(verbosity=2)
