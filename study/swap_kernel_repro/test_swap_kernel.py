"""Exact block-content tests for the CUDA KV swap kernel."""

from pathlib import Path
import unittest

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
SWAP_OPS = load(
    name="kv_offload_lab_swap_test",
    sources=[str(ROOT / "nanovllm/csrc/swap_kernels.cu")],
    verbose=False,
)


def coordinate_tensor(shape: tuple[int, ...], *, device: str) -> torch.Tensor:
    """Give every tensor coordinate a distinct, exactly representable value."""
    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int32)
    return values.reshape(shape).to(device)


def run_swap(src: torch.Tensor, dst: torch.Tensor, mappings: list[tuple[int, int]]) -> None:
    mapping = torch.tensor(mappings, dtype=torch.int64, device="cuda")
    logical_block_bytes = src[:, :, 0].numel() * src.element_size()
    SWAP_OPS.swap_blocks(src, dst, mapping, logical_block_bytes)
    torch.cuda.synchronize()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class SwapKernelLayoutTests(unittest.TestCase):
    def test_non_identity_swap_out_copies_logical_blocks_only(self):
        # [K/V, layer, physical block, token, KV head, head dimension]
        gpu = coordinate_tensor((2, 2, 6, 4, 1, 2), device="cuda")
        cpu = torch.full(
            (2, 2, 4, 4, 1, 2),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        before = gpu.cpu().clone()

        run_swap(gpu, cpu, [(1, 3), (4, 0)])

        self.assertTrue(torch.equal(cpu[:, :, 3], before[:, :, 1]))
        self.assertTrue(torch.equal(cpu[:, :, 0], before[:, :, 4]))
        self.assertTrue(torch.all(cpu[:, :, 1] == -1))
        self.assertTrue(torch.all(cpu[:, :, 2] == -1))

    def test_round_trip_can_restore_to_different_gpu_blocks(self):
        gpu = coordinate_tensor((2, 2, 6, 4, 1, 2), device="cuda")
        cpu = torch.full(
            (2, 2, 4, 4, 1, 2),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        original = gpu.cpu().clone()
        run_swap(gpu, cpu, [(1, 3), (4, 0)])

        gpu[:, :, 1] = -2
        gpu[:, :, 4] = -2
        untouched = gpu.cpu().clone()
        run_swap(cpu, gpu, [(3, 2), (0, 5)])

        self.assertTrue(torch.equal(gpu[:, :, 2].cpu(), original[:, :, 1]))
        self.assertTrue(torch.equal(gpu[:, :, 5].cpu(), original[:, :, 4]))
        for block_id in (0, 1, 3, 4):
            self.assertTrue(torch.equal(gpu[:, :, block_id].cpu(), untouched[:, :, block_id]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
