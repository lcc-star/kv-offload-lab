"""Compare zero-copy and cudaMemcpy2DAsync for KV block transfers."""

import argparse
import csv
from pathlib import Path
from statistics import median
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from study.swap_overlap.benchmark_overlap import (
    SWAP_OPS,
    elapsed,
    overlap_elapsed,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mappings", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.csv"),
    )
    return parser.parse_args()


def validate_copy(launch, src, dst, pairs):
    src.zero_()
    dst.fill_(-1)
    for value, (src_block, _) in enumerate(pairs, start=1):
        src[:, :, src_block].fill_(value)
    launch()
    torch.cuda.synchronize()
    for value, (_, dst_block) in enumerate(pairs, start=1):
        actual = dst[:, :, dst_block].cpu()
        if not torch.all(actual == value):
            raise AssertionError(f"copy mismatch for destination block {dst_block}")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.mappings) < 1:
        raise ValueError("mapping counts must be positive")

    dtype = torch.float16
    num_blocks = max(args.mappings) + 2
    shape = (
        2,
        args.layers,
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
    )
    gpu_cache = torch.empty(shape, dtype=dtype, device="cuda")
    cpu_cache = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    block_bytes = gpu_cache[:, :, 0].numel() * gpu_cache.element_size()

    left = torch.randn(
        (args.matrix_size, args.matrix_size), dtype=dtype, device="cuda"
    )
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    swap_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def launch_compute():
        torch.mm(left, right, out=output)

    rows = []
    for count in args.mappings:
        pairs = [(index, num_blocks - 1 - index) for index in range(count)]
        gpu_mapping = torch.tensor(pairs, dtype=torch.int64, device="cuda")
        cpu_mapping = torch.tensor(pairs, dtype=torch.int64, device="cpu")

        for direction, src, dst in (
            ("gpu_to_cpu", gpu_cache, cpu_cache),
            ("cpu_to_gpu", cpu_cache, gpu_cache),
        ):
            for method in ("zero_copy", "memcpy2d"):
                mapping = gpu_mapping if method == "zero_copy" else cpu_mapping

                def launch_transfer():
                    if method == "zero_copy":
                        SWAP_OPS.swap_blocks(src, dst, mapping, block_bytes)
                    else:
                        SWAP_OPS.copy_blocks_2d(src, dst, mapping, block_bytes)

                validate_copy(launch_transfer, src, dst, pairs)
                for _ in range(args.warmup):
                    launch_transfer()
                    launch_compute()
                torch.cuda.synchronize()

                transfer_times = []
                compute_times = []
                serial_times = []
                overlap_times = []
                for _ in range(args.repeats):
                    transfer_times.append(elapsed(launch_transfer, swap_stream))
                    compute_times.append(elapsed(launch_compute, compute_stream))
                    serial_times.append(
                        elapsed(
                            lambda: (launch_transfer(), launch_compute()),
                            compute_stream,
                        )
                    )
                    overlap_times.append(
                        overlap_elapsed(
                            launch_transfer,
                            launch_compute,
                            swap_stream,
                            compute_stream,
                        )
                    )

                transfer_ms = median(transfer_times)
                compute_ms = median(compute_times)
                serial_ms = median(serial_times)
                overlap_ms = median(overlap_times)
                hidden_ms = serial_ms - overlap_ms
                rows.append(
                    {
                        "method": method,
                        "direction": direction,
                        "mappings": count,
                        "mib": count * block_bytes / 2**20,
                        "transfer_ms": transfer_ms,
                        "effective_gib_s": (
                            count * block_bytes / 2**30 / (transfer_ms / 1000)
                        ),
                        "compute_ms": compute_ms,
                        "serial_ms": serial_ms,
                        "overlap_ms": overlap_ms,
                        "hidden_ms": hidden_ms,
                        "overlap_efficiency": (
                            hidden_ms / min(transfer_ms, compute_ms)
                        ),
                        "improvement": hidden_ms / serial_ms,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"block size: {block_bytes / 2**20:.2f} MiB")
    print("method     direction  maps  GiB/s transfer serial overlap efficiency improve")
    for row in rows:
        print(
            f"{row['method']:<10} "
            f"{row['direction']:<10} "
            f"{row['mappings']:>4} "
            f"{row['effective_gib_s']:>6.2f} "
            f"{row['transfer_ms']:>8.3f} "
            f"{row['serial_ms']:>6.3f} "
            f"{row['overlap_ms']:>7.3f} "
            f"{row['overlap_efficiency']:>9.1%} "
            f"{row['improvement']:>7.1%}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
