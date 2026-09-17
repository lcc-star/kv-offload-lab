"""Measure overlap between the KV swap kernel and independent GPU compute."""

import argparse
import csv
from pathlib import Path
from statistics import median

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parents[2]
SWAP_OPS = load(
    name="kv_offload_lab_overlap",
    sources=[str(ROOT / "nanovllm/csrc/swap_kernels.cu")],
    verbose=False,
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


def elapsed(launch, stream):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record()
        launch()
        end.record()
    end.synchronize()
    return start.elapsed_time(end)


def overlap_elapsed(launch_swap, launch_compute, swap_stream, compute_stream):
    current = torch.cuda.current_stream()
    start = torch.cuda.Event(enable_timing=True)
    swap_done = torch.cuda.Event()
    compute_done = torch.cuda.Event()
    end = torch.cuda.Event(enable_timing=True)

    start.record(current)
    swap_stream.wait_event(start)
    compute_stream.wait_event(start)
    with torch.cuda.stream(swap_stream):
        launch_swap()
        swap_done.record()
    with torch.cuda.stream(compute_stream):
        launch_compute()
        compute_done.record()
    current.wait_event(swap_done)
    current.wait_event(compute_done)
    end.record(current)
    end.synchronize()
    return start.elapsed_time(end)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.mappings) < 1:
        raise ValueError("mapping counts must be positive")

    device = torch.device("cuda")
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
    gpu_cache = torch.empty(shape, dtype=dtype, device=device)
    cpu_cache = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    block_bytes = gpu_cache[:, :, 0].numel() * gpu_cache.element_size()

    left = torch.randn(
        (args.matrix_size, args.matrix_size), dtype=dtype, device=device
    )
    right = torch.randn_like(left)
    output = torch.empty_like(left)
    swap_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def launch_compute():
        torch.mm(left, right, out=output)

    rows = []
    for count in args.mappings:
        pairs = [(index, index) for index in range(count)]
        mapping = torch.tensor(pairs, dtype=torch.int64, device=device)

        def launch_swap():
            SWAP_OPS.swap_blocks(gpu_cache, cpu_cache, mapping, block_bytes)

        for _ in range(args.warmup):
            launch_swap()
            launch_compute()
        torch.cuda.synchronize()

        gpu_cache.zero_()
        for block_id in range(count):
            gpu_cache[:, :, block_id].fill_(block_id + 1)
        cpu_cache.fill_(-1)
        launch_swap()
        torch.cuda.synchronize()
        for block_id in range(count):
            expected = torch.full_like(
                cpu_cache[:, :, block_id], block_id + 1
            )
            if not torch.equal(cpu_cache[:, :, block_id], expected):
                raise AssertionError(f"swap result mismatch for block {block_id}")

        swap_times = []
        compute_times = []
        serial_times = []
        overlap_times = []
        for _ in range(args.repeats):
            swap_times.append(elapsed(launch_swap, swap_stream))
            compute_times.append(elapsed(launch_compute, compute_stream))
            serial_times.append(
                elapsed(lambda: (launch_swap(), launch_compute()), compute_stream)
            )
            overlap_times.append(
                overlap_elapsed(
                    launch_swap, launch_compute, swap_stream, compute_stream
                )
            )

        swap_ms = median(swap_times)
        compute_ms = median(compute_times)
        serial_ms = median(serial_times)
        overlap_ms = median(overlap_times)
        hidden_ms = serial_ms - overlap_ms
        overlap_efficiency = hidden_ms / min(swap_ms, compute_ms)
        improvement = hidden_ms / serial_ms
        effective_gib_s = count * block_bytes / 2**30 / (swap_ms / 1000)
        rows.append(
            {
                "mappings": count,
                "mib": count * block_bytes / 2**20,
                "swap_ms": swap_ms,
                "effective_gib_s": effective_gib_s,
                "compute_ms": compute_ms,
                "serial_ms": serial_ms,
                "overlap_ms": overlap_ms,
                "hidden_ms": hidden_ms,
                "overlap_efficiency": overlap_efficiency,
                "improvement": improvement,
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
    print(f"matrix: {args.matrix_size} x {args.matrix_size}, dtype={dtype}")
    print(
        "maps  MiB    swap_ms  GiB/s compute_ms serial_ms overlap_ms "
        "efficiency improve"
    )
    for row in rows:
        print(
            f"{row['mappings']:>4} "
            f"{row['mib']:>6.1f} "
            f"{row['swap_ms']:>8.3f} "
            f"{row['effective_gib_s']:>6.2f} "
            f"{row['compute_ms']:>10.3f} "
            f"{row['serial_ms']:>9.3f} "
            f"{row['overlap_ms']:>10.3f} "
            f"{row['overlap_efficiency']:>9.1%} "
            f"{row['improvement']:>7.1%}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
