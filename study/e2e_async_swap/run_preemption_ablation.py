"""Compare sync, serial async preemption and capacity-aware batching."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from study.e2e_async_swap import run_matrix


run_matrix.MODES = {
    "sync": ["--mode", "sync"],
    "serial_full_async": [
        "--mode", "async",
        "--unsafe-async-swap-out",
        "--serial-async-preemption",
    ],
    "batched_full_async": [
        "--mode", "async",
        "--unsafe-async-swap-out",
    ],
}
run_matrix.ORDERS = [
    ("sync", "serial_full_async", "batched_full_async"),
    ("batched_full_async", "serial_full_async", "sync"),
    ("serial_full_async", "sync", "batched_full_async"),
]


if __name__ == "__main__":
    run_matrix.main()
