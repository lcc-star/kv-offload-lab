#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 MODEL_DIR OUTPUT_DIR [CUDA_DEVICE]"
    exit 2
fi

MODEL_DIR=$1
OUTPUT_DIR=$2
CUDA_DEVICE=${3:-0}
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
BENCHMARK="$ROOT_DIR/study/e2e_async_swap/benchmark.py"
ANALYZER="$ROOT_DIR/study/e2e_async_swap/analyze_nsys_overlap.py"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
    --model "$MODEL_DIR"
    --mode async
    --profile-active-range
    --num-requests 12
    --input-len 512
    --output-len 8
    --max-num-seqs 12
    --max-model-len 768
    --gpu-memory-utilization 0.35
    --num-gpu-blocks 20
    --cpu-swap-space-gb 2
    --seed 0
)

for MODE in safe unsafe; do
    EXTRA_ARGS=()
    if [[ "$MODE" == unsafe ]]; then
        EXTRA_ARGS+=(--unsafe-async-swap-out)
    fi
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --force-overwrite=true \
        --output="$OUTPUT_DIR/$MODE" \
        "$PYTHON_BIN" "$BENCHMARK" \
        "${COMMON_ARGS[@]}" "${EXTRA_ARGS[@]}" \
        --output "$OUTPUT_DIR/$MODE-result.json"

    nsys stats \
        --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_gpu_kern_sum \
        --format csv \
        --force-export=true \
        "$OUTPUT_DIR/$MODE.nsys-rep" > "$OUTPUT_DIR/$MODE-stats.csv"

    "$PYTHON_BIN" "$ANALYZER" "$OUTPUT_DIR/$MODE.sqlite" \
        --output "$OUTPUT_DIR/$MODE-overlap.json"
done
