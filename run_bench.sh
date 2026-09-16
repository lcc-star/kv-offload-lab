#!/bin/bash
PYTHON=/root/nano-vllm/.venv/bin/python
cd /root/nano-vllm

echo "=== KV Cache Swap Performance Report ==="
echo ""

echo "--- Test 1: Normal (0.9 GPU, no swap) ---"
R1=$($PYTHON bench_swap.py "Normal" 0.9 0 2>/dev/null)
echo "$R1"

echo ""
echo "--- Test 2: Swap enabled (0.12 GPU + 2GB CPU) ---"
R2=$($PYTHON bench_swap.py "Swap" 0.12 2.0 2>/dev/null)
echo "$R2"

echo ""
echo "--- Test 3: Recompute only (0.12 GPU, no swap) ---"
R3=$($PYTHON bench_swap.py "Recompute" 0.12 0 2>/dev/null)
echo "$R3"

echo ""
echo "=== Summary ==="
T1=$(echo "$R1" | python3 -c "import sys,json; print(json.load(sys.stdin)['throughput'])")
T2=$(echo "$R2" | python3 -c "import sys,json; print(json.load(sys.stdin)['throughput'])")
T3=$(echo "$R3" | python3 -c "import sys,json; print(json.load(sys.stdin)['throughput'])")
echo "Normal:    ${T1} tok/s"
echo "Swap:      ${T2} tok/s"
echo "Recompute: ${T3} tok/s"
python3 -c "
t1,t2,t3 = $T1,$T2,$T3
print(f'Swap vs Normal:    {t2/t1*100:.1f}%')
print(f'Recomp vs Normal:  {t3/t1*100:.1f}%')
if t3 > 0:
    print(f'Swap vs Recompute: {t2/t3:.2f}x')
"
