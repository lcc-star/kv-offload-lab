# Swap-in 调度修复实验

## 实验目标

验证 swapped 请求恢复到 GPU 后能够进入本轮 decode，并为跨越 KV block 边界的下一个 token 保留所需空间，避免刚换入就再次被换出。

## 修复前结果

原实现的 5 项测试中有 3 项失败：

| 场景 | 修复前行为 |
|---|---|
| 一次换入 4 个请求 | `num_seqs` 已达到批次上限，decode 循环没有选中任何请求 |
| 2 个 running 加 2 个 swapped | 实际 decode batch 只有 2 个请求 |
| 恢复 2 个 KV blocks，但下一个 token 需要新 block | `can_swap_in` 返回真，恢复后 `can_append` 返回假 |

前两个失败来自 swap-in 数量与实际 decode batch 数量共用 `num_seqs`。第三个失败来自 `can_swap_in` 只计算恢复已有 KV 所需的 block，没有计算本轮 append 可能新增的 block。

## 修复方法

1. 使用 `num_swap_ins` 单独限制本轮恢复的请求数，decode 循环继续使用 `num_seqs` 统计真正进入计算批次的请求。
2. `can_swap_in` 同时计算 CPU KV 恢复空间和当前请求的 append 空间。
3. 连续恢复多个请求时，累计尚未实际分配的 append 预留，防止后续请求占用前面请求的预留空间。

## 修复后结果

```text
test_control_four_running_requests ... ok
test_control_one_swapped_request ... ok
test_four_swap_ins_should_produce_four_decode_requests ... ok
test_multiple_swap_ins_reserve_append_space ... ok
test_swap_in_requires_space_for_next_append ... ok
test_two_swap_ins_should_not_reduce_decode_capacity ... ok

Ran 6 tests
OK
```

新增的多请求测试构造了两个都需要恢复 2 个 KV blocks、随后追加 1 个 block 的请求。在只有 5 个 GPU blocks 时，调度器只恢复队头请求并为它保留 append 空间；另一个请求继续留在 swapped 队列，没有产生立即换出的 mappings。

## 复现

```bash
source .venv/bin/activate
python study/scheduler_repro/test_scheduler.py
```

这些测试直接使用仓库中的 `Scheduler`、`BlockManager` 和 `Sequence`，验证 CPU 侧调度与 block 元数据，不加载模型或执行 CUDA 数据搬运。Swap kernel 的数据正确性由 `study/swap_kernel_repro/test_swap_kernel.py` 单独验证。
