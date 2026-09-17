# D2H Overlap 的 Nsight Systems 分析

## 问题

真实 Qwen3 压力实验中，完全异步的 D2H swap-out 会让 1/12 个请求从第二个生成 token 开始与同步基线分叉。等待 D2H event 后，12/12 请求恢复一致。

本实验回答两个问题：

1. D2H 是否真的与模型 kernel 重叠。
2. 重叠时覆盖了哪些 kernel。

## 方法

使用 Nsight Systems 2025.1，只采集正式请求区间。safe 组在 D2H 提交后等待 event；unsafe 组通过仅诊断用的 unsafe-async-swap-out 选项允许 D2H 与后续模型计算重叠。

负载保持为 Qwen3-1.7B、20 个 GPU KV blocks、12 个请求、512 tokens 输入和 8 tokens 输出。分析器只统计大于 1 MB 的 D2H，排除小型框架复制。

## 结果

| 指标 | safe | unsafe |
| --- | ---: | ---: |
| 输出匹配 | 12/12 | 11/12 |
| 大块 D2H 与 kernel 重叠的复制次数 | 0 | 4 |
| 重叠 kernel 区间数 | 0 | 70 |
| 各区间重叠时间之和 | 0 ms | 1.846 ms |

unsafe 组中重叠最多的是：

| kernel 类别 | 重叠时间 |
| --- | ---: |
| BF16 GEMM 128x256 | 0.899 ms |
| BF16 GEMM 256x128 | 0.348 ms |
| FlashAttention forward | 0.178 ms |
| BF16 GEMM 128x128 | 0.156 ms |
| store_kvcache_kernel | 0.018 ms |

unsafe 组还执行了 18 个 swap-out 和 18 个 swap-in，safe 组各执行 12 个。一次输出分叉改变了后续生成内容，也改变了调度和 swap 轨迹，因此不能直接把两组总复制量差异解释成性能收益或损失。

## 证据边界

时间线证明大块 D2H 与 GEMM、attention 和 KV 写入 kernel 同时执行，并且这种重叠与输出分叉同时出现。Nsight Systems 时间线不包含每个 kernel 实际访问的 KV block 编号，因此不能据此断言 D2H 和 kernel 访问了同一物理地址。

## 首个分叉 token 的数值分析

进一步记录请求 block table、pending D2H blocks 和完整 logits 后发现：

- 分叉发生在请求索引 10 的第 2 个生成 token。
- 该请求在 safe 和 unsafe 两组中都没有发生 swap。
- 该请求的 GPU blocks 与 pending D2H source blocks 没有交集。
- safe 组在该位置有三个候选 logits 同为 16.125，argmax 选择 token 1986。
- unsafe 组 token 13874 为 16.125，其他主要候选为 16.0，argmax 变为 13874。

完整 151,936 维 logits 的比较如下：

| 指标 | 数值 |
| --- | ---: |
| 余弦相似度 | 0.999979 |
| 最大绝对误差 | 0.15625 |
| 平均绝对误差 | 0.06650 |
| RMSE | 0.06970 |
| 误差大于 0.125 的元素 | 6 |

这些结果不支持“分叉请求的 KV block 被 D2H 覆盖”这一解释。更符合现有证据的解释是：D2H overlap 改变了调度轨迹、物理 block 分配或 GPU kernel 并发状态，引起 BF16 数值漂移；该 token 的前三个候选原本恰好并列，因此小幅漂移改变 argmax，随后自回归生成发生级联分叉。

这仍然是基于实验的推断。若要确定具体是哪一个 kernel 产生差异，需要保存逐层激活或对首个分叉步骤执行逐层对比。

## 当前决策

生产配置继续等待 D2H event，只开放已经通过逐 token 一致性验证的 H2D overlap。虽然完整 logits 高度相似，但本项目当前把确定性输出作为验收条件，因此不会用数值容差为 D2H overlap 放行。unsafe-async-swap-out 仅用于诊断，默认关闭，并且要求同时开启 async-swap。

复现实验：

    bash study/e2e_async_swap/profile_d2h_overlap.sh MODEL_DIR OUTPUT_DIR CUDA_DEVICE

脚本生成 safe/unsafe 的 nsys-rep、CSV 汇总和 overlap JSON。报告文件可能包含本机路径和较大的二进制时间线，不纳入 Git。
