# 调度问题复现实验

本次实际执行 CPU 测试，没有修改调度器，没有加载模型、分配 GPU KV 或执行 CUDA 数据搬运。

## 方法

直接使用仓库中的 Scheduler、BlockManager、Sequence。为绕过包初始化触发的 ModelRunner/CUDA 扩展加载，独立进程中仅建立包路径；核心类未替换、未 mock。配置用标量对象构造，block size 使用项目实际默认值 256。用分配 block、追加一个采样 token、调用 preempt 的方式构建请求状态。这里的 GPU block 是真实管理器中的元数据，不含 GPU 张量。

## 实测结果

| 用例 | 结果 | 观察 |
|---|---|---|
| 4 个原有 running 请求 | 通过 | 正常返回 4 个 decode 请求 |
| 1 个 swapped 请求 | 通过 | 正常恢复并选中；映射和 GPU 页表一致 |
| 上限 4，一次换入 4 个请求 | 失败 | schedule 在 assert scheduled_seqs 处触发 AssertionError |
| 上限 4，2 个原有 running＋2 个换入 | 失败 | 显存充足但实际只选中 2 个请求，期望为 4 个 |
| 2 个完整历史 KV block，准备处理第 513 个 token，GPU 仅空闲 2 块 | 失败 | can_swap_in 为真，恢复后 can_append 为假 |

合计 5 项，2 项通过，3 项失败。失败是对未修复源码的复现结果，不是测试通过。

前两项失败指向换入计数与 decode 批次计数混用。最后一项检查的是拟议的更强准入条件“允许恢复就应有下一步空间”；当前 can_swap_in 原本只检查恢复容量。因此这项失败确认了资源条件的缺口，不单独证明实际发生传输抖动、数据错误或 GPU 故障。

## 复现

在安装 numpy、xxhash、transformers 的 Python 环境中，从仓库根目录运行：

```bash
python study/scheduler_repro/test_scheduler.py
```

当前未修复版本预期返回非零退出码，并报告 3 个失败。原始输出保存在同目录 result.txt（含本机路径，仅本地使用）。本次借用已有推理虚拟环境，缺少的 xxhash 4.0.1 安装在临时目录，通过 PYTHONPATH 引入，未修改原环境依赖。

## 下一步

先将换入计数与实际批次计数分离，再讨论恢复时预留 decode 新 block 的策略。修改后复跑这些用例，并扩展混合队列与资源不足测试；在此之前不宣称完成修复。本次没有验证 CUDA 搬运、输出 token 正确性、端到端性能或所有调度边界。

## 被测版本


- `nanovllm/engine/scheduler.py` SHA256：`335d3c829213c31c961a5c6eb603e00000af9e7f7815752635edbc247008baab`
- `nanovllm/engine/block_manager.py` SHA256：`c722fda86637e15555e756315eada8b9d0666b8329805edeedb13efd25d69147`
- `nanovllm/engine/sequence.py` SHA256：`7894d7c558b92ff5fa0a8218f7bf427ab2b69563a6ab997749d8ff284b975f31`
