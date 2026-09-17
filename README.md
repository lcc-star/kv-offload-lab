# KV Offload Lab

面向 LLM 推理系统的 KV Cache 换出、调度与正确性实验项目，由 lcc-star 独立维护。

项目代码由 Nano-vLLM 及其 KV Cache swap 实现衍生而来，现由 lcc-star 独立维护。原始版权与授权条款保留在 [MIT 许可证](LICENSE) 中。

## 当前工作

- 阅读并分析 waiting / running / swapped 队列、GPU/CPU block 管理和换入换出流程。
- 用真实 Scheduler、BlockManager、Sequence 完成无模型 CPU 复现实验。
- 已修复换入计数导致空批次断言、decode 批次缩小等问题，并验证恢复 KV 与下一步 decode 空间条件的区别。
- 已实现可选的单卡异步 KV swap：独立 CUDA stream 提交传输，CUDA event 完成后再提交资源状态。

## 学习与实验入口

- [CPU 调度复现实验报告](study/scheduler_repro/REPORT.md)
- [对应测试代码](study/scheduler_repro/test_scheduler.py)
- [单卡异步 swap 实验报告](study/async_swap/实验报告.md)
- [真实 CUDA 集成测试](study/async_swap/test_async_scheduler_cuda.py)
- [真实模型 A/B 实验报告](study/e2e_async_swap/实验报告.md)
- [真实模型 A/B 基准](study/e2e_async_swap/run_compare.py)
- [设计文档](设计文档.md)

使用 uv 安装环境并运行 CPU 回归测试：

```bash
uv sync
uv run python study/scheduler_repro/test_scheduler.py
```

真实 CUDA 集成测试入口是 study/async_swap/test_async_scheduler_cuda.py。构造 LLM 时传入 async_swap=True 即可启用异步路径。该选项当前只支持 tensor_parallel_size=1，默认关闭。

## 贡献边界与后续方向

当前项目包含模型执行、前缀缓存、张量并行、CUDA Graph、CPU swap、CUDA 拷贝，以及调度复现测试和中文分析。后续修复和优化将以本项目的独立提交记录。

单卡异步 swap 的状态和数据正确性已经通过测试。下一步是在真实模型与受控并发负载下比较同步和异步路径；当前没有真实模型端到端性能收益结论。

## 数据与版本管理

公开仓库保存源码、可复现测试及报告。模型、数据集、原始请求、性能采集文件与带本机路径的日志留在本地。Python 包名暂保留 `nanovllm`，避免为项目命名而修改已有导入。
