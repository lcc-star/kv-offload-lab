# KV Offload Lab

面向 LLM 推理系统的 KV Cache 换出、调度与正确性实验项目，由 lcc-star 独立维护。

项目代码由 Nano-vLLM 及其 KV Cache swap 实现衍生而来，现由 lcc-star 独立维护。原始版权与授权条款保留在 [MIT 许可证](LICENSE) 中。

## 当前工作

- 阅读并分析 waiting / running / swapped 队列、GPU/CPU block 管理和换入换出流程。
- 用真实 Scheduler、BlockManager、Sequence 完成无模型 CPU 复现实验。
- 已复现换入计数导致空批次断言、decode 批次缩小的问题，并验证恢复 KV 与下一步 decode 空间条件的区别。
- 当前尚未修改核心调度逻辑；5 项测试中 2 项正常对照通过、3 项失败。失败用例用于后续修复验证，不表示实现已通过验收。

## 学习与实验入口

- [CPU 调度复现实验报告](study/scheduler_repro/REPORT.md)
- [对应测试代码](study/scheduler_repro/test_scheduler.py)
- [设计文档](设计文档.md)

在安装 numpy、xxhash、transformers 的 Python 环境中，从仓库根目录执行：

```bash
python study/scheduler_repro/test_scheduler.py
```

这个实验不加载模型、不编译 CUDA 扩展、不搬运 GPU KV；当前版本预期出现 3 项失败并返回非零退出码。它只能验证调度与 block 元数据，不能证明实际 KV 传输或生成输出正确。

## 贡献边界与后续方向

当前项目包含模型执行、前缀缓存、张量并行、CUDA Graph、CPU swap、CUDA 拷贝，以及调度复现测试和中文分析。后续修复和优化将以本项目的独立提交记录。

后续先修复并验证调度计数问题，再研究 swap-in 资源预留、共享 block 处理、张量布局与拷贝正确性，最后开展受控性能评估。当前没有经过本项目独立复现的性能收益结论。

## 数据与版本管理

公开仓库保存源码、可复现测试及报告。模型、数据集、原始请求、性能采集文件与带本机路径的日志留在本地。Python 包名暂保留 `nanovllm`，避免为项目命名而修改已有导入。
