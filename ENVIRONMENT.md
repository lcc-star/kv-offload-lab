# 开发环境

## 当前验证环境

- Python 3.12.13
- PyTorch 2.9.1+cu128
- Transformers 4.57.3
- FlashAttention 2.8.3.post1
- CUDA Toolkit 12.9
- NVIDIA A800 80GB PCIe
- uv 0.11.8

## 创建环境

在项目根目录执行：

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.9.1

uv pip install \
  numpy \
  transformers==4.57.3 \
  xxhash \
  ninja \
  packaging \
  psutil \
  setuptools \
  wheel

MAX_JOBS=4 uv pip install --no-build-isolation flash-attn==2.8.3.post1
uv pip install --no-deps --editable .
```

FlashAttention 在当前组合下没有可直接使用的预编译 wheel，会在首次安装时编译 CUDA 源码。A800 机器上的首次构建约需 50 分钟，`MAX_JOBS=4` 用于限制并行任务数和内存占用。

## 验证环境

```bash
source .venv/bin/activate
uv pip check

python -c "import torch, transformers, flash_attn, xxhash; \
print(torch.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
print(transformers.__version__, flash_attn.__version__, xxhash.VERSION)"
```

运行 swap kernel 复现实验：

```bash
CUDA_VISIBLE_DEVICES=0 python study/swap_kernel_repro/test_swap_kernel.py
```

必须先激活虚拟环境。仅直接执行 `.venv/bin/python` 不会把 `.venv/bin` 加入 `PATH`，PyTorch 编译 CUDA 扩展时会因此找不到 Ninja。
