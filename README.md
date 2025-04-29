# universal-cuda-tools

[![PyPI version](https://badge.fury.io/py/universal-cuda-tools.svg)](https://pypi.org/project/universal-cuda-tools/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[GitHub](https://github.com/Tunahanyrd/universal-cuda-tools)

## What is this library?

**universal-cuda-tools** is a lightweight Python package that makes it trivial to:

- Move your functions and data between CPU and GPU
- Automatically convert plain Python, NumPy or CuPy inputs into `torch.Tensor`
- Enforce minimum free GPU memory and fall back to CPU on out-of-memory
- Apply mixed-precision (AMP) with a single flag
- Add timeouts, retries, telemetry and live dashboard metrics
- Run in “dry-run” mode for safe testing
- Use a context manager for block-scoped device/precision control

This toolkit sits on top of PyTorch (and optionally TensorFlow) but works with any Python code, so you can accelerate pure-Python or NumPy-based logic as well as deep-learning models.

## Why use it?

- **Simplicity**  
  One decorator (`@cuda` or `@cuda_advanced`) or one context manager (`DeviceContext`) is all you need.

- **Safety**  
  Automatic GPU→CPU fallback on OOM, plus optional timeouts and retries.

- **Versatility**  
  Supports PyTorch tensors, TensorFlow tensors, NumPy arrays, Python lists, scalars, and more.

- **Observability**  
  Built-in telemetry, memory profiling, and a rudimentary live dashboard to understand performance.

- **Dry-run & Testing**  
  Skip actual execution for fast, side-effect-free tests.

## Installation

```bash
pip install universal-cuda-tools
```

or from source:

```bash
git clone https://github.com/Tunahanyrd/universal-cuda-tools.git
cd universal-cuda-tools
python -m build
pip install dist/universal_cuda_tools-*.whl
```

## Quick Start

### 1. Basic decorator

```python
from cuda_tools import cuda

@cuda(device="cuda", auto_tensorize=True, to_list=True)
def add(a, b):
    return a + b

print(add(3, 4))            # 7
print(add([1,2], [3,4]))    # [4,6]
```

### 2. Advanced decorator

```python
from cuda_tools import cuda_advanced

@cuda_advanced(timeout=1.0, retry=1, use_amp=True, telemetry=True)
def train_step(model, x, y):
    pred = model(x)
    loss = (pred – y).square().mean()
    loss.backward()
    return loss.item()
```

- `timeout`: seconds before raising `TimeoutError`  
- `retry`: number of retry attempts on exception  
- `use_amp`: enable `torch.autocast` (mixed precision)  
- `telemetry`: log execution time and device

### 3. Context manager

```python
from cuda_tools.context import DeviceContext
import numpy as np

with DeviceContext(device='cuda', auto_tensorize=True, use_amp=True):
    x = tensorize_for_universal([1,2,3])
    y = tensorize_for_universal(np.array([4,5,6]))
    z = x + y        # runs on GPU with AMP
```

## Core API

### `@cuda(...)`

A minimal decorator with parameters:

- `device`: `'cuda'`/`'cpu'`/`None` (auto)  
- `verbose`: `True`/`False`  
- `clear_cache`: empty CUDA cache before run  
- `retry`: retry count on failure  
- `min_free_vram`: GB threshold for GPU memory  
- `auto_tensorize`: convert inputs to `torch.Tensor`  
- `to_list`: return Python list from tensor

### `@cuda_advanced(...)`

All of `@cuda` plus:

- `timeout`: seconds before abort  
- `use_amp`: mixed precision via `torch.autocast`  
- `mgpu`: pick least-used GPU  
- `error_callback`: custom handler on exception  
- `telemetry`: log time and device  
- `memory_profiler`: log memory delta  
- `live_dashboard`: simple call count & total time  
- `dry_run`: skip execution, return `None`

### `DeviceContext(...)`

Block-scoped device and precision control:

```python
DeviceContext(device='cuda', use_amp=False, verbose=False, auto_tensorize=False)
```

### Utilities (`cuda_tools.utils`)

- `tensorize_for_universal(obj, device)`: universal tensor conversion  
- `move_to_torch(device, obj)`: move torch/NumPy arrays to device  
- `patch_numpy_with_cupy()`: redirect NumPy calls to CuPy

## Examples and Recipes

See the [official documentation site](https://funeralcs.github.io/posts/cuda_tools_dokuman/) for full examples, neural-network training recipes, and performance benchmarks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
