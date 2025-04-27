# universal-cuda-tools

Universal device management for Python, PyTorch, TensorFlow and NumPy. Supports automatic device selection, mixed precision, retry, timeout, telemetry, checkpointing, and full GPU acceleration even for pure Python math via CuPy.

Provides decorators and context managers for:
- Automatic device selection
- Mixed precision (AMP)
- Memory profiling
- Timeout watchdog
- Telemetry
- TPU & Distributed (experimental)

## Installation
```bash
pip install universal-cuda-tools
```
# Example Usage
## Simple Decorator
```python
from cuda_tools import cuda

@cuda(device='cuda:0', use_amp=True)
def infer(model, x):
    return model(x)
```
## Advanced decorator
```python
from cuda_tools import cuda

@cuda.advanced(auto_benchmark=True, use_amp=True, retry=2, timeout=300)
def train_step(model, x, y):
    # training logic
    loss = model(x).mean()
    return loss
```
## Context manager
```python
from cuda_tools.context import DeviceContext

with DeviceContext(device='cuda:0', use_amp=True):
    output = model(input)
``` 
## Universal Python Code on CUDA

If you want *any* pure-Python or NumPy-based math to run on GPU, use `auto_tensorize=True`:

```python
from cuda_tools import cuda

@cuda(auto_tensorize=True, device='cuda:0')
def pure_python_math(x, y):
    # x,y can be int, float, list or numpy array
    import numpy as np
    a = np.array(x) * np.array(y)
    return np.sin(a) + 5

print(pure_python_math([1,2,3], 10))
```
_Or with context manager:_
```python
from cuda_tools.context import DeviceContext

with DeviceContext(device='cuda:0', universal=True):
    import numpy as np
    result = np.linspace(0,5,1000)**2  # will be computed with CuPy under the hood
```
*src/universal_cuda_tools/decorators.py*:Simple and advanced decorators for device management.
*src/universal_cuda_tools/context.py*: Context manager for device and precision setup/teardown.
*src/universal_cuda_tools/utils.py*: Utility functions for device management.








