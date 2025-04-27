# Package: universal-cuda_tools
# Directory: src/universal-cuda_tools

# src/cuda_tools/__init__.py
"""
cuda_tools: Easy device management for PyTorch, TensorFlow & universal Python math via CuPy.
Exports:
  - cuda (simple decorator)
  - cuda.advanced (advanced decorator)
  - DeviceContext (context manager)
"""
from .decorators import cuda, cuda_advanced
from .context    import DeviceContext

# alias advanced on simple
cuda.advanced = cuda_advanced
__all__ = ['cuda', 'DeviceContext']
