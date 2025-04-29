# Package: universal-cuda_tools
# Directory: src/cuda_tools

  # src/cuda_tools/__init__.py
"""
cuda_tools: Easy device management for PyTorch, TensorFlow & universal Python math via CuPy.
Exports:
  - cuda (simple decorator)
  - cuda.advanced (advanced decorator)
  - DeviceContext (context manager)
"""
from .decorators import cuda, cuda_advanced
from .context import DeviceContext

__all__ = ["cuda", "cuda_advanced", "DeviceContext"]
