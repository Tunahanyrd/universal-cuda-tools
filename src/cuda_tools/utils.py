# src/cuda_tools/utils.py
"""
Utility functions: device selection, tensor moves, split, watchdog, and universal tensorize via CuPy.
"""
import sys, time
import numpy as np
import torch
try:
    import cupy as cp
    _CP_INSTALLED = True
except ImportError:
    cp = None
    _CP_INSTALLED = False
try:
    import tensorflow as tf
    _TF_INSTALLED = True
except ImportError:
    _TF_INSTALLED = False

import contextlib

def select_fastest_torch_device() -> torch.device:
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def split_large_tensor(tensor: torch.Tensor, max_bytes: int = 1<<27):
    nbytes = tensor.numel() * tensor.element_size()
    return torch.chunk(tensor, 2, dim=0) if nbytes>max_bytes else None


def move_to_torch(device: torch.device, obj):
    if isinstance(obj, torch.Tensor): return obj.to(device)
    if isinstance(obj, np.ndarray):   return torch.from_numpy(obj).to(device)
    if _TF_INSTALLED and isinstance(obj, tf.Tensor):
        arr = obj.cpu().numpy()
        return torch.from_numpy(arr).to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_torch(device, o) for o in obj)
    if isinstance(obj, dict):
        return {k: move_to_torch(device, v) for k, v in obj.items()}
    return obj


def move_to_tf(tf_device: str, obj):
    if not _TF_INSTALLED: return obj
    if isinstance(obj, tf.Tensor): return obj
    if isinstance(obj, torch.Tensor):
        arr = obj.cpu().numpy()
        with tf.device(tf_device): return tf.convert_to_tensor(arr)
    if isinstance(obj, np.ndarray):
        with tf.device(tf_device): return tf.convert_to_tensor(obj)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_tf(tf_device, o) for o in obj)
    if isinstance(obj, dict):
        return {k: move_to_tf(tf_device, v) for k, v in obj.items()}
    return obj


def watchdog(fn, timeout: float, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    if timeout and time.time()-start>timeout:
        raise TimeoutError(f"Timeout: {time.time()-start:.2f}s > {timeout}s")
    return result


def tensorize_for_universal(obj):
    """Convert int/float/list/ndarray → CuPy array if available."""
    if not _CP_INSTALLED: return obj
    if isinstance(obj, (int, float)): return cp.array(obj)
    if isinstance(obj, np.ndarray):   return cp.asarray(obj)
    if isinstance(obj, (list, tuple)): return cp.asarray(obj)
    if _CP_INSTALLED and isinstance(obj, cp.ndarray): return obj
    return obj

@contextlib.contextmanager
def patch_numpy_with_cupy():
    orig = sys.modules.get('numpy')
    sys.modules['numpy'] = cp
    try:
        yield
    finally:
        if orig is not None: sys.modules['numpy']=orig
        else: del sys.modules['numpy']
