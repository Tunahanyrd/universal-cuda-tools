# src/cuda_tools/utils.py
"""
Utility functions: device selection, tensor moves, split, watchdog, and universal tensorize via CuPy.
"""
import contextlib
import warnings

import numpy as np
import torch

# TensorFlow opsiyonel
try:
    import tensorflow as tf
    _TF_INSTALLED = True
except ImportError:
    _TF_INSTALLED = False

@contextlib.contextmanager
def patch_numpy_with_cupy():
    """
    Context manager: `import numpy as np` ile CuPy kullanır.
    Eğer CuPy yüklü değilse, normal numpy işlemlerine devam eder.
    """
    import numpy as _np
    try:
        import cupy as _cp
    except ImportError:
        # fallback: boş context
        yield
        return

    orig_array = _np.array
    _np.array = _cp.array
    try:
        yield
    finally:
        _np.array = orig_array

def tensorize_for_universal(obj, device):
    """
    int, float, list, tuple, np.ndarray, torch.Tensor, tf.Tensor → torch.Tensor.
    Supports pure int/float, NumPy scalars, NumPy arrays, lists, tensors.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (np.integer, int, float)):  # <<< Burayı ekledim
        return torch.tensor(obj, device=device)
    if isinstance(obj, (list, tuple)):
        return torch.tensor(obj, device=device)
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            raise TypeError("Cannot auto-tensorize numpy array with dtype=object.")
        return torch.from_numpy(obj).to(device)
    if _TF_INSTALLED and isinstance(obj, tf.Tensor):
        warnings.warn("TensorFlow Tensor detected; converting via numpy.", UserWarning)
        return torch.from_numpy(obj.cpu().numpy()).to(device)
    return obj

def move_to_torch(device, obj):
    """
    torch.Tensor ise cihaza taşır, numpy.ndarray ise tensorize_for_universal ile dönüştürür.
    Diğerlerinde objeyi olduğu gibi döner.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, np.ndarray):
        if obj.dtype == np.object_:
            raise TypeError("Cannot move numpy array with dtype=object to torch.")
        return torch.from_numpy(obj).to(device)
    return obj

