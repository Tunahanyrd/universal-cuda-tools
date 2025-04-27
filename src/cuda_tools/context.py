# src/cuda_tools/context.py
"""
DeviceContext: context manager for device + AMP + universal numpy→cupy patch.
"""
import contextlib, torch
from .utils import patch_numpy_with_cupy, select_fastest_torch_device

class DeviceContext:
    def __init__(self, *, device: str=None, use_amp: bool=False,
                 clear_cache: bool=False, verbose: bool=False,
                 universal: bool=False):
        self.device = torch.device(device) if device else select_fastest_torch_device()
        self.use_amp = use_amp
        self.clear_cache = clear_cache
        self.verbose = verbose
        self.universal = universal
        self._np_patch = None
        self._ctx = None

    def __enter__(self):
        if self.clear_cache and self.device.type=='cuda': torch.cuda.empty_cache()
        if self.universal: self._np_patch = patch_numpy_with_cupy(); self._np_patch.__enter__()
        self._ctx = (torch.autocast(device_type=self.device.type, enabled=True)
                     if self.use_amp and self.device.type=='cuda' else contextlib.nullcontext())
        if self.verbose:
            print(f"[DeviceContext] Enter: device={self.device}, AMP={self.use_amp}, universal={self.universal}")
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._ctx.__exit__(exc_type, exc_val, exc_tb)
        if self.universal: self._np_patch.__exit__(exc_type, exc_val, exc_tb)
        if self.verbose:
            print(f"[DeviceContext] Exit: device={self.device}")
