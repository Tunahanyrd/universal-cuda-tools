# src/cuda_tools/context.py
"""
DeviceContext: context manager for device + AMP + universal numpy→cupy patch.
"""
import contextlib
import logging
import torch
import warnings

from .utils import patch_numpy_with_cupy, tensorize_for_universal

_logger = logging.getLogger(__name__)

class DeviceContext:
    """
    with DeviceContext(...):
        # Bu blokta:
        #  • device seçimi (cuda/CPU)
        #  • AMP (isteğe bağlı)
        #  • auto_tensorize (int/float/np.*/list/tuple → torch.Tensor)
        #    ve CuPy array → torch.Tensor dönüşümü
        #  • clear_cache (CUDA cache temizleme)
        #  • verbose log
    geçerli olur.
    """
    def __init__(self,
                 device='cuda',
                 use_amp=False,
                 verbose=False,
                 auto_tensorize=False):
        self.device = device
        self.use_amp = use_amp
        self.verbose = verbose
        self.auto_tensorize = auto_tensorize
        self._patch = None
        self._amp = None

    def __enter__(self):
        # Seçilen cihaz
        dev = torch.device(self.device)
        if self.verbose:
            _logger.info(f"[Context] use device {dev}")

        # CUDA cache temizle
        if dev.type == 'cuda':
            torch.cuda.empty_cache()

        # AMP başlat
        if self.use_amp:
            self._amp = torch.autocast(device_type=dev.type, enabled=True)
            self._amp.__enter__()

        # NumPy → CuPy patch ve hemen ardından CuPy array → Torch tensorize
        if self.auto_tensorize:
            # 1) np.array → cp.array
            self._patch = patch_numpy_with_cupy()
            self._patch.__enter__()

            # 2) otomatik tensorize: global __builtins__ içine basit wrapper ekleyelim
            #    (int, float, list, tuple, np.* → torch.Tensor(dev))
            builtins = __builtins__
            self._orig_tensorize = getattr(builtins, 'tensorize_for_context', None)
            def tensorize_for_context(obj):
                return tensorize_for_universal(obj, dev)
            setattr(builtins, 'tensorize_for_context', tensorize_for_context)

        return dev

    def __exit__(self, exc_type, exc_val, exc_tb):
        # AMP stop
        if self._amp:
            self._amp.__exit__(exc_type, exc_val, exc_tb)

        # CuPy patch geri al
        if self._patch:
            self._patch.__exit__(exc_type, exc_val, exc_tb)

            # builtins temizle
            builtins = __builtins__
            if hasattr(self, '_orig_tensorize'):
                if self._orig_tensorize is not None:
                    setattr(builtins, 'tensorize_for_context', self._orig_tensorize)
                else:
                    delattr(builtins, 'tensorize_for_context')

        if self.verbose:
            _logger.info("[Context] exit")

        # İstisnayı tekrar fırlat
        return False



