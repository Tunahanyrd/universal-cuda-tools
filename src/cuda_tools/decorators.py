# src/cuda_tools/decorators.py
"""
Simple (@cuda) and Advanced (@cuda.advanced) decorators—now with universal support, dtype validation,
robust timeout, retry on any exception, result detensorization (to_list), and vram cleanup.
"""
import contextlib
import logging
import warnings
import psutil
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time
import numpy as np
import torch

from .utils import (
    tensorize_for_universal,
    move_to_torch,
    patch_numpy_with_cupy,
)

_logger = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=1)


def cuda(func=None, *,
         device=None,
         verbose=True,
         clear_cache=False,
         retry=0,
         min_free_vram=None,
         auto_tensorize=False,
         to_list=False):
    """
    Basit decorator:
      - device seçimi (CUDA/CPU)
      - min_free_vram kontrolü
      - verbose log
      - clear_cache
      - retry + VRAM-fallback
      - auto_tensorize
      - memory profiling
      - to_list
    """
    def decorator(fn):
        def wrapper(*args, **kwargs):
            # Device seçimi
            dev = torch.device(device) if device else torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )

            # VRAM boş alan kontrolü
            if min_free_vram is not None and dev.type == 'cuda':
                props = torch.cuda.get_device_properties(dev)
                total = props.total_memory
                reserved = torch.cuda.memory_reserved(dev)
                free = total - reserved
                free_gb = free / (1024**3)
                if free_gb < min_free_vram:
                    raise RuntimeError(
                        f"Insufficient GPU VRAM: {free_gb:.2f} GB free, "
                        f"{min_free_vram:.2f} GB required."
                    )

            if verbose:
                _logger.info(f"[cuda] use device {dev}")

            if clear_cache and dev.type == 'cuda':
                torch.cuda.empty_cache()

            # Auto-tensorize
            if auto_tensorize:
                args   = [tensorize_for_universal(a, dev) for a in args]
                kwargs = {k: tensorize_for_universal(v, dev) for k, v in kwargs.items()}

            # Argümanları cihaza taşı
            moved_args   = [move_to_torch(dev, a) for a in args]
            moved_kwargs = {k: move_to_torch(dev, v) for k, v in kwargs.items()}

            attempt = 0
            start_mem = psutil.virtual_memory().used

            while True:
                try:
                    result = fn(*moved_args, **moved_kwargs)
                    break
                except Exception as e:
                    # retry mekanizması (her türlü exception için)
                    if attempt < retry:
                        attempt += 1
                        if clear_cache and dev.type == 'cuda':
                            torch.cuda.empty_cache()
                        if verbose:
                            _logger.warning(f"[cuda] retry {attempt} after {e}")
                        continue
                    # OOM fallback (GPU → CPU)
                    msg = str(e).lower()
                    if isinstance(e, RuntimeError) and 'out of memory' in msg and dev.type == 'cuda':
                        if verbose:
                            _logger.warning("[cuda] CUDA OOM, falling back to CPU")
                        dev = torch.device('cpu')
                        moved_args   = [move_to_torch(dev, a) for a in args]
                        moved_kwargs = {k: move_to_torch(dev, v) for k, v in kwargs.items()}
                        result = fn(*moved_args, **moved_kwargs)
                        break
                    # başka hataysa yükselt
                    raise

            if verbose:
                delta = (psutil.virtual_memory().used - start_mem) / 1e9
                _logger.info(f"[cuda] mem Δ: {delta:.2f}GB")

            # to_list desteği
            if to_list and isinstance(result, torch.Tensor):
                return result.tolist()

            return result

        return wrapper

    return decorator(func) if func else decorator


def cuda_advanced(func=None, *,
                  device=None,
                  verbose=True,
                  clear_cache=False,
                  retry=0,
                  min_free_vram=None,
                  auto_tensorize=False,
                  to_list=False,
                  timeout=None,
                  use_amp=False,
                  mgpu=False,
                  error_callback=None,
                  telemetry=False,
                  memory_profiler=True,
                  live_dashboard=False,
                  dry_run=False):
    """
    Gelişmiş decorator:
      - Basit dekoratörün tüm özellikleri
      - timeout
      - AMP (torch.autocast)
      - multi-GPU (mgpu)
      - error_callback
      - telemetry + memory_profiler
      - live_dashboard (basit metrik toplama)
      - dry_run
      - to_list
    """
    def decorator(f):
        # Basit live_dashboard metrik deposu (örnek)
        dashboard_stats = {'calls': 0, 'total_time': 0.0}

        def wrapper(*args, **kwargs):
            # Dry-run modu
            if dry_run:
                if verbose:
                    _logger.info(f"[cuda_advanced] dry_run enabled, skipping '{f.__name__}'")
                return None

            # Device seçimi (multi-GPU veya tek)
            if mgpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
                usages = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
                idx = int(np.argmin(usages))
                dev = torch.device(f'cuda:{idx}')
            else:
                dev = torch.device(device) if device else torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu'
                )

            # VRAM boş alan kontrolü
            if min_free_vram is not None and dev.type == 'cuda':
                props = torch.cuda.get_device_properties(dev)
                total = props.total_memory
                reserved = torch.cuda.memory_reserved(dev)
                free = total - reserved
                free_gb = free / (1024**3)
                if free_gb < min_free_vram:
                    raise RuntimeError(
                        f"Insufficient GPU VRAM: {free_gb:.2f} GB free, "
                        f"{min_free_vram:.2f} GB required."
                    )

            if verbose:
                _logger.info(f"[cuda_advanced] use device {dev}")
            if clear_cache and dev.type == 'cuda':
                torch.cuda.empty_cache()

            # live_dashboard stats update
            if live_dashboard:
                dashboard_stats['calls'] += 1

            # Auto-tensorize + numpy patch
            patch_ctx = patch_numpy_with_cupy() if auto_tensorize else contextlib.nullcontext()
            if auto_tensorize:
                args   = [tensorize_for_universal(a, dev) for a in args]
                kwargs = {k: tensorize_for_universal(v, dev) for k, v in kwargs.items()}

            # AMP context
            amp_ctx = torch.autocast(device_type=dev.type, enabled=use_amp) if use_amp else contextlib.nullcontext()

            attempt = 0
            start_mem = psutil.virtual_memory().used
            start_time = time.time() if telemetry or live_dashboard else None

            while True:
                try:
                    with patch_ctx, amp_ctx:
                        if timeout:
                            future = _executor.submit(f, *args, **kwargs)
                            result = future.result(timeout=timeout)
                        else:
                            result = f(*args, **kwargs)
                    break
                except FuturesTimeoutError:
                    raise TimeoutError(f"Function '{f.__name__}' timed out after {timeout}s")
                except Exception as e:
                    # error_callback
                    if error_callback:
                        error_callback(e)
                        return None
                    if attempt < retry:
                        attempt += 1
                        if clear_cache and dev.type == 'cuda':
                            torch.cuda.empty_cache()
                        if verbose:
                            _logger.warning(f"[cuda_advanced] retry {attempt} after {e}")
                        continue
                    # OOM fallback
                    msg = str(e).lower()
                    if isinstance(e, RuntimeError) and 'out of memory' in msg and dev.type == 'cuda':
                        if verbose:
                            _logger.warning("[cuda_advanced] CUDA OOM, falling back to CPU")
                        dev = torch.device('cpu')
                        with contextlib.nullcontext(), (torch.autocast(device_type=dev.type, enabled=use_amp) if use_amp else contextlib.nullcontext()):
                            result = f(*args, **kwargs)
                        break
                    raise

            # telemetry & memory profiling
            if telemetry:
                elapsed = time.time() - start_time
                _logger.info(f"[cuda_advanced] telemetry: device={dev}, time={elapsed:.4f}s")
            if memory_profiler and verbose:
                delta = (psutil.virtual_memory().used - start_mem) / 1e9
                _logger.info(f"[cuda_advanced] mem Δ: {delta:.2f}GB")

            # live_dashboard log
            if live_dashboard and verbose:
                total_elapsed = time.time() - start_time
                dashboard_stats['total_time'] += total_elapsed
                _logger.info(f"[cuda_advanced] live_dashboard: calls={dashboard_stats['calls']}, total_time={dashboard_stats['total_time']:.4f}s")

            # to_list desteği
            if to_list and isinstance(result, torch.Tensor):
                return result.tolist()

            return result

        return wrapper
    return decorator(func) if func else decorator


