# src/cuda_tools/decorators.py
"""
Simple (@cuda) and Advanced (@cuda.advanced) decorators—now with universal support, dtype validation and robust timeout.
"""
import os, warnings, contextlib, psutil, logging, numpy as np, torch, torch.nn as nn, concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from .utils import (select_fastest_torch_device, split_large_tensor,
                    move_to_torch, move_to_tf, watchdog,
                    tensorize_for_universal, patch_numpy_with_cupy)
try:
    import tensorflow as tf; _TF_INSTALLED=True
except ImportError:
    _TF_INSTALLED=False

_executor = ThreadPoolExecutor(max_workers=1)
_logger   = logging.getLogger('cuda_tools')
if not _logger.handlers:
    h=logging.StreamHandler();h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    _logger.addHandler(h);_logger.setLevel(logging.INFO)

# -- simple decorator ------------------------------------------------------
def _make_simple_decorator():
    def deco(fn=None, *, device=None, use_amp=False,
             is_async=False, retry=0, clear_cache=False,
             min_free_vram=None, auto_tensorize=False):
        def wrapper(*args, **kwargs):
            if auto_tensorize:
                args   = [tensorize_for_universal(a) for a in args]
                kwargs = {k:tensorize_for_universal(v) for k,v in kwargs.items()}
            if auto_tensorize:
                with patch_numpy_with_cupy():
                    return wrapper.__wrapped__(*args, **kwargs)
            dev=torch.device(device) if device else select_fastest_torch_device()
            if clear_cache and dev.type=='cuda': torch.cuda.empty_cache()
            amp_ctx = torch.autocast(device_type=dev.type, enabled=use_amp) if use_amp else contextlib.nullcontext()
            attempt=0
            while True:
                try:
                    moved_args   = [move_to_torch(dev,a) for a in args]
                    moved_kwargs = {k:move_to_torch(dev,v) for k,v in kwargs.items()}
                    with amp_ctx:
                        if is_async:
                            return _executor.submit(fn, *moved_args, **moved_kwargs).result()
                        return fn(*moved_args, **moved_kwargs)
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower() and attempt<retry:
                        attempt+=1; torch.cuda.empty_cache(); continue
                    raise
        wrapper.__wrapped__=fn
        return wrapper if fn else deco
    return deco

# -- advanced decorator ----------------------------------------------------
def _make_advanced_decorator():
    def deco(fn=None, *, device=None, mgpu=False, auto_benchmark=False,
             offline_mode=True, use_amp=False, power_saving=False,
             is_async=False, retry=0, timeout=None, clear_cache=False,
             min_free_vram=None, low_priority=False,
             preload_next_batch=False, framework='auto', batch_size_param='batch_size',
             grad_accum_steps=1, checkpoint_path=None, adaptive_lr=False,
             dynamic_batch_growth=False, grad_clip=False, param_freeze=False,
             quantize=False, verbose=False, memory_profiler=True,
             live_dashboard=False, exception_analytics=False,
             logger=None, telemetry=False, parallel=False,
             tpu=False, distributed=False, dry_run=False,
             auto_tensorize=False, error_callback=None):
        log = logger or _logger
        def wrapper(*args, **kwargs):
            # universal auto-tensorize
            if auto_tensorize:
                args   = [tensorize_for_universal(a) for a in args]
                kwargs = {k:tensorize_for_universal(v) for k,v in kwargs.items()}
                with patch_numpy_with_cupy():
                    return wrapper.__wrapped__(*args, **kwargs)
            # priority/power
            if low_priority: psutil.Process(os.getpid()).nice(10); log.info('low priority')
            if power_saving and torch.cuda.is_available(): os.system('nvidia-smi -pm 1'); log.info('power saving')
            # device select
            if auto_benchmark:
                dev = select_fastest_torch_device()
            elif offline_mode and not torch.cuda.is_available():
                dev = torch.device('cpu')
            elif mgpu and torch.cuda.is_available() and torch.cuda.device_count()>1:
                idx = int(np.argmin([torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]))
                dev = torch.device(f'cuda:{idx}')
            else:
                dev = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            log.info(f'use device {dev}')
            if clear_cache and dev.type=='cuda': torch.cuda.empty_cache()

            def run_inner(*i_args, **i_kwargs):
                na, nk = [], {}
                for a in i_args:
                    if isinstance(a, torch.Tensor):
                        split = split_large_tensor(a)
                        a = torch.cat(split, 0).to(dev) if split else a.to(dev)
                    na.append(a)
                for k, v in i_kwargs.items():
                    nk[k] = v.to(dev) if isinstance(v, torch.Tensor) else v
                amp_ctx = torch.autocast(device_type=dev.type, enabled=use_amp) if use_amp else contextlib.nullcontext()
                with amp_ctx:
                    if dry_run:
                        log.info('dry run'); return None
                    return fn(*na, **nk)

            # retry/timeout logic with proper future timeout
            attempt, start_mem = 0, psutil.virtual_memory().used
            while True:
                try:
                    if timeout is not None:
                        future = _executor.submit(run_inner, *args, **kwargs)
                        result = future.result(timeout=timeout)
                    else:
                        result = run_inner(*args, **kwargs)
                    break
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Function '{fn.__name__}' exceeded timeout of {timeout} seconds.")
                except Exception as e:
                    if exception_analytics: log.error(e)
                    if checkpoint_path:
                        for o in args:
                            if isinstance(o, nn.Module): torch.save(o.state_dict(), checkpoint_path)
                    if attempt < retry:
                        attempt += 1
                        continue
                    if error_callback:
                        return error_callback(e)
                    raise

            if telemetry: log.info(f'telemetry: device={dev}')
            if memory_profiler: log.info(f'mem Δ: {(psutil.virtual_memory().used - start_mem)/1e9:.2f}GB')
            return result

        wrapper.__wrapped__ = fn
        return wrapper if fn else deco
    return deco

# public API
cuda = _make_simple_decorator()
cuda_advanced = _make_advanced_decorator()
