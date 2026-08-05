"""Array-backend seam: let every transform compute with the array module of *its input
block* rather than a hardcoded ``numpy`` / ``scipy``. On CPU the module is numpy and
behaviour is identical; when a region's data is a CuPy array (the GPU device seam, added
separately) the same transforms run on the GPU.

cupy is detected by module name so importing this file never requires cupy to be present.
Critically, this lets us avoid ``numpy.asarray(gpu_array)`` calls, which would silently
copy device data back to the host mid-pipeline.
"""

import threading
from contextlib import contextmanager

import numpy as np


def _is_cupy(x) -> bool:
    return type(x).__module__.split(".")[0] == "cupy"


# --------------------------------------------------------------------------- #
# Execution-device context: the terminal (compute/io.write) sets the default
# device that inherit-ops (device=None) run on; an op's explicit device overrides.
# --------------------------------------------------------------------------- #

_ctx = threading.local()


def current_device() -> str:
    return getattr(_ctx, "device", "cpu")


@contextmanager
def device_context(device):
    """Set the execution device for inherit-ops within the block ('cpu' default)."""
    prev = getattr(_ctx, "device", None)
    _ctx.device = device or "cpu"
    try:
        yield
    finally:
        if prev is None:
            _ctx.__dict__.pop("device", None)
        else:
            _ctx.device = prev


def resolve_device(op_device) -> str:
    """An op's explicit device if set, else the current execution context."""
    return op_device if op_device is not None else current_device()


def to_device(x, device):
    """Move array ``x`` to ``device`` ('cpu' | 'cuda' | 'cuda:N'). Scalars pass through;
    already-on-target arrays are returned untouched (no copy)."""
    if np.isscalar(x) or x is None:
        return x
    dev = device or "cpu"
    if dev == "cpu":
        return asnumpy(x) if _is_cupy(x) else x
    import cupy
    if dev.startswith("cuda:"):
        with cupy.cuda.Device(int(dev.split(":", 1)[1])):
            return cupy.asarray(x)
    return x if _is_cupy(x) else cupy.asarray(x)


def array_namespace(*arrays):
    """Return the array module (``numpy`` or ``cupy``) for the given operands. Scalars are
    ignored; if any operand is a CuPy array, cupy is returned (and imported lazily)."""
    for a in arrays:
        if _is_cupy(a):
            import cupy
            return cupy
    return np


def ndimage_namespace(x):
    """Return ``scipy.ndimage`` or ``cupyx.scipy.ndimage`` to match ``x``'s device."""
    if _is_cupy(x):
        import cupyx.scipy.ndimage as cndi
        return cndi
    import scipy.ndimage as ndi
    return ndi


def asnumpy(x):
    """Host numpy array for ``x`` (D2H copy if it lives on the GPU)."""
    if _is_cupy(x):
        import cupy
        return cupy.asnumpy(x)
    return np.asarray(x)


def asnumpy_pinned(x):
    """Like :func:`asnumpy`, but stages the D2H copy into pinned (page-locked) host memory.

    Pinned transfers are ~1.5-2x faster and, more importantly under many concurrent
    writer threads, avoid contention on CuPy's shared pinned staging buffer (measured
    ~7-18% faster GPU io.write on heavy ops). Used for the streamed region write path;
    falls back to :func:`asnumpy` off-GPU or if pinned allocation is unavailable."""
    if not _is_cupy(x):
        return np.asarray(x)
    try:
        import cupyx
        out = cupyx.empty_pinned(x.shape, dtype=x.dtype)
        x.get(out=out)   # D2H into pinned host on the current stream
        return out
    except Exception:
        import cupy
        return cupy.asnumpy(x)


def is_gpu_array(x) -> bool:
    return _is_cupy(x)


def xp_for_device(device):
    """Array module to *create* new arrays on ``device`` ('cpu'->numpy, 'cuda'->cupy).
    Used by generative sources (creation ops) that have no input block to dispatch on."""
    if (device or "cpu") == "cpu":
        return np
    import cupy
    return cupy


def new_stream(device):
    """A fresh non-blocking CUDA stream for ``device`` ('cuda'/'cuda:N'), else None.
    Giving each worker thread its own stream lets regions overlap on the GPU (one
    region's compute runs while another's H2D/D2H copies), instead of serialising on
    the shared default stream."""
    dev = device or "cpu"
    if dev == "cpu":
        return None
    try:
        import cupy
        if dev.startswith("cuda:"):
            with cupy.cuda.Device(int(dev.split(":", 1)[1])):
                return cupy.cuda.Stream(non_blocking=True)
        return cupy.cuda.Stream(non_blocking=True)
    except Exception:
        return None


@contextmanager
def use_stream(stream):
    """Activate ``stream`` as the current CUDA stream for the block (no-op if None)."""
    if stream is None:
        yield
    else:
        with stream:
            yield
