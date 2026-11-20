"""dyna's arrays read by the native engine (Workplan WP15, task DR6a).

A DynamicArray is a source plus a chain of lazy transforms. When the engine is on
(``_engine.enabled()``) and every link of an array's chain has an engine counterpart, the
array is rebuilt once as an engine array (``dyna_zarr_core.Array``) and every read of it
goes there: one Rust read decodes, slices, transposes, joins, converts and downsamples,
without Python in between. A link the engine does not have (map_overlap, a reduction, a
TIFF or remote source, ...) keeps its Python code; the arrays below it are still read by
the engine wherever their own chains allow. So the Python path stays the reference, and
the engine takes over from the bottom up.

What the engine takes over, and only where it gives the Python path's exact values:

- sources: Zarr v2 / v3 arrays in a local folder, or in a finished zip file;
- slicing (ints, slices with a positive step, None, Ellipsis), transpose, swapaxes,
  expand_dims, squeeze, concatenate, stack;
- ``astype`` where NumPy's result is defined for every value ("same_kind" casts: no
  float -> integer, whose out-of-range values NumPy leaves to the platform);
- ``coarsen(..., "mean")``: both follow tensorstore.downsample.

Everything else returns None here, and the caller uses the Python path.
"""
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

import numpy as np

from . import _engine

#: the attribute caching an array's engine counterpart (dropped when pickling)
CACHE = "_engine_cache"
#: "not built yet" in the cache (None means "cannot be built")
_UNSET = object()
#: threads of one engine read: 1 inside io.write's region workers (each region is one
#: worker; decision DR-5's planner divides max_threads in DR6b); all cores in compute()
_THREADS: ContextVar[int] = ContextVar("dyna_zarr_engine_threads", default=1)
_lock = threading.Lock()


@contextmanager
def engine_threads(n: int):
    """Engine reads in this context use ``n`` threads."""
    token = _THREADS.set(max(1, int(n)))
    try:
        yield
    finally:
        _THREADS.reset(token)


def all_cores() -> int:
    return os.cpu_count() or 1


# ----------------------------------------------------------------------------- arrays

def lowered(array):
    """``array``'s engine counterpart when the engine is on and can read it, else None."""
    if not _engine.enabled():
        return None
    cached = array.__dict__.get(CACHE, _UNSET)
    if cached is _UNSET:
        try:
            cached = _build(array)
        except Exception:                       # noqa: BLE001 - not lowerable: Python path
            cached = None
        if cached is not None and not _same(cached, array):
            cached = None
        with _lock:
            array.__dict__[CACHE] = cached
    return cached


def _same(eng, array) -> bool:
    """The engine array has the DynamicArray's exact shape and element type."""
    try:
        dtype = np.dtype(array.dtype)
    except TypeError:
        return False
    return (tuple(eng.shape) == tuple(array.shape) and dtype.isnative
            and eng.dtype == dtype.name)


def _build(array):
    core = _engine.engine()
    transform = array._transform
    if transform is None:
        return _source(core, array)
    build = _TRANSFORMS.get(type(transform).__name__)
    return None if build is None else build(core, transform)


def _up(array):
    """The engine counterpart of an input of a transform (None stops the lowering)."""
    return lowered(array)


# ---------------------------------------------------------------------------- sources

def _source(core, array):
    """A Zarr array in a local folder or a finished zip, as the engine opens it."""
    z = getattr(array, "_zarr_array", None)
    if z is not None:
        eng = _zarr_python(core, z)
        if eng is not None:
            return eng
    ts = getattr(array, "_ts_array", None)
    if ts is not None:
        return _tensorstore(core, ts)
    return None


def _zarr_python(core, z):
    import zarr.storage as zs
    store = getattr(z, "store", None)
    inner = (getattr(z, "path", "") or "").strip("/")
    if isinstance(store, zs.LocalStore):
        path = Path(str(store.root))
        return core.Array.open(str(path / inner if inner else path))
    if isinstance(store, zs.ZipStore) and getattr(store, "read_only", False):
        return core.Array.open_zip(str(store.path), inner)
    return None


def _tensorstore(core, ts):
    """A stored TensorStore zarr array in a local folder, opened whole (a sliced,
    translated or virtual TensorStore view is not lowered)."""
    if any(int(o) != 0 for o in ts.domain.origin):
        return None
    from urllib.parse import unquote
    url = str(ts.kvstore.url)
    if not url.startswith("file://"):
        return None
    path = unquote(url[len("file://"):])
    if os.name == "nt" and path.startswith("/") and len(path) > 2 and path[2] == ":":
        path = path[1:]
    return core.Array.open(path)


# ------------------------------------------------------------------------- transforms

def _slice(core, t):
    eng = _up(t.array)
    if eng is None:
        return None
    key = _engine_key(t.key)
    return None if key is None else eng.index(key)


def _transpose(core, t):
    eng = _up(t.array)
    return None if eng is None else eng.transpose(list(t.axes))


def _swapaxes(core, t):
    eng = _up(t.array)
    if eng is None:
        return None
    perm = list(range(t.array.ndim))
    perm[t.axis1], perm[t.axis2] = perm[t.axis2], perm[t.axis1]
    return eng.transpose(perm)


def _expand_dims(core, t):
    eng = _up(t.array)
    return None if eng is None else eng.expand_dims(t.axis)


def _squeeze(core, t):
    eng = _up(t.array)
    if eng is None:
        return None
    for a in sorted(t.squeeze_axes, reverse=True):
        eng = eng.squeeze(a)
    return eng


def _parts(t):
    parts = [_up(a) for a in t.arrays]
    return None if any(p is None for p in parts) else parts


def _concatenate(core, t):
    parts = _parts(t)
    return None if parts is None else core.concatenate(parts, t.axis)


def _stack(core, t):
    parts = _parts(t)
    return None if parts is None else core.stack(parts, t.axis)


def _coarsen(core, t):
    if t.method != "mean":
        return None
    eng = _up(t.array)
    return None if eng is None else eng.downsample(list(t.factors), "mean")


def _map_blocks(core, t):
    """Only a cast made by ``astype`` (marked ``cast_to``) whose values NumPy defines."""
    to = getattr(t, "cast_to", None)
    if to is None or len(t.operands) != 1:
        return None
    src = t.operands[0]
    if not hasattr(src, "_transform"):
        return None
    if not np.can_cast(np.dtype(src.dtype), np.dtype(to), casting="same_kind"):
        return None
    eng = _up(src)
    return None if eng is None else eng.astype(np.dtype(to).name)


_TRANSFORMS = {
    "SliceTransform": _slice,
    "TransposeTransform": _transpose,
    "SwapAxesTransform": _swapaxes,
    "ExpandDimsTransform": _expand_dims,
    "SqueezeTransform": _squeeze,
    "ConcatenateTransform": _concatenate,
    "StackTransform": _stack,
    "CoarsenTransform": _coarsen,
    "MapBlocksTransform": _map_blocks,
}


# ------------------------------------------------------------------------------ reads

def _engine_key(key):
    """``key`` as the engine indexes (ints, slices with a positive step, None,
    Ellipsis), or None for anything else (the Python path then reads it)."""
    if not isinstance(key, tuple):
        key = (key,)
    out = []
    for k in key:
        if k is None or k is Ellipsis:
            out.append(k)
        elif isinstance(k, (int, np.integer)) and not isinstance(k, (bool, np.bool_)):
            out.append(int(k))
        elif isinstance(k, slice):
            parts = []
            for v in (k.start, k.stop, k.step):
                if v is None:
                    parts.append(None)
                elif isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)):
                    parts.append(int(v))
                else:
                    return None
            if parts[2] is not None and parts[2] <= 0:
                return None
            out.append(slice(*parts))
        else:
            return None
    return tuple(out)


def read(array, key):
    """``array[key]`` read by the engine, or None when the Python path must read it."""
    eng = lowered(array)
    if eng is None:
        return None
    k = _engine_key(key)
    if k is None:
        return None
    try:
        view = eng.index(k)
    except Exception:                           # noqa: BLE001 - e.g. out of range: Python
        return None                             # path, with its own error
    shape = list(view.shape)
    return view.read([0] * len(shape), shape, threads=_THREADS.get())
