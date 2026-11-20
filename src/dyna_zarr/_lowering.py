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
- ``coarsen(..., "mean")``: both follow tensorstore.downsample;
- the named filters (task DR7) in reflect, mirror, nearest or constant mode, with
  ``cval`` (and ``order`` for gaussian_filter), their values scipy's bit for bit:
  gaussian_filter, gaussian_laplace and laplace (any numeric type),
  gaussian_gradient_magnitude (float32 / float64: for integers scipy's square root goes
  through NumPy casts the engine does not follow), minimum_filter, maximum_filter,
  grey_erosion and grey_dilation (any numeric type), and uniform_filter for integers up
  to 32 bits. The Gaussians run scipy's own kernel weights in scipy's order.

  Not wrap: the Python path wraps each read's box around itself rather than the array,
  so its values depend on the region read, and the engine (which wraps around the
  array) would differ from it. Not uniform_filter of floats: scipy's running sum rounds
  differently depending on where a line starts, which on the Python path is the read
  box (with extended values at its inner edges), so those values depend on the region
  read too; integer sums are exact, whatever the start.

Everything else returns None here, and the caller uses the Python path.

A filter without its own ``device`` follows the device context of the read (cupy on a
GPU); the engine reads on the CPU, so such an array, and every array built on it, is
read by the Python path inside a GPU context (``INHERITS``).
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
#: set on an array whose engine counterpart includes a filter that follows the device
#: context (device=None): read by the engine only in a CPU context
INHERITS = "_engine_inherits_device"
#: "not built yet" in the cache (None means "cannot be built")
_UNSET = object()
#: threads of one engine read: inside a write, what its planner gives each region
#: (regions in flight x threads per region <= max_threads, decision DR-5); all cores in
#: compute()
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
        inherits = cached is not None and _inherits_device(array)
        with _lock:
            array.__dict__[INHERITS] = inherits
            array.__dict__[CACHE] = cached
    return cached


def _inherits_device(array) -> bool:
    """Whether ``array``'s lowered chain has a filter that follows the device context."""
    t = array._transform
    if t is None:
        return False
    if type(t).__name__ == "MapOverlapTransform" and t.device is None:
        return True
    inputs = [getattr(t, "array", None), *getattr(t, "arrays", ()), *getattr(t, "operands", ())]
    # inputs were lowered first (their flags are set); scalar operands have no __dict__
    return any(getattr(a, "__dict__", {}).get(INHERITS, False) for a in inputs)


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


def _map_overlap(core, t):
    """A named filter (its ``kernel`` spec) the engine has, on the CPU, read whole: no
    ``block_info``, no ``align``, no other output type."""
    spec = getattr(t, "kernel", None)
    if spec is None or t.block_info or t.align is not None or t.device not in (None, "cpu"):
        return None
    if np.dtype(t.dtype) != np.dtype(t.array.dtype):
        return None
    if spec.get("mode") not in _FILTER_MODES:
        return None
    build = _KERNELS.get(spec["name"])
    return None if build is None else build(t, spec)


#: the modes in which the engine gives the Python path's values (not wrap: see above)
_FILTER_MODES = {"reflect", "mirror", "nearest", "constant"}


def _gaussian_chain(ndim, sigma, order, mode, truncate):
    """scipy.ndimage.gaussian_filter as engine passes (axis, kind, values, origin, mode):
    per axis with sigma > 1e-15 (as scipy), scipy's own weights for (sigma, order,
    radius), reversed as gaussian_filter1d reverses them. None if scipy's kernel function
    cannot be found (the Python path then keeps the filter)."""
    try:
        from scipy.ndimage._filters import _gaussian_kernel1d
    except ImportError:
        return None
    chain = []
    for axis in range(ndim):
        if not sigma[axis] > 1e-15:
            continue
        radius = int(truncate * float(sigma[axis]) + 0.5)
        weights = _gaussian_kernel1d(sigma[axis], int(order[axis]), radius)[::-1]
        chain.append((axis, "correlate", [float(w) for w in weights], 0, mode))
    return chain


def _gaussian(t, spec):
    """gaussian_filter: one chain."""
    ndim = t.array.ndim
    sigma = _per_axis(spec["sigma"], ndim)
    order = _per_axis(spec["order"], ndim)
    if sigma is None or order is None:
        return None
    chain = _gaussian_chain(ndim, sigma, order, spec["mode"], spec["truncate"])
    eng = None if chain is None else _up(t.array)
    if eng is None:
        return None
    if not chain:
        return eng                              # scipy copies the input
    return eng.filter([chain], cval=float(spec["cval"]))


def _gaussian_terms(t, spec, order, combine):
    """gaussian_laplace (order 2, "sum") and gaussian_gradient_magnitude (order 1,
    "hypot"): one Gaussian chain per axis, with that axis' derivative, each applied to
    the input (scipy's generic_laplace / generic_gradient_magnitude)."""
    ndim = t.array.ndim
    if combine == "hypot" and np.dtype(t.dtype).kind != "f":
        return None
    sigma = _per_axis(spec["sigma"], ndim)
    if sigma is None:
        return None
    chains = []
    for d in range(ndim):
        orders = [order if a == d else 0 for a in range(ndim)]
        chain = _gaussian_chain(ndim, sigma, orders, spec["mode"], spec["truncate"])
        if chain is None:
            return None
        chains.append(chain)
    eng = _up(t.array)
    if eng is None:
        return None
    return eng.filter(chains, combine=combine, cval=float(spec["cval"]))


def _laplace(t, spec):
    """laplace: a [1, -2, 1] pass per axis on the input, added (generic_laplace)."""
    eng = _up(t.array)
    if eng is None:
        return None
    chains = [[(a, "correlate", [1.0, -2.0, 1.0], 0, spec["mode"])]
              for a in range(t.array.ndim)]
    return eng.filter(chains, combine="sum", cval=float(spec["cval"]))


#: uniform_filter goes to the engine for these types only (see the module notes)
_UNIFORM_TYPES = {"int8", "uint8", "int16", "uint16", "int32", "uint32"}

#: the pass kind of each filter with a window
_SIZE_KINDS = {"uniform_filter": "uniform", "minimum_filter": "minimum",
               "grey_erosion": "minimum", "maximum_filter": "maximum",
               "grey_dilation": "maximum"}


def _size_filter(t, spec):
    """uniform_filter, minimum_filter, maximum_filter, grey_erosion, grey_dilation: one
    pass per axis whose size is above 1 (scipy skips the others), with scipy's origin:
    0, except grey_dilation, which negates it and lowers it by one for an even size."""
    name = spec["name"]
    if name == "uniform_filter" and np.dtype(t.dtype).name not in _UNIFORM_TYPES:
        return None
    sizes = _per_axis(spec["size"], t.array.ndim)
    if sizes is None or not all(isinstance(z, (int, np.integer))
                                and not isinstance(z, (bool, np.bool_)) for z in sizes):
        return None
    passes = []
    for axis, z in enumerate(sizes):
        z = int(z)
        if z > 1:
            origin = -(1 - z % 2) if name == "grey_dilation" else 0
            passes.append((axis, _SIZE_KINDS[name], [float(z)], origin, spec["mode"]))
    eng = _up(t.array)
    if eng is None:
        return None
    if not passes:
        return eng                              # scipy copies the input
    return eng.filter([passes], cval=float(spec["cval"]))


#: the named filters the engine reads, by the name their kernel spec records
_KERNELS = {
    "gaussian_filter": _gaussian,
    "gaussian_laplace": lambda t, spec: _gaussian_terms(t, spec, 2, "sum"),
    "gaussian_gradient_magnitude": lambda t, spec: _gaussian_terms(t, spec, 1, "hypot"),
    "laplace": _laplace,
    "uniform_filter": _size_filter,
    "minimum_filter": _size_filter,
    "maximum_filter": _size_filter,
    "grey_erosion": _size_filter,
    "grey_dilation": _size_filter,
}


def _per_axis(value, ndim):
    """scipy's _normalize_sequence: one value for every axis, or one per axis (the
    elements themselves, as scipy passes them on)."""
    if isinstance(value, (str, bytes)):
        return None
    if np.ndim(value) == 0:
        return [value] * ndim
    value = list(value)
    return value if len(value) == ndim else None


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
    "MapOverlapTransform": _map_overlap,
}


# ------------------------------------------------------------------------------ reads

def _device() -> str:
    from .operations._backend import current_device
    return current_device()


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
    if array.__dict__.get(INHERITS) and _device() != "cpu":
        return None                             # the filter runs on that device
    k = _engine_key(key)
    if k is None:
        return None
    try:
        view = eng.index(k)
    except Exception:                           # noqa: BLE001 - e.g. out of range: Python
        return None                             # path, with its own error
    shape = list(view.shape)
    threads = _THREADS.get()
    if threads == 1:
        return view.read([0] * len(shape), shape, threads=1)     # on this thread
    # several threads: inside a write, on its one pool (decision DR-5, see _scope)
    from ._scope import current
    scope = current()
    pool = None if scope is None else scope.pool()
    return view.read([0] * len(shape), shape, threads=threads, pool=pool)
