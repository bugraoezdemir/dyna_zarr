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
  to 32 bits. The Gaussians run scipy's own kernel weights in scipy's order. The
  filters with weighted sums (the Gaussians, correlate, convolve) go to the engine only
  where it gives the installed scipy's values (see _scipy_sums_agree).
  median_filter, rank_filter and percentile_filter (step 4) for any numeric type on
  arrays of two or more dimensions, and for integers up to 32 bits on 1-D arrays (see
  _rank). correlate and convolve with a real n-D kernel (step 5), any numeric type.
- label_tiles (step 8, tilewise-ccl's Phase B): every tile labeled as
  scipy.ndimage.label labels it and mapped through its table, the same labels;
- pointwise operations whose values NumPy defines exactly (step 7): arithmetic,
  comparisons, logic, bit operations, rounding, sqrt, where and clip, each checked once
  against the NumPy dyna runs with (see _lowering_pointwise).

  Every mode, wrap included (the Python path takes a wrapped halo from the far end of
  the axis, as the engine does). Not uniform_filter of floats: scipy's running sum rounds
  differently depending on where a line starts, which on the Python path is the read
  box (with extended values at its inner edges), so those values depend on the region
  read too; integer sums are exact, whatever the start.

Everything else returns None here, and the caller uses the Python path.

A sliding minimum or maximum of floats through a NaN (minimum_filter, maximum_filter,
grey_erosion, grey_dilation, and rank filters of the first or last rank) is lowered, but
a read whose pass starts inside the array and meets a NaN goes back to the Python path:
scipy's loop keeps a NaN or drops it depending on the values before it, so its values
after a NaN depend on where the line starts. The Python path's values then depend on
the region read, and the engine cannot follow them; the engine raises UsePythonPath and
``read`` returns None. Without a NaN, both are scipy's values exactly.

A filter without its own ``device`` follows the device context of the read (cupy on a
GPU); the engine reads on the CPU, so such an array, and every array built on it, is
read by the Python path inside a GPU context (``INHERITS``).
"""
from __future__ import annotations

import operator
import os
import threading
import warnings
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
    """A cast made by ``astype`` (marked ``cast_to``) whose values NumPy defines, or a
    pointwise operation whose values NumPy defines exactly (step 7, see
    _lowering_pointwise)."""
    to = getattr(t, "cast_to", None)
    if to is None:
        from . import _lowering_pointwise
        return _lowering_pointwise.build(core, t, _up)
    if len(t.operands) != 1:
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
    ``block_info``, no ``align``, no other output type; or label_tiles (step 8), whose
    ``block_info`` and ``align`` are its own."""
    spec = getattr(t, "kernel", None)
    if spec is not None and spec.get("name") == "label_tiles":
        return _label_tiles(t, spec)
    if spec is None or t.block_info or t.align is not None or t.device not in (None, "cpu"):
        return None
    if np.dtype(t.dtype) != np.dtype(t.array.dtype):
        return None
    if spec.get("mode") not in _FILTER_MODES:
        return None
    build = _KERNELS.get(spec["name"])
    return None if build is None else build(t, spec)


def _label_tiles(t, spec):
    """tilewise-ccl's Phase B: the tables flattened in C order of the tile grid."""
    if t.device not in (None, "cpu") or not hasattr(t.array, "shape"):
        return None
    tile = tuple(spec["tile"])
    if t.align != tile:
        return None
    eng = _up(t.array)
    if eng is None or not hasattr(eng, "label_tiles"):
        return None
    shape = tuple(t.array.shape)
    grid = [-(-s // w) for s, w in zip(shape, tile)]
    tables, offsets, at = [], [0], 0
    import itertools
    for tidx in itertools.product(*[range(g) for g in grid]):
        tb = np.asarray(spec["tables"][tidx])
        if tb.ndim != 1 or tb.size < 1 or tb.dtype.kind not in "iu":
            return None
        tables.append(tb.astype(np.int64, copy=False))
        at += tb.size
        offsets.append(at)
    flat = np.concatenate(tables) if tables else np.zeros(0, np.int64)
    return eng.label_tiles(list(tile), spec["connectivity"], flat.tobytes(),
                           np.asarray(offsets, dtype=np.int64).tobytes(),
                           np.dtype(spec["dtype"]).name)


#: the modes in which the engine gives the Python path's values (scipy's five)
_FILTER_MODES = {"reflect", "mirror", "nearest", "wrap", "constant"}


# -------------------------------------------------------------------- the scipy check

#: whether the engine's weighted sums give this scipy's values (None: not checked yet)
_sums_agree: Optional[bool] = None
_sums_lock = threading.Lock()


def _same_values(want, got) -> bool:
    return want.dtype == got.dtype and want.shape == got.shape and bool(
        np.array_equal(want, got))


def _probe_sums(filters) -> bool:
    """The engine's correlate1d and n-D correlate against this scipy's, in float64."""
    import scipy.ndimage as ndi
    from scipy.ndimage._filters import _gaussian_kernel1d

    rng = np.random.default_rng(20261127)
    kernels = []
    for size in (3, 5, 9, 21, 31):          # symmetric: below and above 8 pairs of taps
        w = rng.uniform(-1, 1, size)
        kernels.append((w + w[::-1]) / 2)
    for size in (3, 5, 21):                 # antisymmetric
        w = rng.uniform(-1, 1, size)
        kernels.append((w - w[::-1]) / 2)
    for size in (2, 4, 7, 12):              # general, even and odd
        kernels.append(rng.uniform(-1, 1, size))
    for sigma, order in ((1.5, 0), (3.0, 0), (3.0, 1), (3.0, 2)):
        # the Gaussians' own weights, reversed as gaussian_filter1d reverses them
        kernels.append(_gaussian_kernel1d(sigma, order, int(4.0 * sigma + 0.5))[::-1])
    x = rng.standard_normal((6, 67)) * 50
    for w in kernels:
        for axis in (0, 1):
            got = np.empty_like(x)
            filters.correlate1d(x, [float(v) for v in w], axis, got, mode="reflect")
            if not _same_values(ndi.correlate1d(x, w, axis, mode="reflect"), got):
                return False
    y = rng.standard_normal((5, 7, 19)) * 50
    for shape in ((3, 3, 3), (2, 4, 4), (1, 5, 3)):
        w = rng.uniform(-1, 1, shape)
        got = np.empty_like(y)
        filters.correlate(y, list(shape), w.reshape(-1).tolist(), got, mode="reflect")
        if not _same_values(ndi.correlate(y, w, mode="reflect"), got):
            return False
    return True


def _scipy_sums_agree() -> bool:
    """Whether the engine's weighted sums give the installed scipy's values bit for bit
    (checked once per process).

    The engine's kernels follow scipy's C loops and also how the reference wheels were
    compiled: on arm64 the compiler fuses a multiply and an add into one FMA, and which
    of scipy's sums it fuses depends on the scipy release and the build (scipy 1.10's
    macOS arm64 wheel differs from 1.17's in the last bit; a conda-forge or distribution
    build can differ again). So gaussian_filter, gaussian_laplace,
    gaussian_gradient_magnitude, correlate and convolve go to the engine only where a
    probe shows it gives this scipy's float64 values: correlate1d (the Gaussians' pass)
    with symmetric, antisymmetric and general kernels and scipy's own Gaussian weights,
    and the n-D correlate with odd and even kernels. Otherwise they keep the Python
    path, of any type. laplace needs no check: its products by 1 and -2 are exact,
    fused or not.
    """
    global _sums_agree
    with _sums_lock:
        if _sums_agree is None:
            try:
                _sums_agree = _probe_sums(_engine.engine().filters)
            except Exception:                # noqa: BLE001 - no agreement: Python path
                _sums_agree = False
            else:
                if not _sums_agree:
                    _warn_scipy_sums()
        return _sums_agree


def _warn_scipy_sums():
    """Once per process, when the check fails (not when the engine cannot run it)."""
    import scipy
    warnings.warn(
        f"scipy {scipy.__version__} rounds the weighted sums of its filters in another way "
        "than the scipy release dyna-zarr's engine follows (the current one), so "
        "gaussian_filter, gaussian_laplace, gaussian_gradient_magnitude, correlate and "
        "convolve run on the slower Python path here. Their values are this scipy's either "
        'way. Upgrading scipy is strongly recommended: pip install --upgrade scipy',
        UserWarning, stacklevel=2)


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
    if sigma is None or order is None or not _scipy_sums_agree():
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
    if sigma is None or not _scipy_sums_agree():
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


#: types the engine's rank filter reads (scipy's n-D routine)
_RANK_TYPES = {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
               "float32", "float64"}
#: ... and on 1-D arrays, where scipy runs another routine (see _rank)
_RANK_1D_TYPES = {"int8", "uint8", "int16", "uint16", "int32", "uint32"}


def _scipy_rank(spec, n):
    """The rank scipy's _rank_filter picks in a window of n values, or None where scipy
    raises (the Python path then raises scipy's error)."""
    name = spec["name"]
    if name == "median_filter":
        rank = n // 2
    elif name == "percentile_filter":
        p = spec["percentile"]
        # Python and NumPy 64-bit numbers only: scipy computes with the percentile's own
        # type, and a float32 one would round differently
        if isinstance(p, (bool, np.bool_)) or not isinstance(p, (int, float, np.integer)):
            return None
        p = float(p)
        if p < 0.0:
            p += 100.0
        if not 0.0 <= p <= 100.0:
            return None
        rank = n - 1 if p == 100.0 else int(float(n) * p / 100.0)
    else:
        try:
            rank = operator.index(spec["rank"])
        except TypeError:
            return None
    if rank < 0:
        rank += n
    return rank if 0 <= rank < n else None


def _rank(t, spec):
    """median_filter, rank_filter and percentile_filter, as scipy's _rank_filter runs
    them: rank 0 and the last rank are minimum_filter / maximum_filter (separable passes);
    any other rank is the n-D rank filter. On a 1-D array scipy runs another routine
    (_rank_filter_1d, on int64 or the float type), which agrees with the n-D one where
    the window's values are told apart by value alone: integers up to 32 bits, with a
    cval the type holds exactly (that routine converts cval with NumPy, the n-D one with
    a C cast). Other 1-D arrays keep the Python path."""
    ndim = t.array.ndim
    sizes = _per_axis(spec["size"], ndim)
    if sizes is None or not all(isinstance(z, (int, np.integer))
                                and not isinstance(z, (bool, np.bool_)) and z >= 1
                                for z in sizes):
        return None
    sizes = [int(z) for z in sizes]
    n = int(np.prod(sizes))
    rank = _scipy_rank(spec, n)
    if rank is None:
        return None
    mode, cval = spec["mode"], float(spec["cval"])
    if rank in (0, n - 1):
        kind = "minimum" if rank == 0 else "maximum"
        passes = [(a, kind, [float(z)], 0, mode) for a, z in enumerate(sizes) if z > 1]
        eng = _up(t.array)
        if eng is None or not passes:
            return eng                          # a window of one: scipy copies the input
        return eng.filter([passes], cval=cval)
    dtype = np.dtype(t.dtype)
    if dtype.name not in _RANK_TYPES:
        return None
    if ndim == 1:
        if dtype.name not in _RANK_1D_TYPES:
            return None
        info = np.iinfo(dtype)
        if mode == "constant" and not (cval.is_integer() and info.min <= cval <= info.max):
            return None
    eng = _up(t.array)
    return None if eng is None else eng.rank_filter(sizes, rank, mode=mode, cval=cval)


def _correlate(t, spec):
    """correlate and convolve with an n-D kernel: scipy's weights as float64 (C order);
    the engine reverses them and shifts the origins for convolve as scipy does. Complex
    weights keep the Python path (scipy splits them into real and imaginary parts)."""
    w = np.asarray(spec["weights"])
    if w.dtype.kind not in "biuf" or w.ndim != t.array.ndim or 0 in w.shape:
        return None
    if np.dtype(t.dtype).name not in _RANK_TYPES or not _scipy_sums_agree():
        return None
    try:
        cval = float(spec["cval"])
    except (TypeError, ValueError):
        return None
    eng = _up(t.array)
    if eng is None:
        return None
    flat = np.ascontiguousarray(w, dtype=np.float64).reshape(-1)
    return eng.correlate(list(w.shape), flat.tolist(), mode=spec["mode"], cval=cval,
                         convolve=spec["name"] == "convolve")


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
    "median_filter": _rank,
    "rank_filter": _rank,
    "percentile_filter": _rank,
    "correlate": _correlate,
    "convolve": _correlate,
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
    # values that depend on where the read starts (UsePythonPath, see the module notes):
    # the Python path reads them, so both give the same values for every read
    use_python = getattr(_engine.engine(), "UsePythonPath", ())
    from ._scope import current
    scope = current()
    try:
        if threads == 1 and (scope is None or not scope.cache_bytes):
            return view.read([0] * len(shape), shape, threads=1)     # on this thread
        # several threads, or a write with a chunk cache: on the write's one pool
        # (decision DR-5, see _scope; the cache lives in the pool's threads)
        pool = None if scope is None else scope.pool()
        return view.read([0] * len(shape), shape, threads=threads, pool=pool)
    except use_python:
        return None
