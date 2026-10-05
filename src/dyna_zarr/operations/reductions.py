"""Reductions -- the reductive/destructive taxonomy cell.

A reduction collapses one or more axes. The pull model can't cheaply slice a *reduced*
axis (the output element depends on the whole axis), so ``read(key)`` reads only the
kept-axis tile the key asks for but **streams the reduced axis in bounded chunks**,
combining them with an associative reducer (min/max/sum/prod/mean/any/all). That keeps it
memory-bound even for a huge reduced axis, and the result is chunk/region-invariant and
exact vs numpy. Step/integer indices on the output are applied after the reduce (same
crop-at-the-end trick as map_overlap).

``axis=None`` reduces everything to a 0-d result, still streamed. The result is a lazy
DynamicArray (chainable / writable); the DynamicArray.min()/max()/... methods return the
same lazy result.

Small results are computed ONCE. A reduction whose output is at most ``_CACHE_MAX_BYTES``
(a 0-d statistic, per-channel stats, ...) caches its full result on first full read, and
``evaluate_small_reductions`` - run by ``compute`` and ``io.write`` before any region is
read - evaluates the ones a chain will need, fusing reductions of the same input into one
streaming pass. Without that, ``x > x.mean()`` written region by region would re-stream
the mean for every region. Larger results are never cached: they stay blockwise, and
``find_repeated_reductions`` reports where a broadcast makes them re-read.
"""

import builtins
import threading
import weakref
import numpy as np
from typing import Optional, Tuple

from ._base import Transform, _is_int_index, iter_chain
from ._backend import array_namespace, asnumpy, resolve_device, to_device

_DEFAULT_STRIP_BYTES = 64 * 1024 * 1024   # per-read streaming budget for the reduced axis

#: Largest reduction OUTPUT that is cached in memory (1 MiB). A single number always
#: fits; so do per-channel/per-timepoint statistics. Anything larger stays blockwise.
_CACHE_MAX_BYTES = 1 << 20

#: Reductions currently holding a cached result, for io.clear_cache(). Weak, so a cache
#: never keeps an otherwise-unreachable chain alive.
_CACHED = weakref.WeakSet()


def _ident(xp, s, count, ddof):
    return s


def _div_count(xp, s, count, ddof):
    return s / count


def _var_partial(xp, b, ax):
    # accumulate in float64 so the one-pass sum-of-squares identity stays accurate
    bf = b.astype(xp.float64)
    return (xp.sum(bf, axis=ax), xp.sum(bf * bf, axis=ax))


def _pair_add(xp, a, b):
    return (a[0] + b[0], a[1] + b[1])


def _var_finalize(xp, s, count, ddof):
    ssum, ssq = s
    var = (ssq - ssum * ssum / count) / (count - ddof)
    return xp.clip(var, 0.0, None)          # guard tiny negatives from float error


def _std_finalize(xp, s, count, ddof):
    return xp.sqrt(_var_finalize(xp, s, count, ddof))


class _Reducer:
    """Associative reducers use partial(xp, block, axes)->state, combine(xp, a, b)->state,
    finalize(xp, state, count, ddof)->result and are streamed over the reduced axis.
    Non-associative reducers (argmin/argmax) can't be chunked on the reduced axis, so they
    read it whole per kept-tile and apply direct(xp, block, axes) in one shot."""
    def __init__(self, partial=None, combine=None, finalize=_ident,
                 associative=True, direct=None):
        self.partial = partial
        self.combine = combine
        self.finalize = finalize
        self.associative = associative
        self.direct = direct


def _arg_axis(ax, ndim):
    """numpy's ``axis`` for argmin/argmax from the normalised reduced axes ``ax``.

    ``axis=None`` normalises to EVERY axis, which numpy spells ``None`` (a flat index);
    a single axis passes through. Several-but-not-all axes has no numpy meaning.
    """
    if len(ax) == ndim:
        return None
    if len(ax) == 1:
        return ax[0]
    raise ValueError("argmin/argmax take a single axis or axis=None")


_REDUCERS = {
    "min":  _Reducer(lambda xp, b, ax: xp.min(b, axis=ax),  lambda xp, a, b: xp.minimum(a, b)),
    "max":  _Reducer(lambda xp, b, ax: xp.max(b, axis=ax),  lambda xp, a, b: xp.maximum(a, b)),
    "sum":  _Reducer(lambda xp, b, ax: xp.sum(b, axis=ax),  lambda xp, a, b: xp.add(a, b)),
    "prod": _Reducer(lambda xp, b, ax: xp.prod(b, axis=ax), lambda xp, a, b: xp.multiply(a, b)),
    "any":  _Reducer(lambda xp, b, ax: xp.any(b, axis=ax),  lambda xp, a, b: xp.logical_or(a, b)),
    "all":  _Reducer(lambda xp, b, ax: xp.all(b, axis=ax),  lambda xp, a, b: xp.logical_and(a, b)),
    "mean": _Reducer(lambda xp, b, ax: xp.sum(b, axis=ax),  lambda xp, a, b: xp.add(a, b), _div_count),
    "var":  _Reducer(_var_partial, _pair_add, _var_finalize),
    "std":  _Reducer(_var_partial, _pair_add, _std_finalize),
    # non-associative: read the reduced axis whole per kept-tile, apply in one shot
    "argmin": _Reducer(associative=False,
                       direct=lambda xp, b, ax: xp.argmin(b, axis=_arg_axis(ax, b.ndim))),
    "argmax": _Reducer(associative=False,
                       direct=lambda xp, b, ax: xp.argmax(b, axis=_arg_axis(ax, b.ndim))),
    "median": _Reducer(associative=False, direct=lambda xp, b, ax: xp.median(b, axis=ax)),
}


def _normalize_axes(axis, ndim) -> Tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))
    if np.isscalar(axis):
        axis = (axis,)
    out = []
    for a in axis:
        a = int(a) if a >= 0 else ndim + int(a)
        if a < 0 or a >= ndim:
            raise ValueError(f"axis {a} out of bounds for ndim {ndim}")
        out.append(a)
    return tuple(sorted(set(out)))


class ReduceTransform(Transform):
    """Lazy, streaming reduction over one or more axes."""

    def __init__(self, array, reducer, axis=None, keepdims=False,
                 strip_bytes=_DEFAULT_STRIP_BYTES, device=None, ddof=0):
        super().__init__()
        if reducer not in _REDUCERS:
            raise ValueError(f"unknown reducer {reducer!r}; expected {sorted(_REDUCERS)}")
        from ..utils import parse_dtype
        self.array = array
        self.reducer_name = reducer
        self.reducer = _REDUCERS[reducer]
        self.keepdims = keepdims
        self.strip_bytes = strip_bytes
        self.device = device
        self.ddof = ddof
        ndim = array.ndim
        self.R = _normalize_axes(axis, ndim)
        self.kept = tuple(a for a in range(ndim) if a not in self.R)
        self._in_dtype = parse_dtype(array.dtype)[0]

        chunks = array.chunks
        if keepdims:
            self.shape = tuple(1 if a in self.R else array.shape[a] for a in range(ndim))
            self.chunks = (tuple(1 if a in self.R else chunks[a] for a in range(ndim))
                           if chunks else None)
        else:
            self.shape = tuple(array.shape[a] for a in self.kept)
            self.chunks = tuple(chunks[a] for a in self.kept) if chunks else None

        self.dtype = self._infer_dtype()

        # Small outputs are computed once and kept (see the module docstring).
        nbytes = int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize
        self.cacheable = nbytes <= _CACHE_MAX_BYTES
        self._cache = None
        self._cache_lock = threading.Lock()

    def _finish(self, result):
        """A finalized result in the reported dtype (numpy's), on whatever device it is."""
        xp = array_namespace(result)
        result = xp.asarray(result)
        return result if result.dtype == self.dtype else result.astype(self.dtype)

    def describe(self):
        """Short human-readable name, for warnings: ``mean(axis=(0,)) -> (1, 512, 512)``."""
        return f"{self.reducer_name}(axis={self.R}) -> {tuple(self.shape)}"

    # --- cache ---------------------------------------------------------------------
    def _set_cache(self, full):
        """Store the FULL output (host, output shape) unless another thread already did."""
        with self._cache_lock:
            if self._cache is None:
                self._cache = np.asarray(asnumpy(full))
                _CACHED.add(self)

    def ensure_cached(self):
        """Compute and cache the full output once; concurrent callers wait for it."""
        if self._cache is not None:
            return
        with self._cache_lock:
            if self._cache is not None:
                return
            full = tuple(slice(0, s) for s in self.shape)
            self._cache = np.asarray(asnumpy(self._read_uncached(full)))
            _CACHED.add(self)

    def clear_cache(self):
        with self._cache_lock:
            self._cache = None
        _CACHED.discard(self)

    def _infer_dtype(self):
        sample = np.ones((2,) * self.array.ndim, dtype=self._in_dtype)
        if self.reducer_name in ("mean", "var", "std"):
            # numpy's OWN result dtype (float32 stays float32; ints -> float64). Deriving it
            # from our sum/count arithmetic made it depend on the numpy version: NumPy 1.x
            # promotes float32-scalar / int to float64, NumPy 2 does not - and var/std
            # accumulate in float64 by design. Results are cast to this (see _finish).
            return np.asarray(getattr(np, self.reducer_name)(sample, axis=self.R)).dtype
        if not self.reducer.associative:
            res = self.reducer.direct(np, sample, self.R)
        else:
            state = self.reducer.partial(np, sample, self.R)
            res = self.reducer.finalize(np, state, 2 ** len(self.R), self.ddof)
        return np.asarray(res).dtype

    def _stream(self, input_slices):
        """Reduce over self.R. Associative reducers stream the largest reduced axis in
        bounded chunks + combine; non-associative ones read the reduced axis whole."""
        arr = self.array
        if not self.reducer.associative:
            block = to_device(arr._read_direct(tuple(input_slices)), resolve_device(self.device))
            xp = array_namespace(block)
            if block.size == 0:      # empty kept region: numpy median/argmin choke -> build empty
                empty_shape = tuple(s for i, s in enumerate(block.shape) if i not in self.R)
                return xp.empty(empty_shape, dtype=self.dtype)
            return self.reducer.direct(xp, block, self.R)
        kept_elems = 1
        for a in self.kept:
            s = input_slices[a]
            kept_elems *= (s.stop - s.start)
        chunk_axis = builtins.max(self.R, key=lambda a: arr.shape[a])
        other_reduced = 1
        for a in self.R:
            if a != chunk_axis:
                other_reduced *= arr.shape[a]
        denom = builtins.max(1, kept_elems * other_reduced * self._in_dtype.itemsize)
        chunk_len = builtins.max(1, int(self.strip_bytes // denom))

        size = arr.shape[chunk_axis]
        acc = None
        for c in range(0, size, chunk_len):
            slices = list(input_slices)
            slices[chunk_axis] = slice(c, builtins.min(size, c + chunk_len))
            block = to_device(arr._read_direct(tuple(slices)), resolve_device(self.device))
            xp = array_namespace(block)
            p = self.reducer.partial(xp, block, self.R)
            acc = p if acc is None else self.reducer.combine(xp, acc, p)
        count = 1
        for a in self.R:
            count *= arr.shape[a]
        return self.reducer.finalize(xp, acc, count, self.ddof)

    def read(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        # A cached result serves any key. Otherwise a small reduction is cached on its
        # first FULL read; a partial read of an uncached one stays blockwise - only the
        # kept-axis tile asked for is streamed - so e.g. mip[0:10, 0:10] never pays for
        # the whole projection just because the projection would fit the cache.
        if self._cache is None and self.cacheable and _is_full_key(key, self.shape):
            self.ensure_cached()
        cached = self._cache
        if cached is not None:
            return to_device(np.asarray(cached[key]), resolve_device(self.device))
        return self._read_uncached(key)

    def _read_uncached(self, key):
        out_ndim = len(self.shape)
        input_slices = [slice(None)] * self.array.ndim   # reduced axes stay full (streamed)
        params = []                                       # (start, stop, step, is_int) per out axis
        for oi in range(out_ndim):
            k = key[oi]
            osize = self.shape[oi]
            if _is_int_index(k):
                idx = int(k) if k >= 0 else osize + int(k)
                start, stop, step, is_int = idx, idx + 1, 1, True
            else:
                start, stop, step = k.indices(osize)
                is_int = False
            params.append((start, stop, step, is_int))
            in_axis = oi if self.keepdims else self.kept[oi]
            if in_axis in self.kept:
                # contiguous read on this kept input axis; step/int applied after reduce
                input_slices[in_axis] = slice(start, stop)
            # keepdims reduced axis: read stays full; its size-1 output handled in the crop

        block = self._finish(self._stream(input_slices))  # kept-axis order, reduced axes gone
        xp = array_namespace(block)
        if self.keepdims:
            for a in sorted(self.R):
                block = xp.expand_dims(block, a)

        crop = tuple(slice(0, stop - start, step) for (start, stop, step, _) in params)
        block = block[crop]
        for oi in sorted((i for i, p in enumerate(params) if p[3]), reverse=True):
            block = xp.squeeze(block, axis=oi)
        return block


# --------------------------------------------------------------------------- #
# Small-result cache: pre-pass, fusion, clearing
# --------------------------------------------------------------------------- #

def _is_full_key(key, shape):
    """True when ``key`` (padded, ints/slices) selects all of ``shape`` unchanged."""
    for k, n in zip(key, shape):
        if not isinstance(k, slice) or k.indices(n) != (0, n, 1):
            return False
    return True


def _stream_group(transforms):
    """Full results of several associative reductions of the SAME input over the SAME
    axes, from ONE streaming pass: each bounded block is read once and fed to every
    reducer. This is what makes ``x.mean()`` and ``x.std()`` cost one pass, not two."""
    t0 = transforms[0]
    arr = t0.array
    R = t0.R
    kept_elems = 1
    for a in t0.kept:
        kept_elems *= arr.shape[a]
    chunk_axis = builtins.max(R, key=lambda a: arr.shape[a])
    other_reduced = 1
    for a in R:
        if a != chunk_axis:
            other_reduced *= arr.shape[a]
    strip = builtins.min(t.strip_bytes for t in transforms)
    denom = builtins.max(1, kept_elems * other_reduced * t0._in_dtype.itemsize)
    chunk_len = builtins.max(1, int(strip // denom))
    dev = resolve_device(t0.device)

    accs = [None] * len(transforms)
    xp = np
    size = arr.shape[chunk_axis]
    for c in range(0, size, chunk_len):
        slices = [slice(0, n) for n in arr.shape]
        slices[chunk_axis] = slice(c, builtins.min(size, c + chunk_len))
        block = to_device(arr._read_direct(tuple(slices)), dev)
        xp = array_namespace(block)
        for i, t in enumerate(transforms):
            p = t.reducer.partial(xp, block, R)
            accs[i] = p if accs[i] is None else t.reducer.combine(xp, accs[i], p)
    count = 1
    for a in R:
        count *= arr.shape[a]
    results = []
    for t, acc in zip(transforms, accs):
        r = t._finish(t.reducer.finalize(xp, acc, count, t.ddof))
        if t.keepdims:
            for a in sorted(R):
                r = xp.expand_dims(r, a)
        results.append(r)
    return results


def _reductions_in(array):
    """Every ReduceTransform in ``array``'s chain, upstream first."""
    return [n._transform for n in iter_chain(array)
            if isinstance(n._transform, ReduceTransform)]


def _broadcast_consumers(array):
    """``(map_blocks_node, operand, axes)`` for every operand in the chain that is
    broadcast (repeated) along ``axes`` of its map_blocks output."""
    from .pointwise import MapBlocksTransform
    out = []
    for node in iter_chain(array):
        tr = node._transform
        if isinstance(tr, MapBlocksTransform):
            for operand, axes in tr.broadcast_operands():
                out.append((node, operand, axes))
    return out


def evaluate_small_reductions(array):
    """Compute, before any region is read, the cacheable reductions ``array`` will
    re-read: every 0-d one, and every small one feeding a broadcast operand. Reductions
    of the same input over the same axes share ONE streaming pass.

    Called by ``compute()`` and ``io.write``. Partial reads of other small reductions
    are left blockwise (see ReduceTransform.read)."""
    chain = _reductions_in(array)
    wanted = {id(t) for t in chain if t.cacheable and len(t.shape) == 0}
    for _node, operand, _axes in _broadcast_consumers(array):
        wanted.update(id(t) for t in _reductions_in(operand) if t.cacheable)
    targets = [t for t in chain if id(t) in wanted and t._cache is None]
    if not targets:
        return
    groups = {}
    for t in targets:            # insertion order = upstream first
        if t.reducer.associative:
            key = (id(t.array), t.R, t.device)
            groups.setdefault(key, []).append(t)
        else:
            groups[("single", id(t))] = [t]
    for group in groups.values():
        pending = [t for t in group if t._cache is None]
        if not pending:
            continue
        t0 = pending[0]
        if (len(pending) == 1 or not t0.reducer.associative or not t0.R
                or 0 in tuple(t0.array.shape)):
            for t in pending:
                t.ensure_cached()
            continue
        for t, full in zip(pending, _stream_group(pending)):
            t._set_cache(full)


def find_repeated_reductions(array):
    """Broadcast operands whose chain holds a reduction too large to cache, so a
    region-wise read re-streams it once per region along the broadcast axes.

    Returns ``[(consumer_shape, axes, description)]``: ``axes`` are output axes of the
    consuming map_blocks, whose shape is ``consumer_shape`` (the caller maps these onto
    its own regions only when the shapes agree)."""
    found = []
    for node, operand, axes in _broadcast_consumers(array):
        heavy = [t for t in _reductions_in(operand) if not t.cacheable and t._cache is None]
        if heavy:
            found.append((tuple(node.shape), axes, heavy[-1].describe()))
    return found


def clear_chain_cache(array):
    """Drop every cached result in ``array``'s chain (small reductions, histograms)."""
    for node in iter_chain(array):
        clear = getattr(node._transform, "clear_cache", None)
        if clear is not None:
            clear()


def clear_all_caches():
    """Drop every cached reduction result (what ``io.clear_cache()`` calls)."""
    for t in list(_CACHED):
        t.clear_cache()


# --------------------------------------------------------------------------- #
# Public ops -- reductions
# --------------------------------------------------------------------------- #

def reduce(array, reducer, axis=None, keepdims=False, device=None, ddof=0):
    """Lazy streaming reduction with a named ``reducer`` (min/max/sum/prod/mean/var/std/
    any/all/argmin/argmax) over ``axis`` (int, tuple, or None for all). ``device``
    (None=inherit, 'cpu', 'cuda') runs it on that device; ``ddof`` applies to var/std."""
    return array._with_transform(
        ReduceTransform(array, reducer, axis=axis, keepdims=keepdims, device=device, ddof=ddof))


def min(array, axis=None, keepdims=False, device=None):
    """Minimum along axis/axes (lazy, streaming reduction)."""
    return reduce(array, "min", axis=axis, keepdims=keepdims, device=device)


def max(array, axis=None, keepdims=False, device=None):
    """Maximum along axis/axes (lazy, streaming reduction)."""
    return reduce(array, "max", axis=axis, keepdims=keepdims, device=device)


def sum(array, axis=None, keepdims=False, device=None):
    """Sum along axis/axes (lazy, streaming reduction)."""
    return reduce(array, "sum", axis=axis, keepdims=keepdims, device=device)


def prod(array, axis=None, keepdims=False, device=None):
    """Product along axis/axes (lazy, streaming reduction)."""
    return reduce(array, "prod", axis=axis, keepdims=keepdims, device=device)


def mean(array, axis=None, keepdims=False, device=None):
    """Mean along axis/axes (lazy, streaming: running sum / count)."""
    return reduce(array, "mean", axis=axis, keepdims=keepdims, device=device)


def any(array, axis=None, keepdims=False, device=None):
    """Logical OR along axis/axes (lazy, streaming reduction)."""
    return reduce(array, "any", axis=axis, keepdims=keepdims, device=device)


def all(array, axis=None, keepdims=False, device=None):
    """Logical AND along axis/axes (lazy, streaming reduction)."""
    return reduce(array, "all", axis=axis, keepdims=keepdims, device=device)


def var(array, axis=None, keepdims=False, ddof=0, device=None):
    """Variance along axis/axes (lazy, streaming: (sum, sum-of-squares, count), float64)."""
    return reduce(array, "var", axis=axis, keepdims=keepdims, device=device, ddof=ddof)


def std(array, axis=None, keepdims=False, ddof=0, device=None):
    """Standard deviation along axis/axes (lazy, streaming; sqrt of var)."""
    return reduce(array, "std", axis=axis, keepdims=keepdims, device=device, ddof=ddof)


def argmin(array, axis=None, keepdims=False, device=None):
    """Index of the minimum along a single ``axis`` (or flat if None). Reads the reduced
    axis whole per kept-tile (not associatively streamable)."""
    return reduce(array, "argmin", axis=axis, keepdims=keepdims, device=device)


def argmax(array, axis=None, keepdims=False, device=None):
    """Index of the maximum along a single ``axis`` (or flat if None). Reads the reduced
    axis whole per kept-tile (not associatively streamable)."""
    return reduce(array, "argmax", axis=axis, keepdims=keepdims, device=device)


def median(array, axis=None, keepdims=False, device=None):
    """Median along axis/axes (or all if None). Not associatively streamable, so it reads
    the reduced axis/axes whole per kept-tile (memory bounded by the kept region)."""
    return reduce(array, "median", axis=axis, keepdims=keepdims, device=device)


def _iter_region_slices(shape, itemsize, budget):
    """Yield memory-bounded region slice-tuples covering ``shape`` (<= ``budget`` bytes
    each), expanding trailing (contiguous) axes first."""
    import itertools
    if 0 in tuple(shape):                     # an empty array has no regions to read
        return
    budget_elems = builtins.max(1, int(budget) // builtins.max(1, itemsize))
    region = [1] * len(shape)
    acc = 1
    for a in reversed(range(len(shape))):
        region[a] = builtins.min(shape[a], builtins.max(1, budget_elems // acc))
        acc *= region[a]
        if region[a] < shape[a]:
            break
    for start in itertools.product(*[range(0, s, r) for s, r in zip(shape, region)]):
        yield tuple(slice(st, builtins.min(st + r, s))
                    for st, r, s in zip(start, region, shape))


class _ComputedOnce(Transform):
    """A small result that needs a FULL pass over its input to produce any part of it:
    computed once on first read, under a lock, then served from a host cache (cleared by
    io.clear_cache()/arr.clear_cache(), like the small reductions)."""

    def __init__(self):
        super().__init__()
        self._cache = None
        self._cache_lock = threading.Lock()

    def _compute_full(self):                 # -> host ndarray of self.shape
        raise NotImplementedError

    def ensure_cached(self):
        if self._cache is not None:
            return
        with self._cache_lock:
            if self._cache is None:
                self._cache = np.asarray(self._compute_full())
                _CACHED.add(self)

    def clear_cache(self):
        with self._cache_lock:
            self._cache = None
        _CACHED.discard(self)

    def read(self, key):
        self.ensure_cached()
        return np.asarray(self._cache[key])


class HistogramEdgesTransform(_ComputedOnce):
    """Bin edges from the DATA range (``range=None``): exactly numpy's - the array's own
    min/max, a degenerate range widened to (v - 0.5, v + 0.5), numpy's edge dtype - via
    ``numpy.histogram_bin_edges`` on the two extremes. ``lo``/``hi`` are 0-d reductions held
    as ``operands``, so a compute()/io.write pre-pass fuses them into one streaming pass."""

    def __init__(self, lo, hi, bins, src_dtype):
        super().__init__()
        self.operands = [lo, hi]
        self.bins = int(bins)
        self.src_dtype = src_dtype
        self.shape = (self.bins + 1,)
        self.chunks = self.shape
        self.dtype = np.histogram_bin_edges(np.zeros(2, dtype=src_dtype), bins=self.bins).dtype

    def _compute_full(self):
        lo, hi = (np.asarray(asnumpy(o._read_direct(()))) for o in self.operands)
        # numpy raises for a non-finite autodetected range (NaN/inf data); so does this.
        return np.histogram_bin_edges(np.array([lo, hi], dtype=self.src_dtype), bins=self.bins)


class HistogramTransform(_ComputedOnce):
    """Lazy histogram COUNTS: one streaming pass over ``array`` in bounded regions, summing
    per-region histograms that share one set of edges (a histogram is associative), so it
    is memory-bound and exact. ``operands`` holds the edges array (lazy when they depend on
    the data), so its dependencies are visible to the chain walkers."""

    def __init__(self, array, edges, bins, range_, strip_bytes, device):
        super().__init__()
        self.array = array
        self.operands = [edges]
        self.bins = bins                     # int, or None when the edges are given
        self.range = range_                  # (lo, hi) floats when given explicitly, else None
        self.strip_bytes = strip_bytes
        self.device = device
        self.shape = (int(edges.shape[0]) - 1,)
        self.chunks = self.shape
        self.dtype = np.dtype(np.int64)

    def _compute_full(self):
        from ..utils import parse_dtype
        dev = resolve_device(self.device)
        itemsize = parse_dtype(self.array.dtype)[0].itemsize
        edges = None
        if self.range is None:               # explicit or data-derived edges
            edges = np.asarray(asnumpy(self.operands[0]._read_direct(slice(None))))
        counts = None
        for region in _iter_region_slices(self.array.shape, itemsize, self.strip_bytes):
            block = to_device(self.array._read_direct(region), dev)
            xp = array_namespace(block)
            if edges is None:
                # int bins over an explicit range: numpy's own uniform-bin path, per block
                # identical to the whole-array call (same range, same dtype).
                c, _ = xp.histogram(block, bins=self.bins, range=self.range)
            else:
                c, _ = xp.histogram(block, bins=xp.asarray(edges))
            counts = c if counts is None else counts + c
        if counts is None:                   # empty array
            return np.zeros(self.shape, dtype=np.int64)
        return asnumpy(counts).astype(np.int64)


def histogram(array, bins=256, range=None, strip_bytes=_DEFAULT_STRIP_BYTES, device=None):
    """LAZY streaming histogram over the whole array -- the substrate for global thresholds.

    Returns ``(counts, bin_edges)`` like ``numpy.histogram``, both lazy DynamicArrays (as
    with dask, ``counts`` is lazy; here the edges are too, since without ``range`` they
    depend on the data). Nothing is read until one is computed/written/converted. It
    matches numpy exactly, including numpy's rule for a constant array (range widened to
    v - 0.5 .. v + 0.5) and its error for a non-finite autodetected range.

    ``bins`` is an int or an explicit edges array. With an int and no ``range``, the range
    is the data's min/max: two 0-d reductions that compute()/io.write fuse into ONE pass,
    then one pass for the counts. With ``range`` or explicit edges: one pass. The counts are
    computed once and cached (io.clear_cache() / arr.clear_cache() forget them).

    Memory-bound: the output size is fixed by ``bins``; regions of ``strip_bytes`` are
    streamed. For a per-channel/plane histogram, slice first: ``histogram(da[channel])``.
    ``device`` (None=inherit, 'cpu', 'cuda') runs the accumulation there; results are host.
    """
    from dyna_zarr.dynamic_array import DynamicArray
    from ..utils import parse_dtype
    src_dtype = parse_dtype(array.dtype)[0]
    size = int(np.prod(array.shape, dtype=np.int64))

    if not np.isscalar(bins):
        edges_np = np.asarray(bins)
        if edges_np.ndim != 1 or edges_np.size < 2 or np.any(np.diff(edges_np) < 0):
            raise ValueError("bins must be an int or a 1-D, monotonically increasing array "
                             "of at least 2 edges")
        edges, nbins, range_ = DynamicArray(edges_np), None, None
    else:
        nbins = int(bins)
        if nbins < 1:
            raise ValueError(f"bins must be a positive integer, got {bins}")
        if range is not None:
            range_ = (float(range[0]), float(range[1]))
            if range_[0] > range_[1]:
                raise ValueError("max must be larger than min in range parameter.")
            # computed by numpy itself on an empty array: no data needed
            edges = DynamicArray(np.histogram_bin_edges(np.zeros(0, dtype=src_dtype),
                                                        bins=nbins, range=range_))
        elif size == 0:
            range_ = None                    # numpy: an empty array bins over (0, 1)
            edges = DynamicArray(np.histogram_bin_edges(np.zeros(0, dtype=src_dtype), bins=nbins))
        else:
            range_ = None
            edges = array._with_transform(HistogramEdgesTransform(
                min(array, device=device), max(array, device=device), nbins, src_dtype))

    counts = array._with_transform(
        HistogramTransform(array, edges, nbins if range_ is not None else None, range_,
                           strip_bytes, device))
    return counts, edges


def unique(array, strip_bytes=_DEFAULT_STRIP_BYTES, device=None):
    """Streaming distinct values over the whole array (like ``numpy.unique``: a sorted 1-D
    array of the distinct values). Memory-bounded by the running set of distinct values plus
    one region, so it is cheap when there are few distinct values (e.g. a label image) and
    grows with that count otherwise. EAGER (unlike ``histogram``): returns a host numpy
    array, since its length depends on the data and a lazy array needs a known shape.
    """
    from ..utils import parse_dtype
    dev = resolve_device(device)
    dt = parse_dtype(array.dtype)[0]
    if 0 in tuple(array.shape):                # empty array -> no distinct values
        return np.array([], dtype=dt)
    acc = None
    for region in _iter_region_slices(array.shape, dt.itemsize, strip_bytes):
        block = to_device(array._read_direct(region), dev)
        xp = array_namespace(block)
        u = xp.unique(block)
        acc = u if acc is None else xp.unique(xp.concatenate([acc, u]))
    if acc is None:                       # empty array
        return np.array([], dtype=dt)
    return asnumpy(acc)


__all__ = [
    "ReduceTransform", "reduce",
    "min", "max", "sum", "prod", "mean", "any", "all",
    "var", "std", "argmin", "argmax", "median", "histogram", "unique",
]
