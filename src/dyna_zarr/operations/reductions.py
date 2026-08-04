"""Reductions -- the reductive/destructive taxonomy cell.

A reduction collapses one or more axes. The pull model can't cheaply slice a *reduced*
axis (the output element depends on the whole axis), so ``read(key)`` reads only the
kept-axis tile the key asks for but **streams the reduced axis in bounded chunks**,
combining them with an associative reducer (min/max/sum/prod/mean/any/all). That keeps it
memory-bound even for a huge reduced axis, and the result is chunk/region-invariant and
exact vs numpy. Step/integer indices on the output are applied after the reduce (same
crop-at-the-end trick as map_overlap).

``axis=None`` reduces everything to a 0-d result, still streamed. The result is a lazy
DynamicArray (chainable / writable); DynamicArray.min()/max() wrap it and compute eagerly.
"""

import builtins
import numpy as np
from typing import Optional, Tuple

from ._base import Transform, _is_int_index
from ._backend import array_namespace, asnumpy, resolve_device, to_device

_DEFAULT_STRIP_BYTES = 64 * 1024 * 1024   # per-read streaming budget for the reduced axis


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


def _arg_axis(ax):
    if len(ax) == 1:
        return ax[0]
    if len(ax) == 0:
        return None
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
    "argmin": _Reducer(associative=False, direct=lambda xp, b, ax: xp.argmin(b, axis=_arg_axis(ax))),
    "argmax": _Reducer(associative=False, direct=lambda xp, b, ax: xp.argmax(b, axis=_arg_axis(ax))),
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

    def _infer_dtype(self):
        sample = np.ones((2,) * self.array.ndim, dtype=self._in_dtype)
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
        out_ndim = len(self.shape)
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (out_ndim - len(key))

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

        block = self._stream(input_slices)               # kept-axis order, reduced axes gone
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


def histogram(array, bins=256, range=None, strip_bytes=_DEFAULT_STRIP_BYTES, device=None):
    """Streaming histogram over the whole array -- the substrate for global thresholds.

    A histogram is an associative, memory-bound reduction: it equals the sum of per-region
    histograms sharing the same bins, and the output size is fixed (``bins``) regardless of
    array size. Returns ``(counts, bin_edges)`` like ``numpy.histogram``, and matches it
    exactly. ``bins`` may be an int (with an optional ``range`` (lo, hi); if omitted, the
    data min/max are found in one streaming pass) or a precomputed edges array. For a
    per-channel/plane histogram, slice first: ``histogram(da[channel])``. ``device``
    (None=inherit, 'cpu', 'cuda') runs the accumulation on that device; the result is host.
    """
    from ..utils import parse_dtype
    dev = resolve_device(device)
    itemsize = parse_dtype(array.dtype)[0].itemsize
    bins_is_edges = not np.isscalar(bins)
    if not bins_is_edges and range is None:
        lo = float(min(array, device=device).compute())   # streaming min/max (memory-bound)
        hi = float(max(array, device=device).compute())
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo == hi:
            hi = lo + 1.0
        range = (lo, hi)
    if range is not None:
        range = (float(range[0]), float(range[1]))

    counts = None
    edges = None
    for region in _iter_region_slices(array.shape, itemsize, strip_bytes):
        block = to_device(array._read_direct(region), dev)
        xp = array_namespace(block)
        if bins_is_edges:
            c, edges = xp.histogram(block, bins=xp.asarray(bins))
        else:
            c, edges = xp.histogram(block, bins=bins, range=range)
        counts = c if counts is None else counts + c
    if counts is None:   # empty array
        edges = np.asarray(bins, dtype=float) if bins_is_edges else \
            np.linspace(range[0], range[1], int(bins) + 1)
        counts = np.zeros(len(edges) - 1, dtype=np.int64)
    # bring back to host (counts/edges may be cupy) and return like numpy.histogram
    return asnumpy(counts).astype(np.int64), asnumpy(edges)


def unique(array, strip_bytes=_DEFAULT_STRIP_BYTES, device=None):
    """Streaming distinct values over the whole array (like ``numpy.unique``: a sorted 1-D
    array of the distinct values). Memory-bounded by the running set of distinct values plus
    one region, so it is cheap when there are few distinct values (e.g. a label image) and
    grows with that count otherwise. Eager, like ``histogram``: returns a host numpy array.
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
