"""Downsampling: ``coarsen`` reduces non-overlapping windows of integer ``factors`` (downsampling).

``coarsen(x, factors, method)`` with ``method`` in mean / median / min / max / mode
reproduces TensorStore's ``ts.downsample`` exactly, so a pyramid level is the same whichever
writes it:

- every axis gets ``ceil(n / f)`` output elements; a partial window at the far edge is
  reduced over the elements it has (nothing is dropped);
- mean: exact for floats; for integers the window mean is rounded half to even (1.5 -> 2,
  2.5 -> 2) in the input dtype;
- median: the LOWER middle element of the sorted window (never an average), so the result
  is always a value present in the window;
- min / max: NaN is ignored; an all-NaN window gives +inf (min) / -inf (max);
- mode: the most frequent value; ties go to the smallest value.

One deliberate difference: with NaN in a float window, TensorStore's median and mode
depend on the order of the elements (NaN breaks the comparisons its selection relies on).
Here NaN is ignored by both, and an all-NaN window gives NaN. Windows without NaN are
bit-identical.

The output dtype is the input dtype. It is pull-based like every dyna op: an output region
reads exactly the input windows under it.
"""
import itertools
import math

import numpy as np

from ._base import Transform
from ._backend import array_namespace
from .structural import _norm_key

METHODS = ("mean", "median", "min", "max", "mode")


#: Window sizes up to this use a sorting network for median / mode (elementwise, GIL-free,
#: a few times faster than TensorStore at 2x2x2); larger windows sort per window.
_NETWORK_MAX_WINDOW = 64


def _window_views(x, w):
    """The W = prod(w) strided views of ``x`` (every axis a whole number of windows), one
    per position inside the window, in the window's C order. No copies."""
    nd = x.ndim
    n = [s // f for s, f in zip(x.shape, w)]
    xr = x.reshape(tuple(itertools.chain.from_iterable(zip(n, w))))
    return [xr[tuple(itertools.chain.from_iterable((slice(None), o) for o in off))]
            for off in itertools.product(*(range(f) for f in w))]


_NETWORKS = {}


def _sorting_network(n):
    """Comparator pairs sorting ``n`` values: Batcher's odd-even merge sort on the next
    power of two, keeping only the comparators inside ``n`` (the padding acts as +inf)."""
    if n in _NETWORKS:
        return _NETWORKS[n]
    p = 1
    while p < n:
        p *= 2
    pairs = []

    def merge(lo, hi, r):
        step = r * 2
        if step < hi - lo:
            merge(lo, hi, step)
            merge(lo + r, hi, step)
            pairs.extend((i, i + r) for i in range(lo + r, hi - r, step))
        else:
            pairs.append((lo, lo + r))

    def sort(lo, hi):
        if hi - lo >= 1:
            mid = lo + (hi - lo) // 2
            sort(lo, mid)
            sort(mid + 1, hi)
            merge(lo, hi, 1)

    sort(0, p - 1)
    _NETWORKS[n] = [(i, j) for i, j in pairs if j < n]
    return _NETWORKS[n]


def _sorted_views(views, xp):
    """The window values sorted ascending, as W arrays (position k = k-th smallest)."""
    a = [xp.array(v) for v in views]                # contiguous copies: one input's worth
    tmp = xp.empty_like(a[0])
    for i, j in _sorting_network(len(a)):
        xp.minimum(a[i], a[j], out=tmp)
        xp.maximum(a[i], a[j], out=a[j])
        a[i], tmp = tmp, a[i]
    return a


def _mode_sorted(s, xp):
    """Most frequent value of sorted columns ``s``; ties -> the smallest value (runs are in
    ascending order and only a strictly longer run replaces the best)."""
    best = s[0].copy()
    best_len = xp.ones(best.shape, dtype=np.uint8)      # W <= 64 on this path
    run = xp.ones(best.shape, dtype=np.uint8)
    for i in range(1, len(s)):
        same = s[i] == s[i - 1]
        run += 1                    # run = run + 1 where the value repeats, else 1
        run *= same
        run |= ~same
        better = run > best_len
        xp.copyto(best, s[i], where=better)
        xp.copyto(best_len, run, where=better)
    return best


def _window_values(views, xp):
    """(n0, n1, ..., W): each window's values on the last axis (one copy)."""
    return xp.stack(views, axis=-1)


def _mode_last(v, xp):
    """Most frequent value along the last axis; ties -> the smallest value."""
    s = xp.sort(v, axis=-1)
    W = s.shape[-1]
    if W == 1:
        return s[..., 0]
    idx = xp.arange(W)
    new_run = xp.concatenate(
        [xp.ones(s.shape[:-1] + (1,), dtype=bool), s[..., 1:] != s[..., :-1]], axis=-1)
    run_start = xp.maximum.accumulate(xp.where(new_run, idx, 0), axis=-1)
    run_len = idx - run_start + 1               # length of the run so far, at each position
    # argmax = FIRST position reaching the longest run length; runs are in ascending value
    # order, so on a tie the smallest value gets there first
    at = xp.argmax(run_len, axis=-1)
    return xp.take_along_axis(s, at[..., None], axis=-1)[..., 0]


def _int_accumulator(dtype, count):
    """An integer dtype that holds the sum of ``count`` values of ``dtype`` exactly."""
    if dtype.kind == "b":
        dtype = np.dtype(np.uint8)
    info = np.iinfo(dtype)
    for cand in ((np.int32, np.int64) if dtype.kind == "i" else (np.uint32, np.uint64)):
        c = np.iinfo(cand)
        if int(info.max) * count <= c.max and int(info.min) * count >= c.min:
            return np.dtype(cand)
    return np.dtype(np.int64 if dtype.kind == "i" else np.uint64)


def _divmod_sum64(views, W, xp):
    """``divmod(sum(views), W)`` for 64-bit integers, exactly: the sum of W int64 /
    uint64 values overflows 64 bits, so the high and low 32-bit halves are summed
    separately (each fits) and recombined by exact integer division."""
    acc = views[0].dtype
    shift, mask = acc.type(32), acc.type(0xFFFFFFFF)
    H = (views[0] >> shift).astype(acc)          # arithmetic shift: floor for negatives
    L = (views[0] & mask).astype(acc)            # always 0 .. 2**32 - 1
    for v in views[1:]:
        H += v >> shift
        L += v & mask
    Wt = acc.type(W)
    qh, rh = xp.divmod(H, Wt)                    # rh in [0, W)
    q2, r = xp.divmod(rh * acc.type(1 << 32) + L, Wt)   # < W * 2**33: fits for W < 2**30
    return qh * acc.type(1 << 32) + q2, r


def _reduce_uniform(x, w, method):
    """Reduce ``x`` whose every axis is a whole number of windows ``w``.

    The window is taken apart into its W strided views, so every method is a sequence of
    elementwise ufunc passes (which release the GIL) rather than a reduction over several
    strided axes at once."""
    xp = array_namespace(x)
    dtype = x.dtype
    views = _window_views(x, w)
    W = len(views)
    if method in ("min", "max"):
        if dtype.kind == "f":
            op = xp.fmin if method == "min" else xp.fmax
        else:
            op = xp.minimum if method == "min" else xp.maximum
        # separable (and order-free): reduce one axis at a time; each pass shrinks the data
        r = x
        for a, f in enumerate(w):
            if f > 1:
                parts = [r[(slice(None),) * a + (slice(i, None, f),)] for i in range(f)]
                acc = op(parts[0], parts[1])
                for part in parts[2:]:
                    op(acc, part, out=acc)
                r = acc
        r = r.copy() if r is x else r
        if dtype.kind == "f":
            # an all-NaN window gives the reduction's identity, as in TensorStore
            r[xp.isnan(r)] = xp.inf if method == "min" else -xp.inf
        return r
    if method == "mean":
        if dtype.kind == "f":
            # TensorStore's order: a sequential sum over the window in C order,
            # accumulated in the input dtype (float32 for float16), then divided -
            # bit-identical, not just close
            acc = np.float32 if dtype.itemsize < 4 else dtype
            total = views[0].astype(acc, copy=True)
            for v in views[1:]:
                xp.add(total, v, out=total, casting="unsafe")
            return (total / np.float64(W)).astype(dtype)
        if dtype.kind in "iu" and dtype.itemsize == 8:
            q, r = _divmod_sum64(views, W, xp)
        else:
            if W == 1:
                return views[0].copy()
            acc = _int_accumulator(dtype, W)
            total = xp.add(views[0], views[1], dtype=acc)
            for v in views[2:]:
                xp.add(total, v, out=total, casting="unsafe")
            if W & (W - 1) == 0:
                # W = 2**k: round half to even by shifts, in place (the common pyramid case)
                k = W.bit_length() - 1
                odd = (total >> k) & 1
                total += odd
                total += acc.type(W // 2 - 1)
                total >>= k
                return total.astype(dtype)
            q, r = xp.divmod(total, acc.type(W))
        # the mean rounded half to even, in exact integer arithmetic (floor division, so
        # it holds for negative sums too)
        up = (2 * r > W) | ((2 * r == W) & ((q & 1) == 1))
        return (q + up).astype(dtype)
    if method not in ("median", "mode"):
        raise ValueError(f"unknown coarsen method {method!r}; expected one of {METHODS}")
    has_nan = dtype.kind == "f" and bool(xp.isnan(x).any())
    if W <= _NETWORK_MAX_WINDOW and not has_nan:
        s = _sorted_views(views, xp)
        return s[(W - 1) // 2] if method == "median" else _mode_sorted(s, xp)
    v = _window_values(views, xp)
    if method == "mode":
        return _mode_last(v, xp)            # NaN never repeats, so it never wins a run
    if has_nan:
        # NaN ignored: the lower middle of the window's non-NaN values (NaN sorts last);
        # NaN for an all-NaN window
        valid = xp.sum(~xp.isnan(v), axis=-1)
        k = xp.maximum(valid - 1, 0) // 2
        return xp.take_along_axis(xp.sort(v, axis=-1), k[..., None], axis=-1)[..., 0]
    k = (W - 1) // 2
    return xp.partition(v, k, axis=-1)[..., k]


#: Blocks are reduced in window-aligned tiles of about this many input bytes, so the
#: several passes each method makes stay in cache (measured: 1.4-2x faster under 8 threads
#: than reducing a 32 MiB block in one go).
_TILE_BYTES = 2 << 20


def coarsen_block(x, factors, method):
    """Reduce a block whose origin is window-aligned; its far edge may cut windows short
    (the array's own edge), and those are reduced over the elements they have. NumPy or
    CuPy. The block function of ``coarsen``; usable on its own (e.g. with dask)."""
    xp = array_namespace(x)
    factors = tuple(int(f) for f in factors)
    if x.nbytes > _TILE_BYTES:
        # split along the first axis that has more than one window, at window boundaries
        for a, (s, f) in enumerate(zip(x.shape, factors)):
            nwin = -(-s // f)
            if nwin > 1:
                per = max(1, int(nwin * _TILE_BYTES // x.nbytes))
                if per >= nwin:
                    break
                out = xp.empty(tuple(-(-n // g) for n, g in zip(x.shape, factors)),
                               dtype=x.dtype)
                for w0 in range(0, nwin, per):
                    w1 = min(nwin, w0 + per)
                    src = (slice(None),) * a + (slice(w0 * f, min(w1 * f, s)),)
                    dst = (slice(None),) * a + (slice(w0, w1),)
                    out[dst] = coarsen_block(x[src], factors, method)
                return out
    out_shape = tuple(-(-s // f) for s, f in zip(x.shape, factors))
    # per axis: the run of whole windows, then (at an edge) one short window
    pieces = []
    for s, f in zip(x.shape, factors):
        q, t = divmod(s, f)
        axis = []
        if q:
            axis.append((slice(0, q * f), slice(0, q), f))
        if t:
            axis.append((slice(q * f, s), slice(q, q + 1), t))
        pieces.append(axis)
    if all(len(p) == 1 for p in pieces):
        return _reduce_uniform(x, tuple(p[0][2] for p in pieces), method)
    out = xp.empty(out_shape, dtype=x.dtype)
    for combo in itertools.product(*pieces):
        src = tuple(c[0] for c in combo)
        dst = tuple(c[1] for c in combo)
        out[dst] = _reduce_uniform(x[src], tuple(c[2] for c in combo), method)
    return out


class CoarsenTransform(Transform):
    """Lazy ``coarsen``: output element ``i`` along an axis reduces input ``[i*f, (i+1)*f)``."""

    def __init__(self, array, factors, method):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"unknown coarsen method {method!r}; expected one of {METHODS}")
        factors = tuple(int(f) for f in factors)
        if len(factors) != array.ndim:
            raise ValueError(f"factors {factors} must have one entry per axis "
                             f"(the array has {array.ndim})")
        if any(f < 1 for f in factors):
            raise ValueError(f"factors must be >= 1, got {factors}")
        self.array = array
        self.factors = factors
        self.method = method
        self.shape = tuple(-(-s // f) for s, f in zip(array.shape, factors))
        self.dtype = array.dtype
        # an input chunk of c covers c / f output elements: keep the grid where it divides
        self.chunks = (None if array.chunks is None else
                       tuple(max(1, int(c) // f) for c, f in zip(array.chunks, factors)))
        # each output element reads prod(factors) input elements: io.write budgets a
        # region by what it READS, so it must know (see io._region_cost)
        self.read_amplification = math.prod(factors)

    def _pieces(self, src):
        """Cut the input range ``src`` into pieces on the source's chunk grid (rounded to
        whole windows), each with the output range it produces. One piece if the source
        has no grid or the range fits in one."""
        grid = self.array.chunks
        per_axis = []
        for (lo, hi), f, c in zip(((s.start, s.stop) for s in src), self.factors,
                                  grid if grid is not None else (None,) * len(src)):
            P = hi - lo if c is None else max(f, -(-int(c) // f) * f)
            cuts, pos = [], lo
            while pos < hi:
                nxt = min(hi, (pos // P + 1) * P)      # the next grid line (a window edge)
                cuts.append((slice(pos, nxt), slice((pos - lo) // f, -(-(nxt - lo) // f))))
                pos = nxt
            per_axis.append(cuts)
        return itertools.product(*per_axis)

    def read(self, key):
        axinfo = _norm_key(key, self.shape)
        src = tuple(slice(start * f, min(stop * f, n))
                    for (_i, start, stop, _s), f, n in zip(axinfo, self.factors,
                                                           self.array.shape))
        # Read and reduce one source chunk at a time, while it is still in cache: decoding
        # a whole region to memory and reading it back for the reduction was ~1.5x slower
        # than TensorStore, which reduces each chunk as it decodes it. Only one chunk's
        # input is held at a time, too.
        pieces = list(self._pieces(src))
        if len(pieces) == 1:
            block = coarsen_block(self.array._read_direct(src), self.factors, self.method)
        else:
            block = None
            for piece in pieces:
                part = coarsen_block(self.array._read_direct(tuple(p[0] for p in piece)),
                                     self.factors, self.method)
                if block is None:
                    xp = array_namespace(part)
                    block = xp.empty(tuple(-(-(s.stop - s.start) // f)
                                           for s, f in zip(src, self.factors)),
                                     dtype=part.dtype)
                block[tuple(p[1] for p in piece)] = part
        block = block[tuple(slice(None, None, step) for (_i, _a, _b, step) in axinfo)]
        xp = array_namespace(block)
        for a in sorted((i for i, (is_int, *_r) in enumerate(axinfo) if is_int), reverse=True):
            block = xp.squeeze(block, axis=a)
        return block


def coarsen(array, factors, method="mean"):
    """Downsample ``array`` by integer ``factors`` (one per axis; 1 = keep the axis),
    reducing each window with ``method``: 'mean', 'median', 'min', 'max' or 'mode'.
    Matches ``tensorstore.downsample`` exactly - see the module docstring for the edge,
    rounding, median, NaN and tie rules. For plain decimation, slice: ``x[::2, ::2]``."""
    if all(int(f) == 1 for f in factors):
        return array
    return array._with_transform(CoarsenTransform(array, factors, method))


__all__ = ["coarsen", "coarsen_block"]
