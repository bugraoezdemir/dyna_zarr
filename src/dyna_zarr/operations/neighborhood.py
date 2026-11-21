"""Neighbourhood primitive (pull-model ``map_overlap``) and the filters built on it.

A neighbourhood op is shape-preserving but each output element depends on a local window
of the input (halo/``depth``). In the pull model this is just map_blocks with a widened
read window: to produce the block for ``key`` we read ``key`` expanded by ``depth`` on
every side (padding at the true array edges per ``boundary``), apply the (shape-preserving)
function to that expanded block, then crop the halo back off. Because every read pulls its
own halo, the result is independent of the region size that drove it -- i.e. chunk/region
invariant, and exact vs applying the function to the whole array (given ``depth`` >= the
function's radius and a matching ``boundary``).

At an edge of the array the halo is what the array has there, and the function applies
``boundary`` itself, as it would to the whole array. ``wrap`` is the exception: the
function would wrap the block around itself, so a wrap read takes the halo from the far
end of the axis instead (see _wrapped_runs).

This is the ``conservative . unary . neighbourhood`` taxonomy cell.
"""

import numpy as np
from typing import Optional, Union, Tuple, Dict

from ._base import Transform, _is_int_index
from ._backend import array_namespace, ndimage_namespace, resolve_device, to_device


# boundary name (scipy.ndimage convention) -> (numpy.pad mode, extra pad kwargs) for the
# true array edges.
_BOUNDARY_TO_NPPAD = {
    "reflect": ("symmetric", {}),   # scipy 'reflect': (d c b a | a b c d) -- edge duplicated
    "mirror": ("reflect", {}),      # scipy 'mirror':  (d c b | a b c d) -- edge not duplicated
    "nearest": ("edge", {}),
    "wrap": ("wrap", {}),
    "constant": ("constant", {}),
    # odd-reflection = linear extrapolation (2*edge - inner); makes a central difference at
    # the boundary equal the one-sided difference, so gradient() matches numpy at the edges.
    "odd_reflect": ("reflect", {"reflect_type": "odd"}),
}


def _normalize_depth(depth, ndim) -> Tuple[int, ...]:
    """Normalize ``depth`` to a per-axis tuple of symmetric halo widths."""
    if isinstance(depth, dict):
        return tuple(int(depth.get(a, 0)) for a in range(ndim))
    if np.isscalar(depth):
        return (int(depth),) * ndim
    depth = tuple(int(d) for d in depth)
    if len(depth) != ndim:
        raise ValueError(f"depth {depth} does not match array ndim {ndim}")
    return depth


def _infer_overlap_dtype(func, array):
    """Infer output dtype by applying func to a small sample of the array's dtype."""
    from ..utils import parse_dtype
    np_dt = parse_dtype(array.dtype)[0]
    sample_shape = tuple(min(s, 3) for s in array.shape)
    try:
        return np.asarray(func(np.ones(sample_shape, dtype=np_dt))).dtype
    except Exception:
        return np_dt


class MapOverlapTransform(Transform):
    """Apply a shape-preserving neighbourhood ``func`` with a ``depth`` halo, lazily.

    ``func`` receives an expanded block (input window padded to include the halo) and must
    return an array of the *same shape*; the halo is then trimmed. ``depth`` may be an int
    (all axes), a per-axis sequence, or a ``{axis: depth}`` dict. ``boundary`` (scipy.ndimage
    names: reflect/mirror/nearest/wrap/constant) controls how the true array edges are
    extended. For an exact match to ``func`` applied to the whole array, ``depth`` must be
    >= the function's radius and ``boundary`` must match the function's edge mode.
    """

    def __init__(self, array, func, depth, boundary="reflect", dtype=None, name=None,
                 device=None, block_info=False, align=None, kernel=None):
        super().__init__()
        #: which named filter ``func`` is, with its arguments (``{"name": ..., ...}``), set
        #: by the filters below; dyna's native engine reads a transform it recognises
        #: (Workplan DR7, see _lowering). A ``func`` alone is never recognised.
        self.kernel = kernel
        if boundary not in _BOUNDARY_TO_NPPAD:
            raise ValueError(f"unknown boundary {boundary!r}; expected one of "
                             f"{sorted(_BOUNDARY_TO_NPPAD)}")
        self.array = array
        self.func = func
        self.device = device
        #: when True, ``func`` is called ``func(block, location)`` where ``location`` is the
        #: per-axis ``(start, stop)`` of the CORE region this block covers (the requested
        #: key, halo excluded) - the pull-model analogue of dask's ``block_info``
        #: ``array-location``. Lets a position-aware func map a global coordinate ``g`` to a
        #: block index as ``depth + (g - start)`` (uniform, edges included), e.g. to apply a
        #: per-tile lookup by global position. Off by default (the scipy filters are
        #: position-agnostic).
        self.block_info = block_info
        #: optional per-axis read-alignment grid. When set, every read EXPANDS its core to
        #: whole `align`-sized cells (floor start / ceil stop) before adding the halo, so the
        #: func always receives whole cells - then the result is cropped back to the requested
        #: region. Lets a position-aware func that needs whole tiles (e.g. tilewise-ccl Phase B's
        #: per-tile relabel) stay CORRECT for any requested region size, independent of the
        #: consumer's chunking (a region smaller than a cell just re-reads its cell). None =
        #: off (exact region reads).
        self.align = _normalize_depth(align, array.ndim) if align is not None else None
        self.name = name or getattr(func, "__name__", "map_overlap")
        self.depth = _normalize_depth(depth, array.ndim)
        self.boundary = boundary
        self.shape = array.shape
        self.chunks = array.chunks
        self.dtype = np.dtype(dtype) if dtype is not None else _infer_overlap_dtype(func, array)

    def read(self, key):
        ndim = self.array.ndim
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (ndim - len(key))

        read_runs = []        # per axis, the runs of input indices read, joined in order
        pad_widths = []       # (before, after) padding that restores the halo at true edges
        crop_slices = []      # crop the func output back to the core region + apply step
        squeeze_axes = []
        location = []         # per-axis (start, stop) of the CORE region (for block_info)
        for a in range(ndim):
            k = key[a]
            size = self.array.shape[a]
            d = self.depth[a]
            if _is_int_index(k):
                idx = int(k) if k >= 0 else size + int(k)
                start, stop, step = idx, idx + 1, 1
                squeeze_axes.append(a)
            else:
                start, stop, step = k.indices(size)
                if step < 0:
                    raise NotImplementedError("map_overlap: negative-step reads not supported")

            # expand the CORE to whole `al`-cells (floor start / ceil stop, clamped); the
            # func then sees whole cells and we crop back to the requested [start, stop).
            # This applies to an INTEGER index too: the axis is squeezed only AFTER the crop,
            # so alignment and squeezing are independent. Skipping it here (as an earlier
            # version did, "never align a squeezed axis") handed a position-aware func a
            # 1-voxel slab instead of a whole cell - e.g. tilewise-ccl's Phase B then
            # re-labelled that slab, produced local ids that did not match the tile's LUT,
            # and raised IndexError or silently returned wrong labels for `labels[z]`.
            al = self.align[a] if self.align is not None else 1
            astart = (start // al) * al
            astop = min(size, -(-stop // al) * al)

            # The halo is capped PER SIDE at the real data that side can supply.
            # Past that the read clamps to the axis anyway and only the `boundary`
            # padding grows, which adds no information: pre-padding a block and
            # letting the func apply the same mode itself agree EXACTLY (0.0,
            # float64) once the halo is at least the kernel's own reach, which is
            # the halo any correct caller already asks for. Below that they differ,
            # but by kernel truncation, which the uncapped path suffers too.
            #
            # Per SIDE, not one shared depth. A single
            # `d = min(d, max(astart, size - astop))` collapses to 0 on a core that
            # spans the whole axis, dropping the boundary border entirely, and the
            # crop below then reads from the wrong offset: gaussian_s2,
            # gaussian_aniso, gaussian_laplace and gauss_grad_mag all came out
            # wrong at every edge.
            #
            # The win is on short axes. A scalar sigma on a 5D
            # (2, 3, 48, 384, 384) volume gives depth 4 on every axis, so uncapped
            # the length-2 axis pads to 10 and the length-3 to 11 -- a ~19x block
            # inflation that scipy's separable passes then multiply again
            # (17.7 GB peak measured on a 162 MB array).
            d_lo = min(d, astart)
            d_hi = min(d, size - astop)

            # wrap: the function would wrap the block around ITSELF, not the array, so
            # a halo that stops at an edge must come from the far end of the axis. The
            # block then holds the whole halo on both sides and the function's own wrap
            # at the block's edges reaches only the halo, which the crop drops. A block
            # that spans the whole axis anyway (both sides capped) is the axis itself,
            # and the function's wrap is then the array's.
            if self.boundary == "wrap" and (astart - d > 0 or astop + d < size) \
                    and (d_lo < d or d_hi < d):
                d_lo = d_hi = d
                read_runs.append(_wrapped_runs(astart - d, astop + d, size))
            else:
                # read the ALIGNED core +/- halo (already within [0, size) by the cap)
                read_runs.append([slice(astart - d_lo, astop + d_hi)])
            pad_widths.append((0, 0))
            # The aligned core sits at block[d_lo : d_lo+(astop-astart)] -- d_lo, the
            # halo actually present on the leading side, NOT the requested depth.
            # The requested region is the sub-window [start, stop) inside it.
            crop_slices.append(
                slice(d_lo + (start - astart), d_lo + (stop - astart), step))
            location.append((astart, astop))

        block = to_device(_read_runs(self.array, read_runs), resolve_device(self.device))
        xp = array_namespace(block)
        if any(pb or pa for pb, pa in pad_widths):
            mode, pad_kw = _BOUNDARY_TO_NPPAD[self.boundary]
            block = xp.pad(block, pad_widths, mode=mode, **pad_kw)

        # func gets the padded block (dispatching ndimage on its device); with block_info it
        # also gets the core (start, stop) per axis so it can act by global position.
        out = self.func(block, tuple(location)) if self.block_info else self.func(block)
        out = out[tuple(crop_slices)]
        for a in sorted(squeeze_axes, reverse=True):
            out = xp.squeeze(out, axis=a)
        return out


def _wrapped_runs(lo, hi, size):
    """The input indices ``lo .. hi`` (hi excluded) taken modulo ``size``, as slices of
    consecutive indices in order: a halo past an edge continues at the far end, and one
    longer than the axis repeats it."""
    runs = []
    p = lo
    while p < hi:
        q = p % size
        n = min(hi - p, size - q)
        runs.append(slice(q, q + n))
        p += n
    return runs


def _read_runs(array, runs):
    """Read the box whose indices along axis ``a`` are the slices ``runs[a]``, joined in
    order (one read when every axis has a single slice)."""
    def part(axis, prefix):
        if axis == len(runs):
            return array._read_direct(tuple(prefix))
        pieces = [part(axis + 1, prefix + [r]) for r in runs[axis]]
        if len(pieces) == 1:
            return pieces[0]
        return array_namespace(pieces[0]).concatenate(pieces, axis=axis)
    return part(0, [])


# --------------------------------------------------------------------------- #
# depth helpers for the scipy-backed filters
# --------------------------------------------------------------------------- #

def _as_per_axis(value, ndim):
    if np.isscalar(value):
        return (value,) * ndim
    value = tuple(value)
    if len(value) != ndim:
        raise ValueError(f"expected a scalar or length-{ndim} sequence, got {value}")
    return value


def _gaussian_depth(sigma, ndim, truncate):
    sig = _as_per_axis(sigma, ndim)
    # scipy's 1-D Gaussian half-width: int(truncate * sigma + 0.5)
    return tuple(int(truncate * float(s) + 0.5) for s in sig)


def _size_depth(size, ndim):
    sz = _as_per_axis(size, ndim)
    # max half-extent of a length-n window (covers even sizes too)
    return tuple(int(n) // 2 for n in sz)


# --------------------------------------------------------------------------- #
# Public ops -- map_overlap primitive + scipy-backed neighbourhood filters
# --------------------------------------------------------------------------- #

def map_overlap(array, func, depth, boundary="reflect", dtype=None, name=None, device=None,
                block_info=False, align=None):
    """Apply a shape-preserving neighbourhood ``func`` with a ``depth`` halo, lazily and
    chunk-invariantly. Every read pulls its own halo, so the result is independent of the
    region size and exact vs applying ``func`` to the whole array (given depth >= radius and
    matching ``boundary``). The filters below wrap this. ``device`` (None=inherit, 'cpu',
    'cuda') runs it on that device.

    ``block_info=True`` makes the call ``func(block, location)``, where ``location`` is the
    per-axis ``(start, stop)`` of the CORE region the block covers (halo excluded) - the
    pull-model analogue of dask's ``block_info`` ``array-location``. A position-aware func
    can then map a global coordinate ``g`` to a block index via ``depth + (g - start)``
    (uniform across the array, edges included), e.g. to apply a per-tile lookup keyed by
    global position (as in tilewise-ccl's Phase-B label application).

    ``align`` (int or per-axis) expands every read's core to whole ``align``-sized cells
    before adding the halo, so a position-aware func that must see WHOLE cells (e.g. a
    per-tile relabel) stays correct for ANY requested region size - independent of the
    consumer's chunking. A requested region smaller than a cell just re-reads its cell
    (correct but with read amplification), so for speed drive it with regions that are
    whole aligned cells (e.g. write with ``chunks == align``)."""
    return array._with_transform(
        MapOverlapTransform(array, func, depth, boundary=boundary, dtype=dtype,
                            name=name, device=device, block_info=block_info, align=align)
    )


def _named(array, func, depth, boundary, name, device, kernel):
    """map_overlap for one of the named filters below, recording which it is (``kernel``)
    so the native engine can read it."""
    return array._with_transform(
        MapOverlapTransform(array, func, depth, boundary=boundary, name=name,
                            device=device, kernel=kernel))


#: gaussian_filter keywords the engine follows; any other keeps the Python path
_GAUSSIAN_KW = {"order", "cval"}


def gaussian_filter(array, sigma, boundary="reflect", truncate=4.0, device=None, **kw):
    depth = _gaussian_depth(sigma, array.ndim, truncate)
    func = lambda b: ndimage_namespace(b).gaussian_filter(
        b, sigma=sigma, mode=boundary, truncate=truncate, **kw)
    kernel = None
    if set(kw) <= _GAUSSIAN_KW:
        kernel = {"name": "gaussian_filter", "sigma": sigma, "order": kw.get("order", 0),
                  "mode": boundary, "cval": kw.get("cval", 0.0), "truncate": truncate}
    return _named(array, func, depth, boundary, "gaussian_filter", device, kernel)


#: keywords the engine follows for the filters below; any other keeps the Python path
_CVAL_KW = {"cval"}


def _size_kernel(name, size, boundary, kw):
    """The kernel spec of a filter with a window of ``size`` (None: not for the engine)."""
    if not set(kw) <= _CVAL_KW:
        return None
    return {"name": name, "size": size, "mode": boundary, "cval": kw.get("cval", 0.0)}


def uniform_filter(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).uniform_filter(b, size=size, mode=boundary, **kw)
    return _named(array, func, _size_depth(size, array.ndim), boundary, "uniform_filter",
                  device, _size_kernel("uniform_filter", size, boundary, kw))


def median_filter(array, size, boundary="reflect", device=None, **kw):
    """Median of each ``size`` window (scipy.ndimage.median_filter)."""
    func = lambda b: ndimage_namespace(b).median_filter(b, size=size, mode=boundary, **kw)
    return _named(array, func, _size_depth(size, array.ndim), boundary, "median_filter",
                  device, _size_kernel("median_filter", size, boundary, kw))


def rank_filter(array, rank, size, boundary="reflect", device=None, **kw):
    """The element of rank ``rank`` (0 the smallest, -1 the largest) of each ``size``
    window (scipy.ndimage.rank_filter)."""
    func = lambda b: ndimage_namespace(b).rank_filter(b, rank, size=size, mode=boundary, **kw)
    kernel = _size_kernel("rank_filter", size, boundary, kw)
    if kernel is not None:
        kernel["rank"] = rank
    return _named(array, func, _size_depth(size, array.ndim), boundary, "rank_filter",
                  device, kernel)


def percentile_filter(array, percentile, size, boundary="reflect", device=None, **kw):
    """The ``percentile`` (0 to 100, or -100 to 0 counted from the top) of each ``size``
    window, as scipy picks it: the element of rank ``int(n * percentile / 100)`` of the
    window's n values (scipy.ndimage.percentile_filter)."""
    func = lambda b: ndimage_namespace(b).percentile_filter(
        b, percentile, size=size, mode=boundary, **kw)
    kernel = _size_kernel("percentile_filter", size, boundary, kw)
    if kernel is not None:
        kernel["percentile"] = percentile
    return _named(array, func, _size_depth(size, array.ndim), boundary, "percentile_filter",
                  device, kernel)


def minimum_filter(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).minimum_filter(b, size=size, mode=boundary, **kw)
    return _named(array, func, _size_depth(size, array.ndim), boundary, "minimum_filter", device,
                  _size_kernel("minimum_filter", size, boundary, kw))


def maximum_filter(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).maximum_filter(b, size=size, mode=boundary, **kw)
    return _named(array, func, _size_depth(size, array.ndim), boundary, "maximum_filter", device,
                  _size_kernel("maximum_filter", size, boundary, kw))


def grey_erosion(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).grey_erosion(b, size=size, mode=boundary, **kw)
    return _named(array, func, _size_depth(size, array.ndim), boundary, "grey_erosion", device,
                  _size_kernel("grey_erosion", size, boundary, kw))


def grey_dilation(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).grey_dilation(b, size=size, mode=boundary, **kw)
    return _named(array, func, _size_depth(size, array.ndim), boundary, "grey_dilation", device,
                  _size_kernel("grey_dilation", size, boundary, kw))


def _kernel_depth(weights, ndim):
    """Halo per axis for a convolution/correlation kernel: half the kernel size (>= the
    kernel's reach on either side, so map_overlap stays exact vs the whole-array result)."""
    w = np.asarray(weights)
    if w.ndim != ndim:
        raise ValueError(
            f"weights ndim {w.ndim} must match array ndim {ndim}; use size-1 axes to leave "
            f"an axis untouched (e.g. a (1, ky, kx) kernel over a (z, y, x) volume)"
        )
    return w, tuple(int(s) // 2 for s in w.shape)


def convolve(array, weights, boundary="reflect", cval=0.0, device=None):
    """Multidimensional convolution with a ``weights`` kernel (like scipy.ndimage.convolve).
    ``weights`` must have the same ndim as ``array`` (size-1 axes leave an axis untouched).
    The kernel is moved to the block's namespace, so this runs on CPU or GPU per ``device``."""
    w, depth = _kernel_depth(weights, array.ndim)
    func = lambda b: ndimage_namespace(b).convolve(
        b, array_namespace(b).asarray(w), mode=boundary, cval=cval)
    return _named(array, func, depth, boundary, "convolve", device,
                  {"name": "convolve", "weights": w, "mode": boundary, "cval": cval})


def correlate(array, weights, boundary="reflect", cval=0.0, device=None):
    """Multidimensional cross-correlation with a ``weights`` kernel (like
    scipy.ndimage.correlate). Same as ``convolve`` but the kernel is not flipped."""
    w, depth = _kernel_depth(weights, array.ndim)
    func = lambda b: ndimage_namespace(b).correlate(
        b, array_namespace(b).asarray(w), mode=boundary, cval=cval)
    return _named(array, func, depth, boundary, "correlate", device,
                  {"name": "correlate", "weights": w, "mode": boundary, "cval": cval})


def laplace(array, boundary="reflect", device=None, **kw):
    """Laplace filter via the second-derivative [1, -2, 1] stencil (scipy.ndimage.laplace).
    Fixed radius 1 on every axis."""
    func = lambda b: ndimage_namespace(b).laplace(b, mode=boundary, **kw)
    kernel = None
    if set(kw) <= _CVAL_KW:
        kernel = {"name": "laplace", "mode": boundary, "cval": kw.get("cval", 0.0)}
    return _named(array, func, (1,) * array.ndim, boundary, "laplace", device, kernel)


def gaussian_laplace(array, sigma, boundary="reflect", truncate=4.0, device=None, **kw):
    """Laplace of Gaussian (scipy.ndimage.gaussian_laplace). Halo is the Gaussian radius
    ``int(truncate * sigma + 0.5)`` per axis, as for ``gaussian_filter``."""
    depth = _gaussian_depth(sigma, array.ndim, truncate)
    func = lambda b: ndimage_namespace(b).gaussian_laplace(
        b, sigma=sigma, mode=boundary, truncate=truncate, **kw)
    return _named(array, func, depth, boundary, "gaussian_laplace", device,
                  _gaussian_kernel("gaussian_laplace", sigma, boundary, truncate, kw))


def gaussian_gradient_magnitude(array, sigma, boundary="reflect", truncate=4.0,
                                device=None, **kw):
    """Gradient magnitude using Gaussian derivatives (scipy.ndimage.
    gaussian_gradient_magnitude). Halo is the Gaussian radius per axis."""
    depth = _gaussian_depth(sigma, array.ndim, truncate)
    func = lambda b: ndimage_namespace(b).gaussian_gradient_magnitude(
        b, sigma=sigma, mode=boundary, truncate=truncate, **kw)
    return _named(array, func, depth, boundary, "gaussian_gradient_magnitude", device,
                  _gaussian_kernel("gaussian_gradient_magnitude", sigma, boundary, truncate,
                                   kw))


def _gaussian_kernel(name, sigma, boundary, truncate, kw):
    """The kernel spec of gaussian_laplace / gaussian_gradient_magnitude (None: not for
    the engine)."""
    if not set(kw) <= _CVAL_KW:
        return None
    return {"name": name, "sigma": sigma, "mode": boundary, "cval": kw.get("cval", 0.0),
            "truncate": truncate}


__all__ = [
    "MapOverlapTransform", "map_overlap",
    "gaussian_filter", "uniform_filter", "median_filter", "rank_filter", "percentile_filter",
    "minimum_filter", "maximum_filter", "grey_erosion", "grey_dilation",
    "convolve", "correlate",
    "laplace", "gaussian_laplace", "gaussian_gradient_magnitude",
]
