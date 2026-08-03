"""Neighbourhood primitive (pull-model ``map_overlap``) and the filters built on it.

A neighbourhood op is shape-preserving but each output element depends on a local window
of the input (halo/``depth``). In the pull model this is just map_blocks with a widened
read window: to produce the block for ``key`` we read ``key`` expanded by ``depth`` on
every side (padding at the true array edges per ``boundary``), apply the (shape-preserving)
function to that expanded block, then crop the halo back off. Because every read pulls its
own halo, the result is independent of the region size that drove it -- i.e. chunk/region
invariant, and exact vs applying the function to the whole array (given ``depth`` >= the
function's radius and a matching ``boundary``).

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
                 device=None):
        super().__init__()
        if boundary not in _BOUNDARY_TO_NPPAD:
            raise ValueError(f"unknown boundary {boundary!r}; expected one of "
                             f"{sorted(_BOUNDARY_TO_NPPAD)}")
        self.array = array
        self.func = func
        self.device = device
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

        read_slices = []      # region to read from the input (clamped to bounds, step 1)
        pad_widths = []       # (before, after) padding that restores the halo at true edges
        crop_slices = []      # crop the func output back to the core region + apply step
        squeeze_axes = []
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

            # desired expanded span [start-d, stop+d), clamped to [0, size)
            read_slices.append(slice(max(0, start - d), min(size, stop + d)))
            pad_widths.append((max(0, d - start), max(0, (stop + d) - size)))
            # after padding, the block has the core at [d : d+core_len]; then apply the step
            crop_slices.append(slice(d, d + (stop - start), step))

        block = to_device(self.array._read_direct(tuple(read_slices)),
                          resolve_device(self.device))
        xp = array_namespace(block)
        if any(pb or pa for pb, pa in pad_widths):
            mode, pad_kw = _BOUNDARY_TO_NPPAD[self.boundary]
            block = xp.pad(block, pad_widths, mode=mode, **pad_kw)

        out = self.func(block)                 # func dispatches ndimage on block's device
        out = out[tuple(crop_slices)]
        for a in sorted(squeeze_axes, reverse=True):
            out = xp.squeeze(out, axis=a)
        return out


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

def map_overlap(array, func, depth, boundary="reflect", dtype=None, name=None, device=None):
    """Apply a shape-preserving neighbourhood ``func`` with a ``depth`` halo, lazily and
    chunk-invariantly. Every read pulls its own halo, so the result is independent of the
    region size and exact vs applying ``func`` to the whole array (given depth >= radius and
    matching ``boundary``). The filters below wrap this. ``device`` (None=inherit, 'cpu',
    'cuda') runs it on that device."""
    return array._with_transform(
        MapOverlapTransform(array, func, depth, boundary=boundary, dtype=dtype,
                            name=name, device=device)
    )


def gaussian_filter(array, sigma, boundary="reflect", truncate=4.0, device=None, **kw):
    depth = _gaussian_depth(sigma, array.ndim, truncate)
    func = lambda b: ndimage_namespace(b).gaussian_filter(
        b, sigma=sigma, mode=boundary, truncate=truncate, **kw)
    return map_overlap(array, func, depth, boundary=boundary, name="gaussian_filter", device=device)


def uniform_filter(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).uniform_filter(b, size=size, mode=boundary, **kw)
    return map_overlap(array, func, _size_depth(size, array.ndim), boundary=boundary,
                       name="uniform_filter", device=device)


def median_filter(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).median_filter(b, size=size, mode=boundary, **kw)
    return map_overlap(array, func, _size_depth(size, array.ndim), boundary=boundary,
                       name="median_filter", device=device)


def minimum_filter(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).minimum_filter(b, size=size, mode=boundary, **kw)
    return map_overlap(array, func, _size_depth(size, array.ndim), boundary=boundary,
                       name="minimum_filter", device=device)


def maximum_filter(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).maximum_filter(b, size=size, mode=boundary, **kw)
    return map_overlap(array, func, _size_depth(size, array.ndim), boundary=boundary,
                       name="maximum_filter", device=device)


def grey_erosion(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).grey_erosion(b, size=size, mode=boundary, **kw)
    return map_overlap(array, func, _size_depth(size, array.ndim), boundary=boundary,
                       name="grey_erosion", device=device)


def grey_dilation(array, size, boundary="reflect", device=None, **kw):
    func = lambda b: ndimage_namespace(b).grey_dilation(b, size=size, mode=boundary, **kw)
    return map_overlap(array, func, _size_depth(size, array.ndim), boundary=boundary,
                       name="grey_dilation", device=device)


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
    return map_overlap(array, func, depth, boundary=boundary, name="convolve", device=device)


def correlate(array, weights, boundary="reflect", cval=0.0, device=None):
    """Multidimensional cross-correlation with a ``weights`` kernel (like
    scipy.ndimage.correlate). Same as ``convolve`` but the kernel is not flipped."""
    w, depth = _kernel_depth(weights, array.ndim)
    func = lambda b: ndimage_namespace(b).correlate(
        b, array_namespace(b).asarray(w), mode=boundary, cval=cval)
    return map_overlap(array, func, depth, boundary=boundary, name="correlate", device=device)


__all__ = [
    "MapOverlapTransform", "map_overlap",
    "gaussian_filter", "uniform_filter", "median_filter",
    "minimum_filter", "maximum_filter", "grey_erosion", "grey_dilation",
    "convolve", "correlate",
]
