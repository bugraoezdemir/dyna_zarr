"""Prefix-scan ops along a single axis: cumsum / cumprod / cummax / cummin.

A scan is neither pointwise nor a reduction: the output at index ``p`` along the scan axis
depends on ALL input up to ``p``. Two execution paths, with different memory behavior:

- ``io.write`` routes to ``scan_write`` (below), which streams with a BOUNDED CARRY: tile the
  cross-section, walk the scan axis in strips keeping a running accumulator. Peak memory is
  ~ one strip x one cross-section tile, independent of the scan axis and the array size --
  FULLY memory-bounded, read-once/write-once. This is the path that matters.
- The lazy ``read(key)`` (for ``compute`` / sub-slices) can't carry state between independent
  reads, so it reads the input PREFIX ``[0, stop)`` along the scan axis, accumulates, and
  slices back. That is correct and chunk-invariant, but bounded only by the prefix (~ the
  scan axis), so prefer ``io.write`` for large scans.
"""

from __future__ import annotations

from itertools import product as _iproduct

import numpy as np

from ._base import Transform
from ._backend import array_namespace, asnumpy, resolve_device, to_device
from .structural import _norm_key

_DEFAULT_SCAN_MEM = 256 * 1024 * 1024      # per-call streaming budget (bytes)


def _cumsum(xp, b, ax):
    return xp.cumsum(b, axis=ax)


def _cumprod(xp, b, ax):
    return xp.cumprod(b, axis=ax)


def _accumulate_via(ufunc_name):
    """cummax/cummin via ``ufunc.accumulate``; falls back to host numpy if the array
    namespace (e.g. some cupy versions) does not implement ufunc.accumulate."""
    def acc(xp, b, ax):
        uf = getattr(xp, ufunc_name)
        try:
            return uf.accumulate(b, axis=ax)
        except (AttributeError, TypeError, NotImplementedError):
            host = np.asarray(asnumpy(b))
            return xp.asarray(getattr(np, ufunc_name).accumulate(host, axis=ax))
    return acc


_ACC = {
    "cumsum": _cumsum,
    "cumprod": _cumprod,
    "cummax": _accumulate_via("maximum"),
    "cummin": _accumulate_via("minimum"),
}


class ScanTransform(Transform):
    """Lazy prefix scan along one axis. See the module docstring."""

    def __init__(self, array, op, axis, device=None, name=None):
        super().__init__()
        self.array = array
        self.op = op
        self.acc = _ACC[op]
        self.axis = int(axis) if axis >= 0 else array.ndim + int(axis)
        if not (0 <= self.axis < array.ndim):
            raise ValueError(f"scan axis {axis} out of range for ndim {array.ndim}")
        self.device = device
        self.name = name or op
        self.shape = tuple(array.shape)
        # dtype as numpy's accumulate would produce for this input dtype (e.g. cumsum keeps
        # the integer dtype, matching numpy/dask -- users can astype to widen if needed).
        self.dtype = np.asarray(self.acc(np, np.zeros((1,), dtype=array.dtype), 0)).dtype
        self.chunks = None

    def read(self, key):
        norm = _norm_key(key, self.shape)         # per axis (is_int, start, stop, step)
        a = self.axis
        ndim = len(self.shape)

        # Read the full prefix on the scan axis; keep every axis (size-1 for int indices) so
        # the scan axis stays at position ``a`` with no dim-shift bookkeeping.
        read_key = []
        for d in range(ndim):
            is_int, start, stop, step = norm[d]
            if d == a:
                read_key.append(slice(0, stop))               # prefix [0, stop), step 1
            elif is_int:
                read_key.append(slice(start, start + 1))      # keepdim; squeezed at the end
            else:
                read_key.append(slice(start, stop, step))

        dev = resolve_device(self.device)
        block = to_device(self.array._read_direct(tuple(read_key)), dev)
        xp = array_namespace(block)
        acc = self.acc(xp, block, a)

        # Select the requested output range on the scan axis (within the prefix), then drop
        # every axis that was an integer index in the original key.
        is_int_a, start_a, stop_a, step_a = norm[a]
        sel = [slice(None)] * ndim
        sel[a] = slice(start_a, start_a + 1) if is_int_a else slice(start_a, stop_a, step_a)
        acc = acc[tuple(sel)]
        drop = tuple(d for d in range(ndim) if norm[d][0])
        return xp.squeeze(acc, axis=drop) if drop else acc


# op -> (local scan, cross-strip combine with the running carry). carry is size-1 on the scan
# axis and broadcasts over the strip; this makes the scan a bounded-carry stream.
_COMBINE = {
    "cumsum": lambda xp, carry, local: carry + local,
    "cumprod": lambda xp, carry, local: carry * local,
    "cummax": lambda xp, carry, local: xp.maximum(carry, local),
    "cummin": lambda xp, carry, local: xp.minimum(carry, local),
}


def scan_write(source, op, axis, output_path, output_chunks=None,
               max_mem=_DEFAULT_SCAN_MEM, dtype=None, zarr_format=2):
    """Stream a prefix scan of ``source`` along ``axis`` to ``output_path``, FULLY
    memory-bounded. A scan has a bounded carry (a single slab perpendicular to the scan
    axis), so we tile the cross-section and, within each tile, walk the scan axis in strips
    keeping a running carry: ``out_strip = combine(carry, local_scan(strip))``, then
    ``carry = last slab of out_strip``. Peak memory is ~ one strip x one cross-section tile,
    independent of the scan axis and of the array size. Read-once / write-once.
    """
    import zarr
    from ..utils import parse_dtype

    shape = tuple(int(s) for s in source.shape)
    ndim = len(shape)
    a = int(axis) if axis >= 0 else ndim + int(axis)
    accfun = _ACC[op]
    combine = _COMBINE[op]
    dt = parse_dtype(dtype if dtype is not None else source.dtype)[0]
    budget = max(1, int(max_mem) // dt.itemsize)

    oc = tuple(output_chunks) if output_chunks is not None else (
        tuple(source.chunks) if source.chunks else tuple(min(s, 256) for s in shape))
    out = zarr.open(str(output_path), mode="w", shape=shape, chunks=oc,
                    dtype=dt, zarr_format=zarr_format)

    # cross-section tile (all axes but the scan axis) sized to <= budget/2 so the carry fits
    # and a strip of >= 2 slabs also fits; trailing axes grow first for contiguity.
    cross_budget = max(1, budget // 2)
    tile = [1] * ndim
    acc = 1
    for d in reversed([d for d in range(ndim) if d != a]):
        tile[d] = min(shape[d], max(1, cross_budget // acc))
        acc *= tile[d]
        if tile[d] < shape[d]:
            break
    cross_size = 1
    for d in range(ndim):
        if d != a:
            cross_size *= tile[d]
    strip_a = max(1, budget // max(1, cross_size))     # slabs of the scan axis per read

    cross_ranges = [range(0, shape[d], tile[d]) for d in range(ndim) if d != a]
    other_axes = [d for d in range(ndim) if d != a]
    for cross_origin in _iproduct(*cross_ranges):
        base = [None] * ndim
        for d, o in zip(other_axes, cross_origin):
            base[d] = slice(o, min(o + tile[d], shape[d]))
        carry = None
        for a0 in range(0, shape[a], strip_a):
            sl = list(base)
            sl[a] = slice(a0, min(a0 + strip_a, shape[a]))
            sl = tuple(sl)
            block = np.asarray(asnumpy(source._read_direct(sl)))
            local = accfun(np, block, a)
            out_block = local if carry is None else combine(np, carry, local)
            out[sl] = out_block
            last = [slice(None)] * ndim
            last[a] = slice(-1, None)                  # keepdims last slab -> next carry
            carry = out_block[tuple(last)]
    return output_path


def _scan(array, op, axis, device=None):
    return array._with_transform(ScanTransform(array, op, axis, device=device))


def cumsum(array, axis, device=None):
    """Cumulative sum along ``axis`` (shape-preserving)."""
    return _scan(array, "cumsum", axis, device=device)


def cumprod(array, axis, device=None):
    """Cumulative product along ``axis`` (shape-preserving)."""
    return _scan(array, "cumprod", axis, device=device)


def cummax(array, axis, device=None):
    """Cumulative maximum along ``axis`` (shape-preserving)."""
    return _scan(array, "cummax", axis, device=device)


def cummin(array, axis, device=None):
    """Cumulative minimum along ``axis`` (shape-preserving)."""
    return _scan(array, "cummin", axis, device=device)


__all__ = ["ScanTransform", "cumsum", "cumprod", "cummax", "cummin"]
