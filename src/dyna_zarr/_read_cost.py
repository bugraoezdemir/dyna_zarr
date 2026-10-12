"""What reading part of an array really costs: its source's read unit.

An array's ``chunks`` say how it is laid out for WRITING decisions; they do not say what
a read of a piece of it costs. micro-reader images carry a ``read_unit``: the smallest
block the backend decodes at once (a compressed TIFF strip or tile, an IMS chunk, an ND2
frame), or ``None`` when any rectangle is read exactly (uncompressed data). A planner that
reads parts of planes (y/x tiling) needs that, not the advertised chunk.

``read_unit(array)`` follows an op chain back to its sources and maps their units through
the transforms that only move or relabel axes. It returns a per-axis tuple, where all 1s
means "any rectangle is read exactly", or ``None`` when it is unknown.
"""
from __future__ import annotations

from typing import Optional, Tuple


def _merge(units):
    """Several inputs read side by side: the coarsest unit per axis governs."""
    units = list(units)
    if not units or any(u is None for u in units):
        return None
    if len({len(u) for u in units}) != 1:
        return None
    return tuple(max(vals) for vals in zip(*units))


def _source_unit(array) -> Optional[Tuple[int, ...]]:
    """An untransformed array: its source's read unit."""
    from .dynamic_array import _SourceAdapter
    ndim = len(array.shape)
    src = getattr(array, "_zarr_array", None)
    if isinstance(src, _SourceAdapter):
        inner = src._source
        if hasattr(inner, "read_unit"):
            unit = inner.read_unit
            if unit is None:                       # micro-reader: exact rectangles
                return (1,) * ndim
            unit = tuple(int(u) for u in unit)
            return unit if len(unit) == ndim else None
        return None
    # zarr / TensorStore: a read decodes whole chunks (inner chunks when sharded)
    chunks = getattr(array, "chunks", None)
    if chunks is not None and len(chunks) == ndim:
        return tuple(int(c) for c in chunks)
    return None


def read_unit(array) -> Optional[Tuple[int, ...]]:
    """Per-axis read unit of ``array`` (all 1s: exact rectangles), or None if unknown."""
    t = getattr(array, "_transform", None)
    if t is None:
        return _source_unit(array)
    name = type(t).__name__
    if name in ("ConcatenateTransform",):
        return _merge(read_unit(a) for a in t.arrays)
    if name == "StackTransform":
        merged = _merge(read_unit(a) for a in t.arrays)
        if merged is None:
            return None
        return merged[:t.axis] + (1,) + merged[t.axis:]
    if name == "ExpandDimsTransform":
        u = read_unit(t.array)
        return None if u is None else u[:t.axis] + (1,) + u[t.axis:]
    if name == "SqueezeTransform":
        u = read_unit(t.array)
        if u is None:
            return None
        drop = {int(a) % len(u) for a in t.squeeze_axes}
        return tuple(v for i, v in enumerate(u) if i not in drop)
    if name == "TransposeTransform":
        u = read_unit(t.array)
        return None if u is None else tuple(u[a] for a in t.axes)
    if name == "SwapAxesTransform":
        u = read_unit(t.array)
        if u is None:
            return None
        u = list(u)
        u[t.axis1], u[t.axis2] = u[t.axis2], u[t.axis1]
        return tuple(u)
    if name == "SliceTransform":
        return _slice_unit(read_unit(t.array), t.key, len(t.array.shape))
    if name == "MapBlocksTransform":
        # pointwise: same shape in, same shape out (broadcasting operands: unknown)
        arrays = [o for o in t.operands if hasattr(o, "shape") and hasattr(o, "_transform")]
        if not arrays or any(tuple(a.shape) != tuple(array.shape) for a in arrays):
            return None
        return _merge(read_unit(a) for a in arrays)
    return None                       # reshape, pad, neighbourhood ops, ...: unknown


def _slice_unit(unit, key, ndim):
    """A basic index: integers drop an axis, None adds one, slices keep theirs."""
    if unit is None:
        return None
    if not isinstance(key, tuple):
        key = (key,)
    out, axis = [], 0
    for k in key:
        if k is None:
            out.append(1)
        elif k is Ellipsis:
            n = ndim - sum(1 for x in key if x is not None and x is not Ellipsis)
            out.extend(unit[axis:axis + n])
            axis += n
        elif isinstance(k, slice):
            out.append(unit[axis])
            axis += 1
        else:
            try:
                int(k)                       # an integer drops the axis
            except (TypeError, ValueError):
                return None                  # fancy indexing: unknown
            axis += 1
    out.extend(unit[axis:])
    return tuple(out)


#: what a read holds at its peak, per byte read (measured on a sharded 96^3 store,
#: 2026-10-08): zarrista and micro-reader decode straight into the result (1.15x);
#: TensorStore fills its own buffer and copies it out (1.96x). Unknown sources count as
#: TensorStore.
_CHEAP_READ, _COPYING_READ = 1.25, 2.0


def _source_overhead(array) -> float:
    from .dynamic_array import _SourceAdapter
    src = getattr(array, "_zarr_array", None)
    if isinstance(src, _SourceAdapter):            # micro-reader and other custom sources
        return _CHEAP_READ
    try:
        from .backends.zarrista_backend import ZarristaArray
        if isinstance(src, ZarristaArray):
            return _CHEAP_READ
    except Exception:                              # noqa: BLE001 - zarrista not installed
        pass
    return _COPYING_READ


def read_overhead(array) -> float:
    """Peak memory of reading a region of ``array``, per byte of the region: the costliest
    of its sources (1.25 when they read straight into the result, 2.0 otherwise)."""
    t = getattr(array, "_transform", None)
    if t is None:
        return _source_overhead(array)
    parts = getattr(t, "arrays", None)
    if parts is None:
        part = getattr(t, "array", None)
        parts = [part] if part is not None else []
    if not parts:
        return _COPYING_READ
    return max(read_overhead(a) for a in parts)
