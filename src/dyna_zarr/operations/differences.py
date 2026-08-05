"""Discrete differences: diff and gradient (halo-1 neighbourhood ops along an axis)."""

from ._backend import array_namespace
from .pointwise import subtract
from .neighborhood import map_overlap


def diff(array, n=1, axis=-1):
    """Discrete difference along ``axis`` (``out[i] = a[i+1] - a[i]``), ``n`` times.

    Composed from slicing + subtract, so it stays lazy and memory-bound: each read pulls
    two adjacent strips. The axis shrinks by ``n`` (reductive, like numpy.diff).
    """
    ax = axis if axis >= 0 else array.ndim + axis
    for _ in range(int(n)):
        nd = array.ndim
        m = array.shape[ax]                       # explicit positive bounds: SliceTransform
        front = array[tuple(slice(1, m) if i == ax else slice(None) for i in range(nd))]
        back = array[tuple(slice(0, m - 1) if i == ax else slice(None) for i in range(nd))]
        array = subtract(front, back)
    return array


def gradient(array, axis=-1, device=None):
    """Central-difference gradient along a single ``axis`` (shape-preserving, like
    numpy.gradient with edge_order=1).

    A depth-1 map_overlap: interior pixels get true central differences (the halo supplies
    neighbours across region seams); the true array edges use ``odd_reflect`` padding
    (linear extrapolation ``2*edge - inner``), which makes the central difference there
    equal numpy's one-sided edge difference -- so it matches numpy exactly, everywhere.
    """
    ax = axis if axis >= 0 else array.ndim + axis
    depth = tuple(1 if i == ax else 0 for i in range(array.ndim))
    func = lambda b: array_namespace(b).gradient(b, axis=ax)
    return map_overlap(array, func, depth, boundary="odd_reflect", name="gradient", device=device)


__all__ = ["diff", "gradient"]
