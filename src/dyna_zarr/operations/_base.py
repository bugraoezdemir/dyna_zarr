"""Transform base class and shared slice-math helpers for the operations package."""

import numpy as np
from typing import Tuple, Union, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dyna_zarr.dynamic_array import DynamicArray


def _is_int_index(k) -> bool:
    """True if key element is an integer index (drops its axis), not a slice/newaxis."""
    return isinstance(k, (int, np.integer))


def _perm_on_surviving(result: np.ndarray, out_key, axes) -> np.ndarray:
    """Reorder ``result`` into output-axis order for a transpose/swapaxes read.

    ``result`` was read from the underlying array with axes in *input* order, with
    integer-indexed axes already dropped by the read. ``out_key`` is the output-space
    key (len == len(axes)); output position ``i`` maps to input axis ``axes[i]`` and
    ``out_key[i]`` applies to it. We transpose ``result`` into output order, restricted
    to the axes that survive integer indexing. Reduces to ``np.transpose(result, axes)``
    when the key contains no integer indices.
    """
    ndim = len(axes)
    dropped = {axes[i] for i in range(ndim) if _is_int_index(out_key[i])}
    surviving = [ia for ia in range(ndim) if ia not in dropped]
    pos = {ia: j for j, ia in enumerate(surviving)}
    desired = [axes[i] for i in range(ndim) if not _is_int_index(out_key[i])]
    return np.transpose(result, [pos[ia] for ia in desired])


#: Attributes a Transform uses to hold its upstream DynamicArray(s): ``array`` for the
#: unary ops, ``arrays`` for concatenate/stack, ``operands`` for map_blocks (which may
#: also hold plain scalars). Walking these reaches the whole chain.
UPSTREAM_ATTRS = ("array", "arrays", "operands")


def iter_chain(array, _max_depth=256):
    """Every DynamicArray in ``array``'s lazy chain, UPSTREAM FIRST (post-order).

    Upstream-first matters to callers that evaluate things along the chain: a node's
    inputs are always yielded before the node. Each array is yielded once, even when
    the chain is a DAG that reaches it along several paths (``a - a.mean()``).
    """
    seen = set()
    out = []
    stack = [(array, False, 0)]
    while stack:
        node, expanded, depth = stack.pop()
        if node is None or not hasattr(node, "_transform"):
            continue
        if expanded:
            out.append(node)
            continue
        if id(node) in seen or depth > _max_depth:
            continue
        seen.add(id(node))
        stack.append((node, True, depth))
        transform = node._transform
        if transform is None:
            continue
        for attr in UPSTREAM_ATTRS:
            upstream = getattr(transform, attr, None)
            if upstream is None:
                continue
            items = upstream if isinstance(upstream, (list, tuple)) else [upstream]
            for item in reversed(items):
                if hasattr(item, "_transform"):
                    stack.append((item, False, depth + 1))
    return out


class Transform:
    """
    Base class for lazy transformations.
    """

    def __init__(self):
        self.shape = None
        self.chunks = None
        self.dtype = None

    def read(self, key):
        raise NotImplementedError


