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


