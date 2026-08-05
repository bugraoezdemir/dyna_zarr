"""Pull-model map_blocks primitive: elementwise (chunk-independent) ops, memory-bound."""

import numpy as np
from typing import Tuple, Union, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from dyna_zarr.dynamic_array import DynamicArray


from ._base import Transform
from ._backend import array_namespace, resolve_device, to_device


def _infer_mapblocks_dtype(func, operands):
    """Infer an elementwise func's output dtype by applying it to 1-element samples of
    each operand's dtype (scalars passed through), letting NumPy type promotion decide.

    Operand dtypes may be tensorstore dtypes (io.read yields TS-backed arrays), which
    numpy can't consume directly, so normalise via parse_dtype first."""
    from dyna_zarr.dynamic_array import DynamicArray
    from ..utils import parse_dtype
    samples = [
        np.ones((1,), dtype=parse_dtype(o.dtype)[0]) if isinstance(o, DynamicArray) else o
        for o in operands
    ]
    return np.asarray(func(*samples)).dtype


class MapBlocksTransform(Transform):
    """Pull-model ``map_blocks``: apply an ELEMENTWISE (chunk-independent) function to
    one or more equally-shaped DynamicArray operands (plus scalars).

    Because the function is pointwise, ``func(a[key], b[key]) == func(a, b)[key]``, so
    reading an arbitrary output slice just reads that slice from each operand and applies
    the function -- nothing is materialised, so it stays memory-bound. This is the
    ``conservative . pointwise`` taxonomy cell; it is NOT valid for neighbourhood or
    reducing functions (use map_overlap / reduce for those).

    Operands may mix DynamicArrays (read blockwise) and plain scalars (passed through;
    NumPy broadcasts them). All DynamicArray operands must share the same shape.
    """

    def __init__(self, func, operands, dtype=None, name=None, device=None):
        super().__init__()
        from dyna_zarr.dynamic_array import DynamicArray
        self.func = func
        self.operands = list(operands)
        self.device = device
        self.name = name or getattr(func, "__name__", "map_blocks")
        arrays = [o for o in self.operands if isinstance(o, DynamicArray)]
        if not arrays:
            raise ValueError("map_blocks requires at least one DynamicArray operand")
        ref = arrays[0]
        for a in arrays[1:]:
            if a.shape != ref.shape:
                raise ValueError(
                    f"map_blocks operands must share shape; got {a.shape} vs {ref.shape}"
                )
        self.shape = ref.shape
        self.chunks = ref.chunks
        self.dtype = dtype if dtype is not None else _infer_mapblocks_dtype(func, self.operands)

    def read(self, key):
        from dyna_zarr.dynamic_array import DynamicArray
        dev = resolve_device(self.device)
        blocks = [
            to_device(o._read_direct(key), dev) if isinstance(o, DynamicArray)
            else o
            for o in self.operands
        ]
        # xp.asarray keeps the result on its own device (numpy or cupy); a plain
        # np.asarray would force a GPU->host copy mid-pipeline.
        xp = array_namespace(*blocks)
        return xp.asarray(self.func(*blocks))


# --------------------------------------------------------------------------- #
# Public ops -- pointwise ufuncs, all thin wrappers over map_blocks
# --------------------------------------------------------------------------- #

def map_blocks(func, *operands, dtype=None, name=None, device=None):
    """Apply an elementwise (chunk-independent) ``func`` to one or more equally-shaped
    DynamicArrays (plus scalars), lazily and memory-bound. Every pointwise op here (ufuncs,
    comparisons, logical ops, where, clip, astype) is a thin wrapper over this. ``func``
    receives the read block of each operand; for neighbourhood/reducing funcs use
    map_overlap / reduce. ``device`` (None=inherit the execution context, 'cpu', 'cuda')
    runs this op on that device."""
    from dyna_zarr.dynamic_array import DynamicArray
    transform = MapBlocksTransform(func, operands, dtype=dtype, name=name, device=device)
    ref = next(o for o in operands if isinstance(o, DynamicArray))
    return ref._with_transform(transform)


# unary
def abs(array): return map_blocks(np.abs, array, name="abs")
def negative(array): return map_blocks(np.negative, array, name="negative")
def sign(array): return map_blocks(np.sign, array, name="sign")
def sqrt(array): return map_blocks(np.sqrt, array, name="sqrt")
def square(array): return map_blocks(np.square, array, name="square")
def exp(array): return map_blocks(np.exp, array, name="exp")
def log(array): return map_blocks(np.log, array, name="log")
def log2(array): return map_blocks(np.log2, array, name="log2")
def log10(array): return map_blocks(np.log10, array, name="log10")
def floor(array): return map_blocks(np.floor, array, name="floor")
def ceil(array): return map_blocks(np.ceil, array, name="ceil")
def reciprocal(array): return map_blocks(np.reciprocal, array, name="reciprocal")


def round(array, decimals=0):
    return map_blocks(lambda x: np.round(x, decimals=decimals), array, name="round")


def clip(array, a_min=None, a_max=None):
    return map_blocks(lambda x: np.clip(x, a_min, a_max), array, name="clip")


def astype(array, dtype):
    return map_blocks(lambda x: x.astype(dtype), array, dtype=np.dtype(dtype), name="astype")


# binary
def add(a, b): return map_blocks(np.add, a, b, name="add")
def subtract(a, b): return map_blocks(np.subtract, a, b, name="subtract")
def multiply(a, b): return map_blocks(np.multiply, a, b, name="multiply")
def divide(a, b): return map_blocks(np.divide, a, b, name="divide")
def floor_divide(a, b): return map_blocks(np.floor_divide, a, b, name="floor_divide")
def mod(a, b): return map_blocks(np.mod, a, b, name="mod")
def power(a, b): return map_blocks(np.power, a, b, name="power")
def maximum(a, b): return map_blocks(np.maximum, a, b, name="maximum")
def minimum(a, b): return map_blocks(np.minimum, a, b, name="minimum")


# comparisons & logical (bool out)
def greater(a, b): return map_blocks(np.greater, a, b, name="greater")
def greater_equal(a, b): return map_blocks(np.greater_equal, a, b, name="greater_equal")
def less(a, b): return map_blocks(np.less, a, b, name="less")
def less_equal(a, b): return map_blocks(np.less_equal, a, b, name="less_equal")
def equal(a, b): return map_blocks(np.equal, a, b, name="equal")
def not_equal(a, b): return map_blocks(np.not_equal, a, b, name="not_equal")
def logical_and(a, b): return map_blocks(np.logical_and, a, b, name="logical_and")
def logical_or(a, b): return map_blocks(np.logical_or, a, b, name="logical_or")
def logical_xor(a, b): return map_blocks(np.logical_xor, a, b, name="logical_xor")
def logical_not(array): return map_blocks(np.logical_not, array, name="logical_not")


# ternary
def where(condition, x, y): return map_blocks(np.where, condition, x, y, name="where")


# lookup / binning (pointwise; func resolves the array module of the block so it runs on
# whichever device the block lives on)
def isin(array, test_elements, invert=False):
    """Elementwise membership test against ``test_elements`` (bool output)."""
    return map_blocks(lambda x: array_namespace(x).isin(x, test_elements, invert=invert),
                      array, dtype=bool, name="isin")


def digitize(array, bins, right=False):
    """Index of the bin each element falls into (like numpy.digitize)."""
    return map_blocks(lambda x: array_namespace(x).digitize(x, bins, right=right),
                      array, dtype=np.intp, name="digitize")


__all__ = [
    "MapBlocksTransform", "map_blocks",
    "abs", "negative", "sign", "sqrt", "square", "exp", "log", "log2", "log10",
    "floor", "ceil", "reciprocal", "round", "clip", "astype",
    "add", "subtract", "multiply", "divide", "floor_divide", "mod", "power",
    "maximum", "minimum",
    "greater", "greater_equal", "less", "less_equal", "equal", "not_equal",
    "logical_and", "logical_or", "logical_xor", "logical_not", "where",
    "isin", "digitize",
]
