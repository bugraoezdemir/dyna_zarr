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


def _as_operands(operands):
    """Operands as map_blocks holds them: a non-0-d NumPy array becomes a DynamicArray.

    A raw ndarray used to be passed to ``func`` whole, next to operand BLOCKS, so any
    read smaller than the full array paired a region with the entire ndarray and either
    raised or broadcast it against the wrong elements. Wrapping it gives it the same
    blockwise read as every other array operand. Scalars and 0-d arrays pass through.
    """
    from dyna_zarr.dynamic_array import DynamicArray
    out = []
    for o in operands:
        if isinstance(o, np.ndarray) and o.ndim > 0:
            o = DynamicArray(o)
        out.append(o)
    return out


def _operand_key(key, out_shape, op_shape):
    """Map an output read ``key`` onto an operand that BROADCASTS to ``out_shape``.

    NumPy broadcasting aligns shapes from the right. An operand axis of the output's
    size takes the key element unchanged; a size-1 (broadcast) axis reads its single
    element - ``0`` where the key drops that axis, ``slice(0, 1)`` where it keeps it (or
    ``slice(0, 0)`` for an empty selection); output axes the operand lacks are skipped.
    Integer keys drop the same axes from every block, so the blocks stay right-aligned
    and ``func`` broadcasts them exactly as it would the full arrays.
    """
    offset = len(out_shape) - len(op_shape)
    mapped = []
    for j, osz in enumerate(op_shape):
        k = key[offset + j]
        if osz == out_shape[offset + j]:
            mapped.append(k)
        elif isinstance(k, slice):
            empty = len(range(*k.indices(out_shape[offset + j]))) == 0
            mapped.append(slice(0, 0) if empty else slice(0, 1))
        else:
            mapped.append(0)
    return tuple(mapped)


class MapBlocksTransform(Transform):
    """Pull-model ``map_blocks``: apply an ELEMENTWISE (chunk-independent) function to
    one or more DynamicArray operands (plus scalars) that broadcast together.

    Because the function is pointwise, ``func(a[key], b[key]) == func(a, b)[key]``, so
    reading an arbitrary output slice just reads that slice from each operand and applies
    the function -- nothing is materialised, so it stays memory-bound. This is the
    ``conservative . pointwise`` taxonomy cell; it is NOT valid for neighbourhood or
    reducing functions (use map_overlap / reduce for those).

    Operand shapes follow NumPy broadcasting, so an nd array combines with a 0-d result
    (``x > x.mean()``) or a ``keepdims`` one (``x - x.mean(axis=0, keepdims=True)``). A
    broadcast operand is read at the output key collapsed onto its size-1 axes (see
    :func:`_operand_key`). Scalars pass through and NumPy broadcasts them.
    """

    def __init__(self, func, operands, dtype=None, name=None, device=None):
        super().__init__()
        from dyna_zarr.dynamic_array import DynamicArray
        self.func = func
        self.operands = _as_operands(operands)
        self.device = device
        self.name = name or getattr(func, "__name__", "map_blocks")
        arrays = [o for o in self.operands if isinstance(o, DynamicArray)]
        if not arrays:
            raise ValueError("map_blocks requires at least one DynamicArray operand")
        try:
            self.shape = tuple(np.broadcast_shapes(*(a.shape for a in arrays)))
        except ValueError:
            raise ValueError(
                "map_blocks operands must broadcast together; got shapes "
                + ", ".join(str(tuple(a.shape)) for a in arrays)
            ) from None
        # The chunk grid comes from an operand that spans the full output; a broadcast
        # operand's grid describes a different (smaller) array.
        full = [a for a in arrays if tuple(a.shape) == self.shape]
        self.chunks = full[0].chunks if full else None
        self.dtype = dtype if dtype is not None else _infer_mapblocks_dtype(func, self.operands)

    def broadcast_operands(self):
        """``(operand, broadcast_axes)`` for every DynamicArray operand that does NOT
        span the output: ``broadcast_axes`` are the output axes along which it is
        repeated (size 1 there, or absent). Used by io.write to plan around them."""
        from dyna_zarr.dynamic_array import DynamicArray
        out = []
        for o in self.operands:
            if not isinstance(o, DynamicArray) or tuple(o.shape) == self.shape:
                continue
            offset = len(self.shape) - len(o.shape)
            axes = tuple(
                i for i, s in enumerate(self.shape)
                if s > 1 and (i < offset or o.shape[i - offset] == 1)
            )
            if axes:
                out.append((o, axes))
        return out

    def read(self, key):
        from dyna_zarr.dynamic_array import DynamicArray
        dev = resolve_device(self.device)
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        blocks = [
            to_device(o._read_direct(
                key if tuple(o.shape) == self.shape
                else _operand_key(key, self.shape, o.shape)), dev)
            if isinstance(o, DynamicArray)
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
    """Apply an elementwise (chunk-independent) ``func`` to one or more DynamicArrays
    (plus scalars) whose shapes broadcast together, lazily and memory-bound. Every
    pointwise op here (ufuncs, comparisons, logical ops, where, clip, astype) is a thin
    wrapper over this. ``func`` receives the read block of each operand; for
    neighbourhood/reducing funcs use map_overlap / reduce. ``device`` (None=inherit the
    execution context, 'cpu', 'cuda') runs this op on that device."""
    from dyna_zarr.dynamic_array import DynamicArray
    transform = MapBlocksTransform(func, operands, dtype=dtype, name=name, device=device)
    arrays = [o for o in transform.operands if isinstance(o, DynamicArray)]
    # Carry the metadata of an operand that spans the output, not of a broadcast one.
    ref = next((a for a in arrays if tuple(a.shape) == transform.shape), arrays[0])
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
def exp2(array): return map_blocks(np.exp2, array, name="exp2")
def expm1(array): return map_blocks(np.expm1, array, name="expm1")
def log1p(array): return map_blocks(np.log1p, array, name="log1p")
def cbrt(array): return map_blocks(np.cbrt, array, name="cbrt")
def fabs(array): return map_blocks(np.fabs, array, name="fabs")
def positive(array): return map_blocks(np.positive, array, name="positive")
def conjugate(array): return map_blocks(np.conjugate, array, name="conjugate")
def rint(array): return map_blocks(np.rint, array, name="rint")
def trunc(array): return map_blocks(np.trunc, array, name="trunc")
def spacing(array): return map_blocks(np.spacing, array, name="spacing")

# trigonometric / hyperbolic / angle conversion (all pointwise ufuncs)
def sin(array): return map_blocks(np.sin, array, name="sin")
def cos(array): return map_blocks(np.cos, array, name="cos")
def tan(array): return map_blocks(np.tan, array, name="tan")
def arcsin(array): return map_blocks(np.arcsin, array, name="arcsin")
def arccos(array): return map_blocks(np.arccos, array, name="arccos")
def arctan(array): return map_blocks(np.arctan, array, name="arctan")
def sinh(array): return map_blocks(np.sinh, array, name="sinh")
def cosh(array): return map_blocks(np.cosh, array, name="cosh")
def tanh(array): return map_blocks(np.tanh, array, name="tanh")
def arcsinh(array): return map_blocks(np.arcsinh, array, name="arcsinh")
def arccosh(array): return map_blocks(np.arccosh, array, name="arccosh")
def arctanh(array): return map_blocks(np.arctanh, array, name="arctanh")
def deg2rad(array): return map_blocks(np.deg2rad, array, name="deg2rad")
def rad2deg(array): return map_blocks(np.rad2deg, array, name="rad2deg")
def degrees(array): return map_blocks(np.degrees, array, name="degrees")
def radians(array): return map_blocks(np.radians, array, name="radians")

# predicates (bool out)
def signbit(array): return map_blocks(np.signbit, array, name="signbit")
def isfinite(array): return map_blocks(np.isfinite, array, name="isfinite")
def isinf(array): return map_blocks(np.isinf, array, name="isinf")
def isnan(array): return map_blocks(np.isnan, array, name="isnan")


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
def remainder(a, b): return map_blocks(np.remainder, a, b, name="remainder")
def fmod(a, b): return map_blocks(np.fmod, a, b, name="fmod")
def fmax(a, b): return map_blocks(np.fmax, a, b, name="fmax")
def fmin(a, b): return map_blocks(np.fmin, a, b, name="fmin")
def float_power(a, b): return map_blocks(np.float_power, a, b, name="float_power")
def hypot(a, b): return map_blocks(np.hypot, a, b, name="hypot")
def arctan2(a, b): return map_blocks(np.arctan2, a, b, name="arctan2")
def copysign(a, b): return map_blocks(np.copysign, a, b, name="copysign")
def nextafter(a, b): return map_blocks(np.nextafter, a, b, name="nextafter")
def logaddexp(a, b): return map_blocks(np.logaddexp, a, b, name="logaddexp")
def logaddexp2(a, b): return map_blocks(np.logaddexp2, a, b, name="logaddexp2")
def heaviside(a, b): return map_blocks(np.heaviside, a, b, name="heaviside")


# bitwise / shift (integer inputs; on bool these coincide with the logical ops, so
# they also back the &/|/^/~ dunders -- matching numpy/dask exactly)
def bitwise_and(a, b): return map_blocks(np.bitwise_and, a, b, name="bitwise_and")
def bitwise_or(a, b): return map_blocks(np.bitwise_or, a, b, name="bitwise_or")
def bitwise_xor(a, b): return map_blocks(np.bitwise_xor, a, b, name="bitwise_xor")
def invert(array): return map_blocks(np.invert, array, name="invert")
def left_shift(a, b): return map_blocks(np.left_shift, a, b, name="left_shift")
def right_shift(a, b): return map_blocks(np.right_shift, a, b, name="right_shift")
def gcd(a, b): return map_blocks(np.gcd, a, b, name="gcd")
def lcm(a, b): return map_blocks(np.lcm, a, b, name="lcm")
def ldexp(a, b): return map_blocks(np.ldexp, a, b, name="ldexp")


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
    "exp2", "expm1", "log1p", "cbrt", "fabs", "positive", "conjugate",
    "rint", "trunc", "spacing",
    "sin", "cos", "tan", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh",
    "deg2rad", "rad2deg", "degrees", "radians",
    "signbit", "isfinite", "isinf", "isnan",
    "add", "subtract", "multiply", "divide", "floor_divide", "mod", "power",
    "maximum", "minimum",
    "remainder", "fmod", "fmax", "fmin", "float_power", "hypot", "arctan2",
    "copysign", "nextafter", "logaddexp", "logaddexp2", "heaviside",
    "bitwise_and", "bitwise_or", "bitwise_xor", "invert", "left_shift", "right_shift",
    "gcd", "lcm", "ldexp",
    "greater", "greater_equal", "less", "less_equal", "equal", "not_equal",
    "logical_and", "logical_or", "logical_xor", "logical_not", "where",
    "isin", "digitize",
]
