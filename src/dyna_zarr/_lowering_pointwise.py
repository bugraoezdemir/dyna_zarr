"""dyna's pointwise operations read by the native engine (Workplan WP15, task DR7 step 7).

A pointwise operation is a MapBlocksTransform whose function is a NumPy ufunc (what
``a + b``, ``a > t``, ``~m`` and dyna's ``operations.add`` / ``greater`` / ... build),
``np.where``, or dyna's ``clip`` / ``round`` (which record their arguments). The engine
reads it when NumPy's result is defined exactly, so the engine gives the Python path's
values bit for bit:

- arithmetic: add, subtract, multiply, divide, negative, positive, absolute, fabs,
  square, sign, sqrt, reciprocal, maximum, minimum, copysign; integers wrap on overflow;
- rounding: floor, ceil, trunc, rint, and round with decimals=0;
- comparisons, logical operations and the tests: greater, greater_equal, less,
  less_equal, equal, not_equal, logical_and / or / xor / not, isnan, isinf, isfinite,
  signbit;
- bit operations: bitwise_and / or / xor, invert, left_shift, right_shift;
- where, and clip with scalar bounds.

The operation runs in the type of NumPy's loop: the result type for arithmetic (which
dyna has inferred with NumPy itself), ``np.result_type`` of the operands for
comparisons (NumPy 1's value-based casting and NumPy 2's weak Python scalars alike,
since both are what ``np.result_type`` does with the same operands). Every array operand
must have the result's shape (the engine does not broadcast) and every scalar must be a
real number the loop type holds exactly (an integer loop) or that NumPy rounds into it
(a float loop).

Staying with the Python path: the transcendental functions (sin, exp, log, power,
cbrt, hypot, arctan2, logaddexp, ...: NumPy's own SIMD code and the platform's libm do
not agree to the bit), fmax / fmin (NumPy's results for signed zeros and NaN pairs
depend on the array's length), floor_divide, mod, remainder, fmod, gcd, lcm, ldexp,
heaviside, nextafter, spacing, isin, digitize, saturate_cast, float16 and complex
numbers, round with other decimals, clip with array bounds, comparisons between int64
and uint64 (exact in NumPy 2, through float64 in NumPy 1), add and multiply with a NaN
scalar (which of two NaNs NumPy keeps then depends on the array's length: the scalar's
in its SIMD code, the second operand's in its short loop), operands that broadcast, and
anything on a GPU device.

Each operation is checked once per process against the NumPy dyna runs with (``_agrees``):
both compute the same special values (signed zeros, NaNs of both signs, infinities, the
integer extremes, shifts past the width), at a length that runs NumPy's SIMD code and its
tail, and must agree bit for bit. A NumPy whose rule differs (clip's for signed zeros
changed between NumPy 1.26 and 2.x) keeps that operation on the Python path.
"""
from __future__ import annotations

import threading

import numpy as np

#: the ufuncs the engine has, by NumPy's name
_UFUNCS = {
    "add", "subtract", "multiply", "divide", "maximum", "minimum", "bitwise_and",
    "bitwise_or", "bitwise_xor", "left_shift", "right_shift", "copysign", "negative",
    "positive", "absolute", "fabs", "square", "sign", "floor", "ceil", "trunc", "rint",
    "sqrt", "reciprocal", "invert", "greater", "greater_equal", "less", "less_equal",
    "equal", "not_equal", "logical_and", "logical_or", "logical_xor", "logical_not",
    "isnan", "isinf", "isfinite", "signbit",
}
#: older NumPy names
_ALIASES = {"true_divide": "divide"}
#: the operations whose result is bool
_GIVES_BOOL = {
    "greater", "greater_equal", "less", "less_equal", "equal", "not_equal", "logical_and",
    "logical_or", "logical_xor", "logical_not", "isnan", "isinf", "isfinite", "signbit",
}
#: loop and result types the engine computes in
_TYPES = {"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
          "float32", "float64"}


def spec(t):
    """``(name, operands)`` of a pointwise MapBlocksTransform the engine may read, or
    None. ``round`` with decimals=0 of an integer array is ``("copy", [array])``."""
    func = t.func
    ops = list(t.operands)
    if isinstance(func, np.ufunc):
        name = _ALIASES.get(func.__name__, func.__name__)
        if name in _UFUNCS and func.nin == len(ops) and func.nout == 1:
            return name, ops
        return None
    if func is np.where and len(ops) == 3:
        return "where", ops
    rec = getattr(t, "pointwise", None)
    if rec is None or len(ops) != 1:
        return None
    if rec[0] == "clip":
        lo, hi = rec[1], rec[2]
        if lo is None and hi is None:
            return None
        # np.clip with one bound is np.minimum / np.maximum (numpy's _clip)
        if lo is None:
            return "minimum", [ops[0], hi]
        if hi is None:
            return "maximum", [ops[0], lo]
        return "clip", [ops[0], lo, hi]
    if rec[0] == "round" and rec[1] == 0:
        return ("rint" if np.dtype(t.dtype).kind == "f" else "copy"), ops
    return None


def _dtype(a):
    return np.dtype(a.dtype)


def _loop(name, ops, out, is_array):
    """The loop type numpy runs ``name`` in, or None."""
    if name in _GIVES_BOOL:
        if out != np.dtype(bool):
            return None
        if len(ops) == 1:
            return _dtype(ops[0])
        try:
            loop = np.result_type(*[_dtype(o) if is_array(o) else o for o in ops])
        except (TypeError, ValueError, OverflowError):
            return None
        types = [_dtype(o) if is_array(o) else np.asarray(o).dtype for o in ops]
        if loop.kind == "f" and all(d.kind in "biu" for d in types):
            return None             # int64 against uint64: exact in NumPy 2, float64 in 1
        return loop
    return out                      # arithmetic, where, clip: the result's type


def _scalar(v, dt):
    """The bytes of scalar ``v`` in loop type ``dt`` as NumPy casts it, or None."""
    a = np.asarray(v)
    if a.ndim != 0 or a.dtype.kind not in "biuf":
        return None
    with np.errstate(all="ignore"):
        c = a.astype(dt)
    if dt.kind in "biu":
        if a.dtype.kind == "f" or int(c) != int(a):
            return None             # not held exactly: NumPy raises or special-cases it
    return c.tobytes()


def build(core, t, up):
    """The engine counterpart of the pointwise transform ``t`` (``up`` lowers an
    operand), or None."""
    from .dynamic_array import DynamicArray
    if not hasattr(core, "pointwise") or getattr(t, "device", None) not in (None, "cpu"):
        return None
    found = spec(t)
    if found is None:
        return None
    name, ops = found
    is_array = lambda o: isinstance(o, DynamicArray)        # noqa: E731
    arrays = [o for o in ops if is_array(o)]
    if not arrays or any(tuple(a.shape) != tuple(t.shape) for a in arrays):
        return None
    out = np.dtype(t.dtype)
    if name == "copy":
        return up(ops[0]) if _dtype(ops[0]) == out else None
    loop = _loop(name, ops, out, is_array)
    if loop is None or loop.name not in _TYPES or out.name not in _TYPES:
        return None
    operands, scalars = [], {}
    for k, o in enumerate(ops):
        if is_array(o):
            eng = up(o)
            if eng is None:
                return None
            operands.append(eng)
            continue
        want = np.dtype(bool) if name == "where" and k == 0 else loop
        b = _scalar(o, want)
        if b is None:
            return None
        if name in ("add", "multiply") and loop.kind == "f" and np.isnan(
                np.frombuffer(b, dtype=loop)[0]):
            return None             # which NaN NumPy keeps depends on the array's length
        operands.append(b)
        scalars[k] = b
    if not _agrees(core, name, loop, len(ops), scalars):
        return None
    return core.pointwise(name, operands, loop.name)


# ------------------------------------------------------------------- the NumPy check

_checked: dict = {}
_check_lock = threading.Lock()
#: the probe's least length: NumPy's SIMD body and a tail
_LENGTH = 1031


def _specials(dt):
    """Values where NumPy's rules could differ from the engine's, in type ``dt``."""
    if dt == np.dtype(bool):
        return np.array([False, True])
    if dt.kind == "f":
        f = np.finfo(dt)
        v = [-0.0, 0.0, np.nan, -np.nan, np.inf, -np.inf, 1.0, -1.0, 0.5, -0.5, 1.5, -2.5,
             2.5, 3.0, -7.25, float(f.tiny) / 4, float(f.max), -float(f.max), 1e-30]
        with np.errstate(all="ignore"):
            return np.array(v, dtype=np.float64).astype(dt)
    info = np.iinfo(dt)
    bits = dt.itemsize * 8
    v = {info.min, info.max, 0, 1, 2, 3, 7, bits - 1, bits, bits + 1, info.max // 2,
         info.max - 1}
    if info.min < 0:
        v |= {-1, -2, -bits, info.min + 1, info.min // 2}
    return np.array(sorted(v), dtype=dt)


def _probe_inputs(name, loop, n, scalars):
    """The probe's length, its operands (NumPy values) and their engine form (bytes,
    scalar?)."""
    s = _specials(loop)
    arrays = [k for k in range(n) if k not in scalars]
    # every combination of special values among the array operands, repeated to at
    # least _LENGTH, with a tail past NumPy's SIMD width
    grids = np.meshgrid(*([np.arange(len(s))] * len(arrays)), indexing="ij")
    combos = [g.reshape(-1) for g in grids]
    length = max(_LENGTH, len(combos[0]) + 7)
    reps = -(-length // len(combos[0]))
    vals, eng = [], []
    j = 0
    for k in range(n):
        want = np.dtype(bool) if name == "where" and k == 0 else loop
        if k in scalars:
            v = np.frombuffer(scalars[k], dtype=want)[0]
            vals.append(v)
            eng.append((scalars[k], True))
            continue
        idx = np.tile(combos[j], reps)[:length]
        j += 1
        if want == loop:
            a = s[idx]
        else:
            a = (idx % 3 == 0)              # where's condition
        a = np.ascontiguousarray(a, dtype=want)
        vals.append(a)
        eng.append((a.tobytes(), False))
    return length, vals, eng


def _numpy(name, vals):
    if name == "where":
        return np.where(*vals)
    if name == "clip":
        return np.clip(*vals)
    return getattr(np, name)(*vals)


def _agrees(core, name, loop, n, scalars):
    """Whether the engine computes ``name`` in ``loop`` (with these scalars) as this
    NumPy does, on the probe of special values (cached per process)."""
    key = (name, loop.name, n, tuple(sorted(scalars.items())))
    with _check_lock:
        hit = _checked.get(key)
    if hit is not None:
        return hit
    ok = False
    try:
        length, vals, eng = _probe_inputs(name, loop, n, scalars)
        with np.errstate(all="ignore"):
            want = np.asarray(_numpy(name, vals))
        out = np.dtype(bool) if name in _GIVES_BOOL else loop
        got = np.frombuffer(core.pointwise_apply(name, eng, length, loop.name), dtype=out)
        if want.dtype == out and want.shape == got.shape:
            if out.kind == "f":
                u = np.dtype(f"u{out.itemsize}")
                ok = bool(np.array_equal(want.view(u), got.view(u)))
            else:
                ok = bool(np.array_equal(want, got))
    except Exception:                       # noqa: BLE001 - no agreement: Python path
        ok = False
    with _check_lock:
        _checked[key] = ok
    return ok
