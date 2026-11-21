"""dyna's pointwise operations read by the native engine (Workplan WP15, task DR7 step 7).

The engine has the ufuncs whose results NumPy defines exactly (arithmetic, comparisons,
logic, bit operations, rounding, sqrt), where and clip, each run in NumPy's loop type
and checked once against the NumPy installed (dyna_zarr._lowering_pointwise). Every case
is read with the engine and with the Python path and must give the same bits, dtype and
shape, whole and region by region, and equal NumPy on the whole array. Operations NumPy
does not define to the bit stay in Python.
"""
import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from dyna_zarr import _engine, _lowering, io
from dyna_zarr import _lowering_pointwise as lp
from dyna_zarr import operations as o

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")
if not hasattr(core, "pointwise"):
    pytest.skip("this dyna-zarr-core has no pointwise operations", allow_module_level=True)

SHAPE = (7, 45, 38)
TYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
         "float32", "float64"]
NAMES = sorted(lp._UFUNCS) + ["where", "clip"]


@pytest.fixture(autouse=True)
def _engine_on():
    saved = _engine._choice
    _engine.set_engine(True)
    with np.errstate(all="ignore"):
        yield
    _engine._choice = saved


def _arity(name):
    return 3 if name in ("where", "clip") else getattr(np, name).nin


def _has_loop(name, loop):
    n = _arity(name)
    size = np.dtype(loop).itemsize
    try:
        core.pointwise_apply(name, [(b"\0" * (1 if name == "where" and k == 0 else size),
                                     True) for k in range(n)], 1, loop)
        return True
    except core.ReadError:
        return False


def _bits(a):
    a = np.asarray(a)
    return a.view(f"u{a.dtype.itemsize}") if a.dtype.kind == "f" else a


def _same_bits(got, want, msg=""):
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape, (got.dtype, want.dtype, msg)
    np.testing.assert_array_equal(_bits(got), _bits(want), err_msg=msg)


def _values(dtype, n, seed):
    """Random values of ``dtype`` with the special ones mixed in."""
    dt = np.dtype(dtype)
    rng = np.random.default_rng(seed)
    if dt == np.dtype(bool):
        return rng.random(n) < 0.5
    sp = lp._specials(dt)
    if dt.kind == "f":
        v = rng.normal(scale=50, size=n).astype(dt)
    else:
        info = np.iinfo(dt)
        v = rng.integers(info.min, info.max, size=n, dtype=dt, endpoint=True)
        v[::3] = rng.integers(max(info.min, -40), min(info.max, 40), size=v[::3].size)
    pick = rng.random(n) < 0.2
    v[pick] = rng.choice(sp, size=int(pick.sum()))
    return v


# ---------------------------------------------------------------------------- the kernel

@pytest.mark.parametrize("loop", TYPES)
@pytest.mark.parametrize("name", NAMES)
def test_kernel_equals_numpy(name, loop):
    """Every operation in every loop type it has: the NumPy check passes, and random
    values with the special ones mixed in give NumPy's bits."""
    if not _has_loop(name, loop):
        pytest.skip(f"no {loop} loop for {name}")
    dt = np.dtype(loop)
    n = _arity(name)
    scalars = {}
    if name == "clip":
        lo, hi = (-0.0, 3.5) if dt.kind == "f" else (2, 9 if dt != np.dtype(bool) else 1)
        scalars = {1: np.asarray(lo, dt).tobytes(), 2: np.asarray(hi, dt).tobytes()}
    used = name
    agrees = lp._agrees(core, name, dt, n, scalars)
    if not agrees and name in lp._NAN_VARIANTS and dt.kind == "f":
        used = name + ":nan_first"           # this NumPy keeps the first operand's NaN
        agrees = lp._agrees(core, used, dt, n, scalars)
    if not agrees:
        pytest.skip(f"this NumPy computes {name} in {loop} by a rule the engine does not "
                    "follow (NumPy's arm64 maximum, for one); the Python path reads it")
    length = 5003
    vals, eng = [], []
    for k in range(n):
        if k in scalars:
            vals.append(np.frombuffer(scalars[k], dtype=dt)[0])
            eng.append((scalars[k], True))
            continue
        v = _values(bool if name == "where" and k == 0 else dt, length, seed=k + 7)
        vals.append(v)
        eng.append((v.tobytes(), False))
    want = np.asarray(lp._numpy(name, vals))
    got = np.frombuffer(core.pointwise_apply(used, eng, length, loop), dtype=want.dtype)
    _same_bits(got, want)


def test_kernel_errors_name_the_way_out():
    with pytest.raises(core.ReadError, match="not one the engine has"):
        core.pointwise_apply("sin", [(b"\0" * 8, False)], 1, "float64")
    with pytest.raises(core.ReadError, match="no int16 loop"):
        core.pointwise_apply("sqrt", [(b"\0" * 2, False)], 1, "int16")
    with pytest.raises(core.ReadError, match="takes 2 operands"):
        core.pointwise_apply("add", [(b"\0" * 2, False)], 1, "int16")
    with pytest.raises(core.ReadError, match="bytes, not"):
        core.pointwise_apply("add", [(b"\0" * 2, False), (b"\0" * 3, False)], 1, "int16")


# ------------------------------------------------------------------------ through dyna

def _data(shape, dtype, seed=0):
    return _values(dtype, int(np.prod(shape)), seed).reshape(shape)


def _zarr(path, data, chunks=(2, 16, 16)):
    z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype)
    z[...] = data
    return io.read(str(path))


def _lowered_unless_numpy_differs(v):
    """The engine counterpart of ``v``; None only where the NumPy check refused an
    operation of it (a NumPy whose rule differs keeps the Python path, by design)."""
    lp._checked.clear()
    eng = _lowering.lowered(v)
    if eng is None:
        assert False in lp._checked.values(), "not lowered, and no NumPy check refused it"
    return eng


def _both(fn):
    _engine.set_engine(True)
    e = fn()
    _engine.set_engine(False)
    p = fn()
    _engine.set_engine(True)
    return e, p


def _keys(shape):
    a, b, c = shape
    return [
        (slice(None), slice(None), slice(None)),
        (slice(1, a - 1), slice(2, b - 3), slice(c // 3, c)),
        (a // 2, slice(None), slice(None, None, 2)),
        (slice(None, None, 3), b - 1, slice(1, 2)),
        (slice(a - 1, a), slice(b - 2, b), slice(c - 3, c)),
    ]


#: name -> (dyna's expression of a and b, NumPy's of x and y)
CASES = {
    "add": (lambda a, b: a + b, lambda x, y: x + y),
    "subtract-scalar": (lambda a, b: a - 3, lambda x, y: x - 3),
    "reflected": (lambda a, b: 7 - a, lambda x, y: 7 - x),
    "multiply-float": (lambda a, b: a * 0.37, lambda x, y: x * 0.37),
    "divide": (lambda a, b: a / b, lambda x, y: x / y),
    "negative": (lambda a, b: -a, lambda x, y: -x),
    "absolute": (lambda a, b: abs(a), lambda x, y: abs(x)),
    "square": (lambda a, b: o.square(a), lambda x, y: np.square(x)),
    "sqrt": (lambda a, b: o.sqrt(a), lambda x, y: np.sqrt(x)),
    "maximum": (lambda a, b: o.maximum(a, b), lambda x, y: np.maximum(x, y)),
    "minimum-scalar": (lambda a, b: o.minimum(a, 5), lambda x, y: np.minimum(x, 5)),
    "greater": (lambda a, b: a > b, lambda x, y: x > y),
    "less-equal-scalar": (lambda a, b: a <= 4, lambda x, y: x <= 4),
    "equal-float": (lambda a, b: a == 2.5, lambda x, y: x == 2.5),
    "not-equal": (lambda a, b: a != b, lambda x, y: x != y),
    "where": (lambda a, b: o.where(a > b, a, b), lambda x, y: np.where(x > y, x, y)),
    "where-scalar": (lambda a, b: o.where(a > 3, a, 0), lambda x, y: np.where(x > 3, x, 0)),
    "clip": (lambda a, b: o.clip(a, 2, 30), lambda x, y: np.clip(x, 2, 30)),
    "clip-low": (lambda a, b: o.clip(a, 3, None), lambda x, y: np.clip(x, 3, None)),
    "round": (lambda a, b: o.round(a), lambda x, y: np.round(x)),
    "floor": (lambda a, b: o.floor(a), lambda x, y: np.floor(x)),
    "rint": (lambda a, b: o.rint(a), lambda x, y: np.rint(x)),
    "sign": (lambda a, b: o.sign(a), lambda x, y: np.sign(x)),
    "isnan": (lambda a, b: o.isnan(a), lambda x, y: np.isnan(x)),
    "signbit": (lambda a, b: o.signbit(a), lambda x, y: np.signbit(x)),
    "masks": (lambda a, b: (a > 3) & ~(b < 10) | (a == b),
              lambda x, y: (x > 3) & ~(y < 10) | (x == y)),
    "logical-xor": (lambda a, b: o.logical_xor(a, b), lambda x, y: np.logical_xor(x, y)),
    "bits": (lambda a, b: (a & b) ^ ~a, lambda x, y: (x & y) ^ ~x),
    "shift": (lambda a, b: a << 3, lambda x, y: x << 3),
    "copysign": (lambda a, b: o.copysign(a, b), lambda x, y: np.copysign(x, y)),
    "threshold-cast": (lambda a, b: ((a * 0.5) > 3).astype("uint8"),
                       lambda x, y: ((x * 0.5) > 3).astype("uint8")),
}

PAIRS = [("uint8", "uint8"), ("uint16", "int16"), ("int32", "float32"),
         ("float32", "float32"), ("float64", "uint8"), ("int64", "int64"), ("bool", "bool")]


@pytest.mark.parametrize("pair", PAIRS, ids=["-".join(p) for p in PAIRS])
@pytest.mark.parametrize("case", list(CASES))
def test_pointwise_reads_as_python_and_numpy(tmp_path, case, pair):
    build, ref = CASES[case]
    x, y = _data(SHAPE, pair[0], 1), _data(SHAPE, pair[1], 2)
    try:
        want = ref(x, y)
    except (TypeError, np.exceptions.DTypePromotionError):
        pytest.skip("NumPy has no loop for these types")
    v = build(_zarr(tmp_path / "a.zarr", x), _zarr(tmp_path / "b.zarr", y))
    if want.dtype.name in lp._TYPES:
        _lowered_unless_numpy_differs(v)
    for key in _keys(v.shape):
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        _same_bits(e, p, f"{key}")
        _same_bits(e, want[key], f"{key}")


#: expressions the engine does not take: they keep the Python path
STAYS = {
    "sin": lambda a, b, c: o.sin(a),
    "exp": lambda a, b, c: o.exp(a),
    "power": lambda a, b, c: a ** 2,
    "fmax": lambda a, b, c: o.fmax(a, b),
    "floor-divide": lambda a, b, c: a // 3.0,
    "mod": lambda a, b, c: a % 3.0,
    "nan-scalar": lambda a, b, c: a + np.nan,
    "nan-times": lambda a, b, c: np.float32(np.nan) * a,
    "broadcast": lambda a, b, c: a + b[:1],
    "float16": lambda a, b, c: a.astype("float16") + 1,
    "complex": lambda a, b, c: a * 1j,
    "round-2": lambda a, b, c: o.round(a, 2),
    "function": lambda a, b, c: o.map_blocks(lambda u: u + 1, a),
    "int64-uint64": lambda a, b, c: c.astype("int64") > c.astype("uint64"),
    "out-of-range": lambda a, b, c: c < 300,
}


@pytest.mark.parametrize("expr", list(STAYS))
def test_other_expressions_keep_the_python_path(tmp_path, expr):
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float32", 1))
    b = _zarr(tmp_path / "b.zarr", _data(SHAPE, "float32", 2))
    c = _zarr(tmp_path / "c.zarr", _data(SHAPE, "uint8", 3))
    v = STAYS[expr](a, b, c)
    assert _lowering.lowered(v) is None
    assert _lowering.lowered(a) is not None              # the sources still are
    for key in _keys(v.shape)[:2]:
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        _same_bits(e, p)


def test_a_gpu_device_is_not_lowered(tmp_path):
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float32"))
    assert _lowering.lowered(o.map_blocks(np.add, a, a, device="cuda")) is None
    _lowered_unless_numpy_differs(o.map_blocks(np.add, a, a, device="cpu"))


def test_an_operation_numpy_disagrees_on_keeps_the_python_path(tmp_path):
    """The NumPy check decides: an operation it found different stays in Python."""
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float32"))
    # both of the engine's NaN variants disagree
    keys = [("subtract", "float32", 2, ()), ("subtract:nan_first", "float32", 2, ())]
    saved = {k: lp._checked.get(k) for k in keys}
    for k in keys:
        lp._checked[k] = False
    try:
        assert _lowering.lowered(a - a) is None
    finally:
        for k, v in saved.items():
            if v is None:
                lp._checked.pop(k, None)
            else:
                lp._checked[k] = v
    _lowered_unless_numpy_differs(a - a)


def test_a_filter_chain_written_as_python(tmp_path):
    """gaussian -> threshold -> where -> astype, written in regions by io.write: the
    whole chain in the engine, as the Python path and scipy / NumPy give it."""
    data = _data((12, 96, 80), "uint16", 4)
    a = _zarr(tmp_path / "a.zarr", data, chunks=(4, 32, 32))
    g = o.gaussian_filter(a, 1.5)
    v = o.where(g > 30000, g, 0).astype("uint8")
    _lowered_unless_numpy_differs(v)
    out = {}
    for on in (True, False):
        _engine.set_engine(on)
        path = tmp_path / f"out-{on}.zarr"
        io.write(v, str(path), chunks=(4, 32, 32), max_threads=3, memory_budget_mb=24)
        out[on] = zarr.open_array(str(path), mode="r")[...]
    _same_bits(out[True], out[False])
    gd = ndi.gaussian_filter(data, 1.5)
    _same_bits(out[True], np.where(gd > 30000, gd, 0).astype("uint8"))
