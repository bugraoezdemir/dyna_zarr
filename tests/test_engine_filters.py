"""dyna's named filters read by the native engine (Workplan WP15, task DR7 steps 2-5).

gaussian_filter records which filter it is on its MapOverlapTransform; dyna_zarr's
_lowering builds the engine's separable view from scipy's own kernel weights. Every case
is read with the engine and with the Python path and must give the same values, dtype
and shape, whole, region by region and written by io.write, and equal scipy on the
whole array. Calls the engine cannot match exactly stay in Python.
"""
import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from dyna_zarr import _engine, _lowering, io
from dyna_zarr import operations as o

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")
if not hasattr(core.Array, "separable"):
    pytest.skip("this dyna-zarr-core has no filter views", allow_module_level=True)

SHAPE = (7, 45, 38)


@pytest.fixture(autouse=True)
def _engine_on():
    saved = _engine._choice
    _engine.set_engine(True)
    yield
    _engine._choice = saved


def _data(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return rng.normal(scale=100, size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(max(info.min, -10**6), min(info.max, 10**6), size=shape,
                        dtype=dtype, endpoint=True)


def _zarr(path, data, chunks=(2, 16, 16)):
    z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype)
    z[...] = data
    return io.read(str(path))


def _both(fn):
    """fn() with the engine and with the Python path."""
    _engine.set_engine(True)
    e = fn()
    _engine.set_engine(False)
    p = fn()
    _engine.set_engine(True)
    return e, p


def _same(e, p):
    assert e.dtype == p.dtype and e.shape == p.shape
    np.testing.assert_array_equal(e, p)


def _keys(shape):
    a, b, c = shape
    return [
        (slice(None), slice(None), slice(None)),
        (slice(1, a - 1), slice(2, b - 3), slice(c // 3, c)),
        (a // 2, slice(None), slice(None, None, 2)),
        (slice(None, None, 3), b - 1, slice(1, 2)),
        (0, 0, slice(0, 5)),
        (slice(a - 1, a), slice(b - 2, b), slice(c - 3, c)),
        (slice(0, 0), slice(None), slice(None)),
    ]


#: (sigma, extra keywords); per-axis sigmas, a sigma of 0 (that axis untouched), orders
CASES = {
    "scalar": (2.0, {}),
    "per-axis": ((1.0, 2.0, 1.5), {}),
    "zero-axis": ((0.0, 3.0, 0.7), {}),
    "orders": ((0.8, 1.5, 1.2), {"order": (0, 1, 2)}),
    "large": ((1.0, 6.0, 9.0), {}),          # radius 36 > the last axis' 38 / 2
}


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "float32", "float64"])
@pytest.mark.parametrize("mode", ["reflect", "mirror", "nearest", "constant"])
@pytest.mark.parametrize("case", list(CASES))
def test_gaussian_reads_as_python_and_scipy(tmp_path, dtype, mode, case):
    sigma, kw = CASES[case]
    if np.dtype(dtype).kind != "f" and "order" in kw:
        kw = {"order": (0, 1, 0)}            # one derivative: integers keep their range
    if mode == "constant":
        kw = {**kw, "cval": 3.0}
    data = _data(SHAPE, dtype)
    v = o.gaussian_filter(_zarr(tmp_path / "a.zarr", data), sigma, boundary=mode, **kw)
    assert _lowering.lowered(v) is not None
    want = ndi.gaussian_filter(data, sigma, mode=mode, **kw)
    for key in _keys(v.shape):
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        _same(e, p)
        np.testing.assert_array_equal(e, want[key])


def test_written_as_python(tmp_path):
    """io.write in regions (a small budget), with the engine and with the Python path."""
    data = _data((12, 96, 80), "uint16")
    v = o.gaussian_filter(_zarr(tmp_path / "a.zarr", data, chunks=(4, 32, 32)),
                          (1.0, 2.5, 2.0), boundary="mirror")
    assert _lowering.lowered(v) is not None
    out = {}
    for on in (True, False):
        _engine.set_engine(on)
        path = tmp_path / f"out-{on}.zarr"
        io.write(v, str(path), chunks=(4, 32, 32), max_threads=3, memory_budget_mb=24)
        out[on] = zarr.open_array(str(path), mode="r")[...]
    _same(out[True], out[False])
    np.testing.assert_array_equal(out[True], ndi.gaussian_filter(data, (1.0, 2.5, 2.0),
                                                                 mode="mirror"))


CHAINS = {
    "filter-of-mean": lambda a: o.gaussian_filter(o.coarsen(a, (1, 2, 2)), 1.5),
    "filter-of-slice": lambda a: o.gaussian_filter(a[1:6, ::2, 3:], (0.5, 1.0, 2.0)),
    "cast-of-filter": lambda a: o.gaussian_filter(a, 1.0).astype("float32"),
    "filter-of-cast": lambda a: o.gaussian_filter(a.astype("float32"), 1.0, order=1),
    "filter-of-filter": lambda a: o.gaussian_filter(o.gaussian_filter(a, 2.0), 1.0,
                                                    boundary="nearest"),
    "slice-of-filter": lambda a: o.transpose(o.gaussian_filter(a, 1.2)[2:, 5:30], (2, 0, 1)),
}


@pytest.mark.parametrize("chain", list(CHAINS))
def test_chains_read_as_python(tmp_path, chain):
    v = CHAINS[chain](_zarr(tmp_path / "a.zarr", _data(SHAPE, "uint16")))
    assert _lowering.lowered(v) is not None
    _same(*_both(v.compute))


#: calls the engine does not take: they keep the Python path (and still read as before)
STAYS = {
    # the Python path wraps each read's box around itself, not the array (see _lowering)
    "wrap": lambda a: o.gaussian_filter(a, 1.0, boundary="wrap"),
    "radius": lambda a: o.gaussian_filter(a, 1.0, radius=2),
    "output": lambda a: o.gaussian_filter(a, 1.0, output=np.float64),
    "user-function": lambda a: o.map_overlap(a, lambda b: ndi.gaussian_filter(b, 1.0), 4),
    "block-info": lambda a: o.map_overlap(a, lambda b, loc: b, 1, block_info=True),
    "align": lambda a: o.map_overlap(a, lambda b: b, 1, align=8),
    # floats: the Python path's running sum depends on the region read (see _lowering)
    "uniform-float": lambda a: o.uniform_filter(a, 3),
    "median-footprint": lambda a: o.median_filter(a, 3, footprint=np.ones((3, 3, 3), bool)),
    "percentile-float32": lambda a: o.percentile_filter(a, np.float32(30.0), 3),
    "minimum-origin": lambda a: o.minimum_filter(a, 3, origin=1),
    "gradient-wrap": lambda a: o.gaussian_gradient_magnitude(a, 1.0, boundary="wrap"),
}


@pytest.mark.parametrize("call", list(STAYS))
def test_other_calls_keep_the_python_path(tmp_path, call):
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float32"))
    v = STAYS[call](a)
    assert _lowering.lowered(v) is None
    assert _lowering.lowered(a) is not None              # the source still is
    _same(*_both(v.compute))


def test_unsupported_types_keep_the_python_path(tmp_path):
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float16"))
    assert _lowering.lowered(o.gaussian_filter(a, 1.0)) is None


def test_a_gpu_device_is_not_lowered(tmp_path):
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float32"))
    assert _lowering.lowered(o.gaussian_filter(a, 1.0, device="cuda")) is None


def test_a_filter_following_the_device_context_stays_out_of_a_gpu_context(tmp_path):
    """device=None follows the context of the read (cupy inside a GPU context); the
    engine reads on the CPU, so it stands aside there, for every array built on it."""
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float32"))
    g = o.gaussian_filter(a, 1.0)
    on_top = g[1:4].astype("float64")
    pinned = o.gaussian_filter(a, 1.0, device="cpu")
    for v in (g, on_top, pinned):
        assert _lowering.lowered(v) is not None
    assert g.__dict__[_lowering.INHERITS] and on_top.__dict__[_lowering.INHERITS]
    assert not pinned.__dict__[_lowering.INHERITS]
    assert not a.__dict__[_lowering.INHERITS]
    from dyna_zarr.operations._backend import device_context
    with device_context("cuda"):
        assert _lowering.read(g, (slice(0, 2),)) is None
        assert _lowering.read(pinned, (slice(0, 2),)) is not None
    assert _lowering.read(g, (slice(0, 2),)) is not None



# --------------------------------------------------------------------- step 3 (DR7)

#: name -> (the filter, whether the engine takes it for a dtype)
_INT32 = {"int8", "uint8", "int16", "uint16", "int32", "uint32"}
STEP3 = {
    "uniform": (lambda a, m, kw: o.uniform_filter(a, (2, 3, 4), boundary=m, **kw),
                lambda dt: dt in _INT32),
    "minimum": (lambda a, m, kw: o.minimum_filter(a, (3, 2, 5), boundary=m, **kw),
                lambda dt: True),
    "maximum": (lambda a, m, kw: o.maximum_filter(a, 4, boundary=m, **kw),
                lambda dt: True),
    "grey_erosion": (lambda a, m, kw: o.grey_erosion(a, (2, 3, 1), boundary=m, **kw),
                     lambda dt: True),
    "grey_dilation": (lambda a, m, kw: o.grey_dilation(a, (4, 3, 2), boundary=m, **kw),
                      lambda dt: True),
    "laplace": (lambda a, m, kw: o.laplace(a, boundary=m, **kw), lambda dt: True),
    "gaussian_laplace": (lambda a, m, kw: o.gaussian_laplace(a, (1.0, 1.5, 0.0),
                                                            boundary=m, **kw),
                         lambda dt: True),
    "gaussian_gradient_magnitude": (
        lambda a, m, kw: o.gaussian_gradient_magnitude(a, (0.8, 1.2, 2.0), boundary=m, **kw),
        lambda dt: dt in ("float32", "float64")),
}


def _scipy(name, data, mode, kw):
    f = {"uniform": lambda: ndi.uniform_filter(data, (2, 3, 4), mode=mode, **kw),
         "minimum": lambda: ndi.minimum_filter(data, (3, 2, 5), mode=mode, **kw),
         "maximum": lambda: ndi.maximum_filter(data, 4, mode=mode, **kw),
         "grey_erosion": lambda: ndi.grey_erosion(data, (2, 3, 1), mode=mode, **kw),
         "grey_dilation": lambda: ndi.grey_dilation(data, (4, 3, 2), mode=mode, **kw),
         "laplace": lambda: ndi.laplace(data, mode=mode, **kw),
         "gaussian_laplace": lambda: ndi.gaussian_laplace(data, (1.0, 1.5, 0.0), mode=mode,
                                                          **kw),
         "gaussian_gradient_magnitude": lambda: ndi.gaussian_gradient_magnitude(
             data, (0.8, 1.2, 2.0), mode=mode, **kw)}
    return f[name]()


def _with_specials(data):
    """NaN and signed zeros in a float array (where min / max order matters)."""
    if data.dtype.kind == "f":
        flat = data.reshape(-1)
        flat[[3, 50, 400]] = [np.nan, -0.0, 0.0]
    return data


@pytest.mark.parametrize("dtype", ["uint8", "int16", "uint16", "int32", "float32",
                                   "float64"])
@pytest.mark.parametrize("mode", ["reflect", "mirror", "nearest", "constant"])
@pytest.mark.parametrize("name", list(STEP3))
def test_step3_filters_read_as_python_and_scipy(tmp_path, dtype, mode, name):
    build, takes = STEP3[name]
    kw = {"cval": 3.0} if mode == "constant" else {}
    data = _data(SHAPE, dtype)
    if name in ("minimum", "maximum", "grey_erosion", "grey_dilation"):
        data = _with_specials(data)
    v = build(_zarr(tmp_path / "a.zarr", data), mode, kw)
    assert (_lowering.lowered(v) is not None) == takes(dtype)
    want = _scipy(name, data, mode, kw)
    # uniform_filter of floats stays on the Python path, whose running sum starts at each
    # read's box: equal to scipy on the whole array only to rounding (float64: 3.6e-14
    # measured), which is why the engine does not take it
    rounding = name == "uniform" and not takes(dtype)
    for key in _keys(v.shape):
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        _same(e, p)
        if rounding:
            np.testing.assert_allclose(e, want[key], rtol=1e-6, atol=1e-10)
            continue
        np.testing.assert_array_equal(e, want[key])
        if e.dtype.kind == "f":
            np.testing.assert_array_equal(np.signbit(e), np.signbit(want[key]))


def test_step3_written_as_python(tmp_path):
    """A chain of step 3 filters written in regions, engine against Python path."""
    data = _data((12, 96, 80), "int16")
    a = _zarr(tmp_path / "a.zarr", data, chunks=(4, 32, 32))
    v = o.grey_dilation(o.laplace(o.uniform_filter(a, 3), boundary="nearest"), 4)
    assert _lowering.lowered(v) is not None
    out = {}
    for on in (True, False):
        _engine.set_engine(on)
        path = tmp_path / f"out-{on}.zarr"
        io.write(v, str(path), chunks=(4, 32, 32), max_threads=3, memory_budget_mb=24)
        out[on] = zarr.open_array(str(path), mode="r")[...]
    _same(out[True], out[False])
    want = ndi.grey_dilation(ndi.laplace(ndi.uniform_filter(data, 3), mode="nearest"), 4)
    np.testing.assert_array_equal(out[True], want)


# --------------------------------------------------------------------- step 4 (DR7)

#: name -> (dyna's filter, scipy's); ranks inside the window, the first and the last
#: (scipy's minimum / maximum filters), percentiles and even sizes
STEP4 = {
    "median": (lambda a, m, kw: o.median_filter(a, 3, boundary=m, **kw),
               lambda d, m, kw: ndi.median_filter(d, 3, mode=m, **kw)),
    "median-flat": (lambda a, m, kw: o.median_filter(a, (1, 5, 5), boundary=m, **kw),
                    lambda d, m, kw: ndi.median_filter(d, (1, 5, 5), mode=m, **kw)),
    "rank-even": (lambda a, m, kw: o.rank_filter(a, 5, (2, 3, 4), boundary=m, **kw),
                  lambda d, m, kw: ndi.rank_filter(d, 5, (2, 3, 4), mode=m, **kw)),
    "rank-negative": (lambda a, m, kw: o.rank_filter(a, -3, (3, 3, 2), boundary=m, **kw),
                      lambda d, m, kw: ndi.rank_filter(d, -3, (3, 3, 2), mode=m, **kw)),
    "rank-first": (lambda a, m, kw: o.rank_filter(a, 0, (3, 1, 4), boundary=m, **kw),
                   lambda d, m, kw: ndi.rank_filter(d, 0, (3, 1, 4), mode=m, **kw)),
    "rank-last": (lambda a, m, kw: o.rank_filter(a, -1, 3, boundary=m, **kw),
                  lambda d, m, kw: ndi.rank_filter(d, -1, 3, mode=m, **kw)),
    "percentile": (lambda a, m, kw: o.percentile_filter(a, 37.5, (3, 4, 3), boundary=m, **kw),
                   lambda d, m, kw: ndi.percentile_filter(d, 37.5, (3, 4, 3), mode=m, **kw)),
    "percentile-top": (lambda a, m, kw: o.percentile_filter(a, -10, 3, boundary=m, **kw),
                       lambda d, m, kw: ndi.percentile_filter(d, -10, 3, mode=m, **kw)),
    "percentile-100": (lambda a, m, kw: o.percentile_filter(a, 100, (2, 2, 3), boundary=m,
                                                            **kw),
                       lambda d, m, kw: ndi.percentile_filter(d, 100, (2, 2, 3), mode=m,
                                                              **kw)),
}


def _ties_and_specials(shape, dtype, seed=0):
    """Few distinct values; floats with NaN, -0.0 and 0.0 (where scipy's quickselect
    order decides)."""
    rng = np.random.default_rng(seed)
    data = (rng.integers(-6, 7, size=shape) * 7).astype(dtype) if np.dtype(dtype).kind in "if"         else rng.integers(0, 13, size=shape).astype(dtype) * 7
    if data.dtype.kind == "f":
        flat = data.reshape(-1)
        pick = rng.random(flat.size)
        flat[pick < 0.03] = np.nan
        flat[(pick >= 0.03) & (pick < 0.1)] = -0.0
    return data


@pytest.mark.parametrize("dtype", ["uint8", "int8", "uint16", "int32", "uint64", "int64",
                                   "float32", "float64"])
@pytest.mark.parametrize("mode", ["reflect", "mirror", "nearest", "constant"])
@pytest.mark.parametrize("name", list(STEP4))
def test_step4_filters_read_as_python_and_scipy(tmp_path, dtype, mode, name):
    build, ref = STEP4[name]
    kw = {"cval": 14.0} if mode == "constant" else {}
    data = _ties_and_specials(SHAPE, dtype)
    v = build(_zarr(tmp_path / "a.zarr", data), mode, kw)
    assert _lowering.lowered(v) is not None
    want = ref(data, mode, kw)
    # the first and last ranks are sliding minima / maxima: through a NaN, scipy's values
    # depend on where a line starts, so the Python path's region reads differ from scipy
    # on the whole array (the engine hands those reads to it: see the next test)
    whole_only = name in ("rank-first", "rank-last", "percentile-100") and data.dtype.kind == "f"
    for i, key in enumerate(_keys(v.shape)):
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        _same(e, p)
        if whole_only and i > 0:
            continue
        np.testing.assert_array_equal(e, want[key])
        if e.dtype.kind == "f":
            np.testing.assert_array_equal(np.signbit(e), np.signbit(want[key]))


@pytest.mark.parametrize("name", ["minimum", "maximum", "grey_erosion", "grey_dilation"])
@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_sliding_extremes_through_nan_read_as_python(tmp_path, name, mode):
    """Many NaNs: a read whose pass starts inside the array goes back to the Python
    path (UsePythonPath), so engine and Python path agree on every read; reads from the
    array's start stay with the engine, and a whole read equals scipy."""
    build, _ = STEP3[name]
    kw = {"cval": 3.0} if mode == "constant" else {}
    data = _ties_and_specials(SHAPE, "float32", seed=5)
    v = build(_zarr(tmp_path / "a.zarr", data), mode, kw)
    assert _lowering.lowered(v) is not None
    for key in _keys(v.shape):
        _same(*_both(lambda: np.asarray(v._read_direct(key))))
    whole = (slice(None),) * 3
    assert _lowering.read(v, whole) is not None
    assert _lowering.read(v, (slice(2, 5), slice(10, 30), slice(5, 20))) is None
    np.testing.assert_array_equal(np.asarray(v._read_direct(whole)), _scipy(name, data, mode, kw))


@pytest.mark.parametrize("dtype, cval, takes", [
    ("uint16", 0.0, True), ("int32", -5.0, True),
    # scipy's 1-D routine selects among int64 or float values, and converts cval with
    # NumPy: the engine takes integers up to 32 bits, with a cval the type holds
    ("int64", 0.0, False), ("float32", 0.0, False), ("uint8", 2.5, False),
    ("uint8", 300.0, False)])
def test_step4_on_1d_arrays(tmp_path, dtype, cval, takes):
    data = _ties_and_specials((300,), dtype)
    a = _zarr(tmp_path / "a.zarr", data, chunks=(64,))
    v = o.median_filter(a, 5, boundary="constant", cval=cval)
    assert (_lowering.lowered(v) is not None) == takes
    e, p = _both(v.compute)
    _same(e, p)
    if takes:
        np.testing.assert_array_equal(e, ndi.median_filter(data, 5, mode="constant", cval=cval))
    # rank 0: scipy's minimum filter, which the engine reads for every type
    assert _lowering.lowered(o.rank_filter(a, 0, 5)) is not None


def test_step4_errors_stay_scipys(tmp_path):
    """A rank outside the window keeps the Python path, which raises scipy's error."""
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "uint16"))
    for v in (o.rank_filter(a, 27, 3), o.percentile_filter(a, 120, 3)):
        assert _lowering.lowered(v) is None
        with pytest.raises(RuntimeError):
            v.compute()


def test_step4_written_as_python(tmp_path):
    """A median of a filter written in regions, engine against Python path."""
    data = _ties_and_specials((12, 96, 80), "float32", seed=3)
    a = _zarr(tmp_path / "a.zarr", data, chunks=(4, 32, 32))
    v = o.percentile_filter(o.median_filter(a, (3, 5, 5), boundary="mirror"), 80, 3)
    assert _lowering.lowered(v) is not None
    out = {}
    for on in (True, False):
        _engine.set_engine(on)
        path = tmp_path / f"out-{on}.zarr"
        io.write(v, str(path), chunks=(4, 32, 32), max_threads=3, memory_budget_mb=24)
        out[on] = zarr.open_array(str(path), mode="r")[...]
    _same(out[True], out[False])
    want = ndi.percentile_filter(ndi.median_filter(data, (3, 5, 5), mode="mirror"), 80, 3)
    np.testing.assert_array_equal(out[True], want)



# --------------------------------------------------------------------- step 5 (DR7)

def _kernel(shape, seed=0, zeros=False):
    w = np.random.default_rng(seed).uniform(-1, 1, shape)
    if zeros:
        w.reshape(-1)[::4] = 0.0
    return w


STEP5 = {
    "correlate": (lambda a, m, kw: o.correlate(a, _kernel((3, 3, 3)), boundary=m, **kw),
                  lambda d, m, kw: ndi.correlate(d, _kernel((3, 3, 3)), mode=m, **kw)),
    "convolve": (lambda a, m, kw: o.convolve(a, _kernel((3, 3, 3), 1), boundary=m, **kw),
                 lambda d, m, kw: ndi.convolve(d, _kernel((3, 3, 3), 1), mode=m, **kw)),
    "correlate-even": (lambda a, m, kw: o.correlate(a, _kernel((2, 3, 4), 2), boundary=m, **kw),
                       lambda d, m, kw: ndi.correlate(d, _kernel((2, 3, 4), 2), mode=m, **kw)),
    "convolve-even": (lambda a, m, kw: o.convolve(a, _kernel((2, 4, 1), 3, True), boundary=m,
                                                  **kw),
                      lambda d, m, kw: ndi.convolve(d, _kernel((2, 4, 1), 3, True), mode=m,
                                                    **kw)),
    "convolve-int": (lambda a, m, kw: o.convolve(a, np.ones((1, 3, 3), "int64"), boundary=m,
                                                 **kw),
                     lambda d, m, kw: ndi.convolve(d, np.ones((1, 3, 3), "int64"), mode=m,
                                                   **kw)),
}


@pytest.mark.parametrize("dtype", ["uint8", "int16", "uint16", "int32", "uint64", "float32",
                                   "float64"])
@pytest.mark.parametrize("mode", ["reflect", "mirror", "nearest", "constant"])
@pytest.mark.parametrize("name", list(STEP5))
def test_step5_filters_read_as_python_and_scipy(tmp_path, dtype, mode, name):
    build, ref = STEP5[name]
    kw = {"cval": -3.5} if mode == "constant" else {}
    data = _with_specials(_data(SHAPE, dtype))
    v = build(_zarr(tmp_path / "a.zarr", data), mode, kw)
    assert _lowering.lowered(v) is not None
    want = ref(data, mode, kw)
    for key in _keys(v.shape):
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        _same(e, p)
        np.testing.assert_array_equal(e, want[key])
        if e.dtype.kind == "f":
            np.testing.assert_array_equal(np.signbit(e), np.signbit(want[key]))


def test_step5_complex_weights_keep_the_python_path(tmp_path):
    a = _zarr(tmp_path / "a.zarr", _data(SHAPE, "float32"))
    w = _kernel((3, 3, 3)) + 0j
    assert _lowering.lowered(o.correlate(a, w)) is None
    assert _lowering.lowered(o.convolve(a, w, boundary="wrap")) is None


def test_step5_written_as_python(tmp_path):
    """A convolution of a median written in regions, engine against Python path."""
    data = _data((12, 96, 80), "uint16")
    a = _zarr(tmp_path / "a.zarr", data, chunks=(4, 32, 32))
    w = _kernel((3, 5, 5), 7)
    v = o.convolve(o.median_filter(a, 3), w, boundary="nearest")
    assert _lowering.lowered(v) is not None
    out = {}
    for on in (True, False):
        _engine.set_engine(on)
        path = tmp_path / f"out-{on}.zarr"
        io.write(v, str(path), chunks=(4, 32, 32), max_threads=3, memory_budget_mb=24)
        out[on] = zarr.open_array(str(path), mode="r")[...]
    _same(out[True], out[False])
    np.testing.assert_array_equal(out[True], ndi.convolve(ndi.median_filter(data, 3), w,
                                                          mode="nearest"))
