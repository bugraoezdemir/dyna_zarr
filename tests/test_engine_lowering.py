"""dyna's arrays read by the native engine (Workplan WP15, task DR6a; dyna_zarr/_lowering).

Every chain is read twice, with the engine and with the Python path, and must give the
same values, dtype and shape: whole (compute), region by region (_read_direct with the
keys transforms send: ints, steps, None, Ellipsis), and written by io.write. Links the
engine does not have stay in Python, and what is below them is still read by the engine.
"""
import os
import pickle
import zipfile

import numpy as np
import pytest
import zarr

from dyna_zarr import _engine, _lowering, io
from dyna_zarr import operations as o

pytestmark = pytest.mark.engine
pytest.importorskip("dyna_zarr_core")


@pytest.fixture(autouse=True)
def _engine_on():
    saved = _engine._choice
    _engine.set_engine(True)
    yield
    _engine._choice = saved


def _data(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.random(shape) > 0.5
    if np.dtype(dtype).kind == "f":
        return rng.normal(scale=100, size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(max(info.min, -10**6), min(info.max, 10**6), size=shape,
                        dtype=dtype, endpoint=True)


def _zarr(path, data, chunks=(2, 16, 16), **kw):
    z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype, **kw)
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


@pytest.fixture
def arr(tmp_path):
    return _zarr(tmp_path / "a.zarr", _data((6, 40, 50), "uint16"))


CHAINS = {
    "source": lambda a: a,
    "slice": lambda a: a[1:5, ::3, 4:],
    "int": lambda a: a[2],
    "negative": lambda a: a[-2:, -30:-3:4],
    "newaxis": lambda a: a[None, :, 5, None],
    "ellipsis": lambda a: a[..., 3:9],
    "transpose": lambda a: o.transpose(a, (2, 0, 1)),
    "swapaxes": lambda a: o.swap_axes(a, 0, 2),
    "expand_dims": lambda a: o.expand_dims(a, 1),
    "squeeze": lambda a: o.squeeze(a[2:3], 0),
    "concatenate": lambda a: o.concatenate([a, a[:, ::2]], axis=1),
    "stack": lambda a: o.stack([a, o.transpose(o.transpose(a, (1, 0, 2)), (1, 0, 2))], axis=2),
    "astype-float": lambda a: a.astype("float32"),
    "astype-narrow": lambda a: a.astype("int8"),
    "mean": lambda a: o.coarsen(a, (2, 2, 2)),
    "mean-odd": lambda a: o.coarsen(a, (1, 3, 4)),
    "mean-of-slice": lambda a: o.coarsen(a[:, 3:, :], (1, 2, 2))[:, ::2],
    "mean-of-mean": lambda a: o.coarsen(o.coarsen(a, (1, 2, 2)), (2, 2, 2)),
    "levels-chain": lambda a: o.transpose(o.coarsen(a.astype("float64"), (2, 2, 2)), (1, 2, 0)),
    # pointwise (DR7 step 7; tests/test_engine_pointwise.py has the rest)
    "add": lambda a: a + 1,
}

#: chains with a link the engine lacks: Python on top, the engine below where it can
MIXED = {
    "median": lambda a: o.coarsen(a, (2, 2, 2), "median"),
    "sin": lambda a: o.sin(a),
    "float-to-int": lambda a: a.astype("float32").astype("uint8"),
    "map_overlap": lambda a: a.map_overlap(lambda b: b * 2, depth=1),
    "flip": lambda a: o.flip(a, 1),
    "reduction": lambda a: a - a.mean(axis=0, keepdims=True).astype("uint16"),
}


@pytest.mark.parametrize("chain", list(CHAINS))
def test_lowered_chains_read_as_python(arr, chain):
    v = CHAINS[chain](arr)
    assert _lowering.lowered(v) is not None
    _same(*_both(v.compute))


@pytest.mark.parametrize("chain", list(MIXED))
def test_mixed_chains_read_as_python(arr, chain):
    v = MIXED[chain](arr)
    assert _lowering.lowered(v) is None
    assert _lowering.lowered(arr) is not None          # the source still is
    _same(*_both(v.compute))


def _keys(shape):
    """Region keys of the kinds dyna's transforms send to _read_direct (ints, slices with
    a positive step, NumPy ints, an empty range), inside ``shape``."""
    a, b, c = shape
    return [
        (slice(1, a - 1), slice(2, b - 3), slice(c // 3, c)),
        (a // 2, slice(None), slice(None, None, 2)),
        (slice(None, None, 3), b - 1, slice(1, 2)),
        (np.int64(0), slice(np.int64(0), np.int64(b), np.int64(5)), slice(None)),
        (slice(0, 0), slice(None), slice(None)),
    ]


@pytest.mark.parametrize("chain", ["source", "slice", "transpose", "mean", "concatenate",
                                   "astype-float", "levels-chain", "mean-of-mean"])
def test_region_reads_as_python(arr, chain):
    v = CHAINS[chain](arr)                              # all 3-D
    for key in _keys(v.shape):
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        _same(e, p)
        np.testing.assert_array_equal(e, v.compute()[key])


def test_newaxis_after_an_int_as_numpy(tmp_path):
    """The Python path put a new axis after an int index in the wrong place (dyna bug,
    found by comparing with the engine)."""
    data = _data((6, 40, 50), "uint16")
    a = _zarr(tmp_path / "a.zarr", data)
    for key in [(None, slice(None), 5, None), (5, None), (None, 2, None, slice(1, 3)),
                (slice(1, 3), None, 4)]:
        for on in (True, False):
            _engine.set_engine(on)
            np.testing.assert_array_equal(a[key].compute(), data[key])


def test_empty_coarsen_region(arr):
    v = o.coarsen(arr, (2, 2, 2))
    _same(*_both(lambda: np.asarray(v._read_direct((slice(0, 0), slice(None), slice(None))))))


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int16", "uint16", "int32",
                                   "uint32", "int64", "uint64", "float16", "float32",
                                   "float64"])
def test_every_dtype(tmp_path, dtype):
    a = _zarr(tmp_path / "a.zarr", _data((5, 21, 23), dtype), chunks=(2, 8, 8))
    for v in (a, a[1:, ::2], o.coarsen(a, (2, 2, 3)), o.transpose(a, (1, 2, 0))):
        assert _lowering.lowered(v) is not None
        _same(*_both(v.compute))


def test_written_as_python(arr, tmp_path):
    v = o.coarsen(o.concatenate([arr, arr], axis=0)[:, 2:, ::2], (2, 2, 1)).astype("float32")
    assert _lowering.lowered(v) is not None
    out = {}
    for on in (True, False):
        _engine.set_engine(on)
        path = tmp_path / f"out-{on}.zarr"
        io.write(v, str(path), chunks=(2, 8, 8), max_threads=3)
        out[on] = zarr.open_array(str(path), mode="r")[...]
    _same(out[True], out[False])


def test_sources(tmp_path):
    data = _data((3, 30, 40), "int32")
    # zarr v2
    v2 = _zarr(tmp_path / "v2.zarr", data, chunks=(1, 16, 16), zarr_format=2)
    assert _lowering.lowered(v2) is not None
    _same(*_both(v2.compute))
    # a finished zip
    folder = tmp_path / "img.ome.zarr"
    _zarr(folder / "0", data, chunks=(1, 16, 16))
    zpath = tmp_path / "img.ome.zarr.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(folder):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, tmp_path).replace(os.sep, "/"))
    z = io.read(str(zpath / "0"))
    assert _lowering.lowered(z) is not None
    _same(*_both(z.compute))
    # a TensorStore handle wrapped as is: lowered when stored and whole
    ts = pytest.importorskip("tensorstore")
    h = ts.open({"driver": "zarr3", "kvstore": {"driver": "file",
                 "path": str(tmp_path / "img.ome.zarr" / "0")}}).result()
    from dyna_zarr.dynamic_array import DynamicArray
    whole = DynamicArray(h)
    assert _lowering.lowered(whole) is not None
    _same(*_both(whole.compute))
    part = DynamicArray(h[1:, 5:])                      # a TensorStore view: not lowered
    assert _lowering.lowered(part) is None
    _same(*_both(part.compute))


def test_not_lowered_sources(tmp_path):
    from dyna_zarr.dynamic_array import DynamicArray
    data = _data((3, 30, 40), "uint16")
    assert _lowering.lowered(DynamicArray(data)) is None            # in memory
    big = zarr.create_array(str(tmp_path / "be.zarr"), shape=data.shape, chunks=(1, 16, 16),
                            dtype=">u2", zarr_format=2)
    big[...] = data
    b = io.read(str(tmp_path / "be.zarr"))
    if np.dtype(b.dtype).isnative:
        pytest.skip("this reader returns native byte order")
    assert _lowering.lowered(b) is None                              # big-endian
    _same(*_both(b.compute))


def test_engine_off_reads_in_python(arr):
    _engine.set_engine(False)
    assert _lowering.lowered(arr) is None
    _engine.set_engine(True)
    assert _lowering.lowered(arr) is not None


def test_pickles_without_the_engine_array(arr):
    v = o.coarsen(arr, (2, 2, 2))
    want = v.compute()
    assert _lowering.CACHE in v.__dict__
    back = pickle.loads(pickle.dumps(v))
    assert _lowering.CACHE not in back.__dict__
    np.testing.assert_array_equal(back.compute(), want)
