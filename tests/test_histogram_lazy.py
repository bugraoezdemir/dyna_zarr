"""histogram is lazy, exact against numpy, and costs the minimum number of passes.

It used to compute immediately (three passes with an automatic range: min, max, counts)
and returned numpy arrays. Now it returns lazy (counts, edges) - as dask does for counts -
reads nothing until used, fuses min+max into one pass (so an automatic range costs 2
passes, an explicit range 1), and caches the counts until io.clear_cache().
"""
import contextlib
import io as _stdio
import threading

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations as ops


@pytest.fixture
def source_reads(monkeypatch):
    lock = threading.Lock()
    n = {"elems": 0}
    orig = DynamicArray._read_direct

    def rd(self, key):
        out = orig(self, key)
        if self._transform is None and self.ndim >= 2:       # count the data source only
            with lock:
                n["elems"] += int(np.asarray(out).size)
        return out

    monkeypatch.setattr(DynamicArray, "_read_direct", rd)
    return n


@pytest.fixture(autouse=True)
def _fresh():
    io.clear_cache()
    yield
    io.clear_cache()


def _da(x, chunks=(3, 4, 5)):
    return DynamicArray(zarr.array(x, chunks=chunks))


F32 = (np.random.default_rng(0).random((6, 8, 10)) * 255).astype("f4")
U16 = (np.random.default_rng(1).random((6, 8, 10)) * 4000).astype("u2")
CASES = [
    ("auto", dict(bins=64)),
    ("auto-1bin", dict(bins=1)),
    ("range", dict(bins=32, range=(0, 255))),
    ("range-narrow", dict(bins=7, range=(10.5, 20.25))),
    ("edges", dict(bins=np.linspace(0, 255, 40))),
    ("edges-uneven", dict(bins=np.array([0, 1, 5, 50, 51, 200, 4000]))),
]


@pytest.mark.parametrize("data", [F32, U16], ids=["float32", "uint16"])
@pytest.mark.parametrize("name, kw", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("strip", [64 * 1024 * 1024, 256], ids=["one-strip", "many-strips"])
def test_matches_numpy_exactly(data, name, kw, strip):
    c, e = ops.histogram(_da(data), strip_bytes=strip, **kw)
    assert isinstance(c, DynamicArray) and isinstance(e, DynamicArray)
    cn, en = np.histogram(data, **kw)
    assert c.shape == cn.shape and e.shape == en.shape
    np.testing.assert_array_equal(c.compute(), cn)
    np.testing.assert_array_equal(e.compute(), en)            # edges exact, incl. dtype
    assert e.dtype == en.dtype and c.dtype == np.int64


@pytest.mark.parametrize("x", [np.full((4, 5), 7.0, "f4"), np.full((4, 5), 3, "i2"),
                               np.zeros((3, 0), "f4")], ids=["constant-float", "constant-int", "empty"])
def test_numpy_edge_rules(x):
    # constant array: numpy widens the range to (v - 0.5, v + 0.5) (dyna used v .. v+1,
    # putting every count in a different bin); empty array: range (0, 1).
    c, e = ops.histogram(DynamicArray(zarr.array(x, chunks=(2, 2) if x.size else (3, 1))), bins=5)
    cn, en = np.histogram(x, bins=5)
    np.testing.assert_array_equal(c.compute(), cn)
    np.testing.assert_array_equal(e.compute(), en)


def test_non_finite_auto_range_raises_like_numpy():
    x = np.ones((4, 5), "f4")
    x[1, 2] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        np.histogram(x, bins=4)
    with pytest.raises(ValueError, match="not finite"):
        ops.histogram(_da(x, (2, 2)), bins=4)[0].compute()


def test_bad_arguments_raise_at_call_time():
    a = _da(F32)
    with pytest.raises(ValueError):
        ops.histogram(a, bins=0)
    with pytest.raises(ValueError):
        ops.histogram(a, bins=[3, 2, 1])
    with pytest.raises(ValueError):
        ops.histogram(a, bins=4, range=(5, 1))


def test_nothing_is_read_until_used(source_reads):
    a = _da(F32)
    for kw in (dict(bins=64), dict(bins=32, range=(0, 255)), dict(bins=np.linspace(0, 255, 9))):
        ops.histogram(a, **kw)
        a.histogram(**kw)
        np.histogram(a, **kw)
    assert source_reads["elems"] == 0


def test_pass_counts_and_cache(source_reads):
    a = _da(F32)
    c, e = ops.histogram(a, bins=64)
    c.compute()
    assert source_reads["elems"] == 2 * F32.size           # min+max fused, then counts
    source_reads["elems"] = 0
    c.compute(); e.compute()
    assert source_reads["elems"] == 0                      # cached
    io.clear_cache()
    c.compute()
    assert source_reads["elems"] == 2 * F32.size
    source_reads["elems"] = 0
    ops.histogram(a, bins=32, range=(0, 255))[0].compute()
    assert source_reads["elems"] == F32.size               # explicit range: one pass


def test_cache_is_stale_until_cleared(tmp_path):
    z = zarr.create_array(str(tmp_path / "a.zarr"), shape=F32.shape, chunks=(3, 4, 5), dtype="f4")
    z[...] = F32
    c, _ = ops.histogram(io.read(str(tmp_path / "a.zarr")), bins=8, range=(0, 512))
    first = c.compute().copy()
    z[...] = F32 + 256                                     # every value moves up 4 bins
    np.testing.assert_array_equal(c.compute(), first)
    c.clear_cache()
    np.testing.assert_array_equal(c.compute(), np.histogram(F32 + 256, bins=8, range=(0, 512))[0])


def test_numpy_dispatch_and_method():
    a = _da(F32)
    c, e = np.histogram(a, bins=16, range=(0, 255))
    assert isinstance(c, DynamicArray)
    np.testing.assert_array_equal(c.compute(), np.histogram(F32, bins=16, range=(0, 255))[0])
    c10, _ = np.histogram(a)                               # numpy's own default: 10 bins
    assert c10.shape == (10,)
    np.testing.assert_array_equal(np.asarray(a[2].histogram(8)[0]), np.histogram(F32[2], bins=8)[0])
    with pytest.raises(TypeError):
        np.histogram(a, bins=4, density=True)             # refused, not silently ignored
    with pytest.raises(TypeError):
        np.histogram(a, bins=4, weights=np.ones_like(F32))


def test_lazy_counts_compose_and_write(tmp_path):
    a = _da(F32)
    c, e = ops.histogram(a, bins=32)
    cn, en = np.histogram(F32, bins=32)
    np.testing.assert_allclose((c / c.sum()).compute(), cn / cn.sum())
    assert int(c.sum()) == F32.size
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(c, str(tmp_path / "counts.zarr"))
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "counts.zarr"))[...], cn)


def test_histogram_of_a_lazy_chain():
    a = _da(F32)
    c, _ = ops.histogram(ops.gaussian_filter(a, 1) * 2, bins=20)
    import scipy.ndimage as ndi
    np.testing.assert_array_equal(c.compute(), np.histogram(ndi.gaussian_filter(F32, 1) * 2, bins=20)[0])
