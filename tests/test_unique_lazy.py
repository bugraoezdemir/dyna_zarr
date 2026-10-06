"""ops.unique is lazy until its shape is needed.

Its length depends on the data, so the DynamicArray stores a DEFERRED shape: building
it reads nothing, and the first shape or value need (.shape, .size, compute, a write,
building another op on it) runs ONE streaming pass that yields both the values and the
length. The result is cached like the small reductions, until io.clear_cache().
"""
import contextlib
import io as _stdio
import threading

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations as ops

LABELS = np.random.default_rng(0).integers(0, 7, (16, 32, 32)).astype("uint16")


@pytest.fixture
def source_reads(monkeypatch):
    """Counter of elements read from untransformed (source) arrays."""
    lock = threading.Lock()
    count = {"n": 0}
    orig = DynamicArray._read_direct

    def counting(self, key):
        out = orig(self, key)
        if self._transform is None:
            with lock:
                count["n"] += int(np.asarray(out).size)
        return out

    monkeypatch.setattr(DynamicArray, "_read_direct", counting)
    return count


@pytest.fixture(autouse=True)
def _fresh_cache():
    io.clear_cache()
    yield
    io.clear_cache()


def _src(tmp_path, data=LABELS):
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=data.shape, chunks=(4, 16, 16),
                          dtype=data.dtype)
    z[...] = data
    return io.read(str(tmp_path / "src.zarr"))


def test_building_reads_nothing(tmp_path, source_reads):
    u = ops.unique(_src(tmp_path))
    v = DynamicArray(u)                                  # a copy stays deferred too
    assert isinstance(u, DynamicArray) and isinstance(v, DynamicArray)
    assert u.dtype == LABELS.dtype and u.ndim == 1 and u.chunks is None
    assert source_reads["n"] == 0


@pytest.mark.parametrize("need", ["shape", "size", "nbytes", "compute", "asarray", "index"])
def test_first_need_is_one_pass_then_cached(tmp_path, source_reads, need):
    u = ops.unique(_src(tmp_path))
    first = {"shape": lambda: u.shape, "size": lambda: u.size, "nbytes": lambda: u.nbytes,
             "compute": u.compute, "asarray": lambda: np.asarray(u),
             "index": lambda: u[2:4].compute()}[need]()
    assert source_reads["n"] == LABELS.size
    assert u.shape == (7,)
    np.testing.assert_array_equal(u.compute(), np.unique(LABELS))
    np.testing.assert_array_equal(u[2:4].compute(), np.unique(LABELS)[2:4])
    assert source_reads["n"] == LABELS.size              # everything after: from the cache
    del first


def test_building_on_it_runs_the_pass_once(tmp_path, source_reads):
    u = ops.unique(_src(tmp_path))
    w = u * 2 + 1                                        # needs the shape: the pass runs here
    assert source_reads["n"] == LABELS.size
    np.testing.assert_array_equal(w.compute(), np.unique(LABELS) * 2 + 1)
    assert source_reads["n"] == LABELS.size


def test_write(tmp_path, source_reads):
    u = ops.unique(_src(tmp_path))
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(u, str(tmp_path / "u.zarr"))
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "u.zarr"))[...],
                                  np.unique(LABELS))
    assert source_reads["n"] == LABELS.size


def test_computed_once_under_concurrency(tmp_path, source_reads):
    u = ops.unique(_src(tmp_path))
    barrier = threading.Barrier(8)
    shapes = []

    def worker():
        barrier.wait()
        shapes.append(u.shape)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(shapes) == {(7,)}
    assert source_reads["n"] == LABELS.size


def test_cached_until_cleared(tmp_path):
    u = ops.unique(_src(tmp_path))
    assert u.shape == (7,)
    zarr.open_array(str(tmp_path / "src.zarr"))[0, 0, 0] = 99
    assert u.shape == (7,)                               # cached: the data is assumed fixed
    io.clear_cache()
    assert u.shape == (8,)
    assert int(u.compute()[-1]) == 99


@pytest.mark.parametrize("data", [
    np.array([3.0, np.nan, 1.0, np.nan, 3.0, -0.0, 0.0], "f4").reshape(1, 7),
    np.zeros((0, 4), "i4"),
    np.full((5, 5), 42, "u1"),
], ids=["floats-nan", "empty", "constant"])
def test_matches_numpy(data):
    u = ops.unique(DynamicArray(data))
    np.testing.assert_array_equal(u.compute(), np.unique(data))
    assert u.shape == np.unique(data).shape and u.dtype == data.dtype


def test_numpy_dispatch_is_lazy_and_refuses_what_it_cannot_do(tmp_path, source_reads):
    a = _src(tmp_path)
    u = np.unique(a)
    assert isinstance(u, DynamicArray) and source_reads["n"] == 0
    np.testing.assert_array_equal(u.compute(), np.unique(LABELS))
    for kw in (dict(return_counts=True), dict(return_index=True), dict(return_inverse=True),
               dict(axis=0), dict(equal_nan=False)):
        with pytest.raises(TypeError):                   # refused, not ignored
            np.unique(a, **kw)
