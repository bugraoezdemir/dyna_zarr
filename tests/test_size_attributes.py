"""DynamicArray.size / itemsize / nbytes match numpy and never read data."""
import threading

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations as ops


@pytest.fixture
def no_reads(monkeypatch):
    calls = []
    lock = threading.Lock()
    orig = DynamicArray._read_direct

    def rd(self, key):
        with lock:
            calls.append(key)
        return orig(self, key)

    monkeypatch.setattr(DynamicArray, "_read_direct", rd)
    yield
    assert calls == [], "size/itemsize/nbytes must not read data"


X = np.arange(6 * 7 * 5, dtype="f4").reshape(6, 7, 5)


@pytest.mark.parametrize("build, ref", [
    (lambda a: a, lambda x: x),
    (lambda a: a[1:4, ::2], lambda x: x[1:4, ::2]),
    (lambda a: a[2, 3], lambda x: x[2, 3]),
    (lambda a: a.astype("u1"), lambda x: x.astype("u1")),
    (lambda a: a.astype("f8") + 1, lambda x: x.astype("f8") + 1),
    (lambda a: ops.max(a, axis=0), lambda x: x.max(0)),
    (lambda a: a.mean(), lambda x: np.asarray(x.mean())),                    # 0-d
    (lambda a: ops.reshape(a, (42, 5)), lambda x: x.reshape(42, 5)),
    (lambda a: a[3:3], lambda x: x[3:3]),                                    # empty
    (lambda a: ops.histogram(a, bins=12)[0], lambda x: np.histogram(x, bins=12)[0]),
], ids=["source", "slice", "int-index", "astype-u1", "f8-chain", "reduce", "0-d", "reshape",
        "empty", "histogram"])
def test_match_numpy_without_reading(no_reads, build, ref):
    a = DynamicArray(zarr.array(X, chunks=(2, 3, 5)))
    r, e = build(a), ref(X)
    assert (r.size, r.itemsize, r.nbytes) == (e.size, e.itemsize, e.nbytes)
    assert isinstance(r.size, int) and isinstance(r.nbytes, int)


def test_generated_and_tensorstore_sources(no_reads, tmp_path):
    z = ops.zeros((300, 1024, 1024), dtype="u2")
    assert (z.size, z.itemsize, z.nbytes) == (300 * 1024 * 1024, 2, 300 * 1024 * 1024 * 2)
    zarr.create_array(str(tmp_path / "a.zarr"), shape=(4, 9), chunks=(2, 3), dtype="i8")
    t = io.read(str(tmp_path / "a.zarr"))                   # TensorStore-backed
    assert (t.size, t.itemsize, t.nbytes) == (36, 8, 288)
