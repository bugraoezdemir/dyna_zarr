"""Dask arrays over readers that must not be read from several threads at once.

bioio's ND2 reader returns a ``resource_backed_dask_array.ResourceBackedDaskArray`` (a dask
subclass): its ``compute()`` opens the file and closes it afterwards, so concurrent computes
close each other's handle ("Attempt to read from closed nd2 file") or read wrong pixels.
dyna recognises the type and serialises its reads, in io.write and in from_array. It used to
miss dask SUBCLASSES altogether (the type check looked at the module name), so io.write
crashed on them with an unrelated TypeError.
"""
import threading
import time

import numpy as np
import pytest
import zarr

da = pytest.importorskip("dask.array")

import dyna_zarr  # noqa: E402
from dyna_zarr.dynamic_array import _is_dask, _needs_read_lock  # noqa: E402
from dyna_zarr.io import io  # noqa: E402

DATA = np.arange(16 * 32 * 32, dtype="uint16").reshape(16, 32, 32)


class ConcurrencyDetector:
    """A reader that fails if two reads overlap in time (as a shared file handle would)."""

    def __init__(self):
        self._busy = threading.Lock()
        self.reads = 0

    def read(self, i):
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("concurrent read of a non-thread-safe handle")
        try:
            time.sleep(0.01)                 # widen the window a race would hit
            self.reads += 1
            return DATA[i:i + 1]
        finally:
            self._busy.release()


def _graph(reader):
    import dask
    blocks = [da.from_delayed(dask.delayed(reader.read)(i), shape=(1, 32, 32),
                              dtype=DATA.dtype) for i in range(DATA.shape[0])]
    return da.concatenate(blocks, axis=0)


# a stand-in for resource_backed_dask_array.ResourceBackedDaskArray (same module name)
FakeResourceBacked = type("ResourceBackedDaskArray", (da.Array,),
                          {"__module__": "resource_backed_dask_array"})


def _resource_backed(reader):
    x = _graph(reader)
    return FakeResourceBacked(x.dask, x.name, x.chunks, dtype=x.dtype, meta=x._meta)


def test_detection():
    plain = _graph(ConcurrencyDetector())
    rb = _resource_backed(ConcurrencyDetector())
    assert _is_dask(plain) and _is_dask(rb)          # subclasses are dask arrays too
    assert not _needs_read_lock(plain)
    assert _needs_read_lock(rb)


def test_the_detector_sees_concurrency(tmp_path):
    """Sanity: without serialisation the parallel pump does hit the race."""
    with pytest.raises(RuntimeError, match="concurrent read"):
        io.write(_graph(ConcurrencyDetector()), str(tmp_path / "o.zarr"), max_threads=8)


def test_io_write_serialises_a_resource_backed_array(tmp_path, capsys):
    reader = ConcurrencyDetector()
    io.write(_resource_backed(reader), str(tmp_path / "o.zarr"), chunks=(1, 32, 32),
             max_threads=8)
    assert "reads through one shared file handle" in capsys.readouterr().out
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "o.zarr"))[...], DATA)
    assert reader.reads == DATA.shape[0]              # each block read once


def test_from_array_locks_a_resource_backed_array_by_default(tmp_path):
    arr = dyna_zarr.from_array(_resource_backed(ConcurrencyDetector()))
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(1, 32, 32), max_threads=8)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "o.zarr"))[...], DATA)


def test_lock_is_allowed_for_any_dask_array(tmp_path):
    """A dask graph can wrap any reader: the caller who knows it is unsafe can say so
    (arithmetic or stacking drops the resource-backed type)."""
    derived = _graph(ConcurrencyDetector()) + 0
    arr = dyna_zarr.from_array(derived, lock=True)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(1, 32, 32), max_threads=8)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "o.zarr"))[...], DATA)


def test_lock_false_opts_out():
    arr = dyna_zarr.from_array(_resource_backed(ConcurrencyDetector()), lock=False)
    assert arr._source._lock is None


def test_lock_still_refused_for_safe_sources():
    with pytest.raises(ValueError, match="safe to read from several threads"):
        dyna_zarr.from_array(np.zeros(4), lock=True)
