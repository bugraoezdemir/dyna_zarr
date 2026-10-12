"""io.write's two execution engines: ``"fused"`` (default) and ``"pipeline"``.

Everything before the execution loop (output, regions, bands) is shared, so the two
engines must write BIT-IDENTICAL output for every kind of chain. The fused engine's own
contract - the first error stops the write and is re-raised, the early timeouts fire - is
tested on a duck source wrapped with ``from_array``.
"""
import threading
import time

import numpy as np
import pytest
import zarr

import dyna_zarr
from dyna_zarr import operations as ops
from dyna_zarr.backends.zarrista_backend import ZARRISTA_AVAILABLE
from dyna_zarr.io import io

ENGINES = ("fused", "pipeline")


@pytest.fixture
def src(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, (24, 40, 48), dtype="uint16")
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=data.shape, chunks=(8, 8, 16),
                          dtype=data.dtype)
    z[...] = data
    return data, io.read(str(tmp_path / "src.zarr"))


CASES = {
    "copy": lambda a: a,
    "add1": lambda a: a + 1,
    "narrow_uint8": lambda a: (a // 16).astype("uint8"),
    "gauss_thresh": lambda a: (ops.gaussian_filter(a.astype("float32"), 1.5) > 2000)
                              .astype("uint8"),
    "mip": lambda a: a.max(axis=0),
    "bcast_minus_mean": lambda a: a.astype("float32") - a.mean(axis=0, keepdims=True),
    "concat": lambda a: ops.concatenate([a, a[:, :13]], axis=1),
    "sliced": lambda a: a[3:21, 5:37:2, ::3],
}


def _write_both(tmp_path, lazy, **kw):
    out = {}
    for engine in ENGINES:
        path = tmp_path / f"{engine}.zarr"
        io.write(lazy, str(path), engine=engine, max_threads=4, **kw)
        out[engine] = zarr.open_array(str(path))[...]
    return out


@pytest.mark.parametrize("case", sorted(CASES))
def test_engines_write_identical_output(tmp_path, src, case):
    data, a = src
    lazy = CASES[case](a)
    out = _write_both(tmp_path, lazy, region_size_mb=0.02)
    assert out["fused"].dtype == out["pipeline"].dtype
    np.testing.assert_array_equal(out["fused"], out["pipeline"])
    np.testing.assert_array_equal(out["fused"], np.asarray(lazy))


def test_engines_identical_with_explicit_region_shape(tmp_path, src):
    _, a = src
    out = _write_both(tmp_path, a + 7, chunks=(8, 8, 16), region_shape=(8, 16, 16))
    np.testing.assert_array_equal(out["fused"], out["pipeline"])


@pytest.mark.skipif(not ZARRISTA_AVAILABLE, reason="zarrista is not installed")
def test_engines_identical_sharded_zarrista(tmp_path, src):
    data, a = src
    out = _write_both(tmp_path, a + 1, backend="zarrista", chunks=(8, 8, 16),
                      shard_coefficients=(1, 2, 3))
    np.testing.assert_array_equal(out["fused"], out["pipeline"])
    np.testing.assert_array_equal(out["fused"], data + 1)


# --- fused-specific contract ----------------------------------------------------------

class _Duck:
    """A minimal array: reads sleep ``delay`` seconds; a read touching row ``fail_at``
    raises. Counts the reads that ran."""

    def __init__(self, shape=(64, 32), delay=0.0, fail_at=None):
        self.shape, self.dtype, self.ndim = shape, np.dtype("float32"), len(shape)
        self.delay, self.fail_at = delay, fail_at
        self.reads = 0
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            self.reads += 1
        if self.delay:
            time.sleep(self.delay)
        rows = key[0]
        if self.fail_at is not None and rows.start <= self.fail_at < rows.stop:
            raise RuntimeError("boom at row %d" % self.fail_at)
        return np.ones(tuple(len(range(*k.indices(n))) for k, n in zip(key, self.shape)),
                       dtype=self.dtype)


def test_fused_first_error_is_raised_and_stops_the_write(tmp_path):
    duck = _Duck(fail_at=8)
    arr = dyna_zarr.from_array(duck, chunks=(4, 32))
    with pytest.raises(RuntimeError, match="boom at row 8"):
        io.write(arr, str(tmp_path / "o.zarr"), chunks=(4, 32), region_shape=(4, 32),
                 max_threads=1)
    # one worker, in order: rows 0-3, 4-7, then the failing 8-11 - nothing after it
    assert duck.reads == 3


def test_fused_early_tenth_timeout(tmp_path):
    arr = dyna_zarr.from_array(_Duck(delay=0.05), chunks=(4, 32))
    with pytest.raises(TimeoutError, match="Early tenth timeout"):
        io.write(arr, str(tmp_path / "o.zarr"), chunks=(4, 32), region_shape=(4, 32),
                 max_threads=1, early_tenth_timeout=0.01)


def test_fused_generous_timeouts_do_not_fire(tmp_path):
    arr = dyna_zarr.from_array(_Duck(), chunks=(4, 32))
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(4, 32), region_shape=(4, 32),
             max_threads=2, early_tenth_timeout=60, early_quarter_timeout=60)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "o.zarr"))[...],
                                  np.ones((64, 32), "float32"))
