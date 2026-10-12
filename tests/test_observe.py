"""io.write(observe=) / io.Extrema: statistics in the same pass as the write.

``observe(key, block)`` sees every block exactly as written (after the cast) with its
location; io.Extrema turns that into per-channel min/max, so a caller that wants the
written data's range (OMERO windows) does not read the output a second time.
"""
import threading

import numpy as np
import pytest
import zarr

import dyna_zarr
from dyna_zarr import operations as ops
from dyna_zarr.io import io

SHAPE = (3, 4, 40, 50)                      # (c, z, y, x): 3 channels


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(3)
    d = rng.integers(0, 60000, size=SHAPE, dtype="uint16")
    d[1] //= 7                              # channels with different ranges
    d[2] += 1000
    return d


@pytest.fixture
def source(data, tmp_path):
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=(1, 2, 16, 16),
                          dtype=data.dtype)
    z[...] = data
    return io.read(str(tmp_path / "src.zarr"))


def _expect(a, axis=0):
    axes = tuple(i for i in range(a.ndim) if i != axis)
    return a.min(axis=axes).tolist(), a.max(axis=axes).tolist()


@pytest.mark.parametrize("engine", ["fused", "pipeline"])
def test_per_channel_extrema_in_the_write(source, data, tmp_path, engine):
    ext = io.Extrema(axis=0)
    io.write(source, str(tmp_path / "o.zarr"), engine=engine, region_size_mb=0.1, observe=ext)
    assert (ext.min, ext.max) == _expect(data)
    assert all(isinstance(v, int) for v in ext.min + ext.max)      # python scalars


def test_observe_sees_every_block_once_with_its_location(source, data, tmp_path):
    seen, lock = [], threading.Lock()

    def observe(key, block):
        with lock:
            seen.append((key, block.copy()))
    io.write(source, str(tmp_path / "o.zarr"), region_size_mb=0.1, observe=observe)
    rebuilt = np.zeros_like(data)
    covered = np.zeros(SHAPE, dtype=int)
    for key, block in seen:
        rebuilt[key] = block
        covered[key] += 1
    assert (covered == 1).all()
    np.testing.assert_array_equal(rebuilt, data)


def test_values_are_the_written_ones_after_the_cast(tmp_path):
    x = np.array([[-5.6, 0.4, 3.5], [300.0, 12.0, 7.0]], dtype="float32")
    ext = io.Extrema(axis=0)
    io.write(dyna_zarr.from_array(x), str(tmp_path / "o.zarr"), dtype="uint8", observe=ext)
    assert ext.min == [0, 7] and ext.max == [4, 255]      # saturated + rounded half-even


def test_whole_array_extrema_and_nan(tmp_path):
    x = np.arange(24, dtype="float32").reshape(4, 6)
    ext = io.Extrema()
    io.write(dyna_zarr.from_array(x), str(tmp_path / "a.zarr"), observe=ext)
    assert (ext.min, ext.max) == ([0.0], [23.0])
    x[2, 3] = np.nan
    ext = io.Extrema(axis=0)
    io.write(dyna_zarr.from_array(x), str(tmp_path / "b.zarr"), observe=ext)
    assert np.isnan(ext.min[2]) and np.isnan(ext.max[2]) and ext.min[3] == 18.0


def test_dask_arrays(data, tmp_path):
    da = pytest.importorskip("dask.array")
    ext = io.Extrema(axis=0)
    io.write(da.from_array(data, chunks=(1, 2, 20, 25)), str(tmp_path / "o.zarr"),
             observe=ext, dtype="uint8")
    assert (ext.min, ext.max) == _expect(np.clip(data, 0, 255).astype("uint8"))


def test_write_levels_observes_the_first_output(source, data, tmp_path):
    ext = io.Extrema(axis=0)
    outs = [dict(path=str(tmp_path / f"L{i}.zarr"), factors=(1, 2 ** i, 2 ** i, 2 ** i))
            for i in range(3)]
    io.write_levels(source, outs, method="mean", observe=ext)
    assert (ext.min, ext.max) == _expect(data)


def test_staged_writes_refuse_observe(source, tmp_path):
    with pytest.raises(ValueError, match="observe"):
        io.write(ops.reshape(source, (3, 8000)), str(tmp_path / "o.zarr"),
                 observe=io.Extrema())
    assert not (tmp_path / "o.zarr").exists()
