"""
Streaming reindex writer for reshape/flatten: output-chunk driven, one input chunk at a
time. Correct on any chunking (incl. non-cubic + partial chunks), hard-bounded memory,
race-free. io.write auto-routes reshape/flatten (outermost) to it.
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations as ops
from dyna_zarr.reindex import reindex_write


@pytest.fixture
def da_arr():
    arr = np.random.default_rng(0).random((12, 10, 8, 6)).astype(np.float32)
    return DynamicArray(zarr.array(arr, chunks=(5, 4, 3, 4))), arr    # non-cubic, partial


@pytest.mark.parametrize("out_shape,out_chunks", [
    ((120, 48), (7, 9)),           # 4D -> 2D
    ((24, 10, 24), (7, 4, 10)),    # 4D -> 3D
    ((6, 20, 8, 6), (4, 7, 5, 4)), # 4D -> 4D
    ((12 * 10 * 8 * 6,), (100,)),  # flatten
])
def test_reindex_write_matches_numpy(da_arr, out_shape, out_chunks, tmp_path):
    da, arr = da_arr
    out = str(tmp_path / "o.zarr")
    reindex_write(da, out_shape, out, out_chunks)
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr.reshape(out_shape))


def test_lru_cache_same_result(da_arr, tmp_path):
    da, arr = da_arr
    a = str(tmp_path / "a.zarr")
    b = str(tmp_path / "b.zarr")
    reindex_write(da, (120, 48), a, (7, 9), cache_size=0)
    reindex_write(da, (120, 48), b, (7, 9), cache_size=8)
    np.testing.assert_array_equal(zarr.open(a, mode="r")[:], zarr.open(b, mode="r")[:])
    np.testing.assert_array_equal(zarr.open(a, mode="r")[:], arr.reshape(120, 48))


def test_iowrite_routes_reshape_flatten(da_arr, tmp_path):
    da, arr = da_arr
    r = str(tmp_path / "r.zarr")
    f = str(tmp_path / "f.zarr")
    io.write(ops.reshape(da, (80, 72)), r, zarr_format=2, chunks=(16, 3))
    io.write(ops.flatten(da), f, zarr_format=2, chunks=(64,))
    np.testing.assert_array_equal(zarr.open(r, mode="r")[:], arr.reshape(80, 72))
    np.testing.assert_array_equal(zarr.open(f, mode="r")[:], arr.ravel())


def test_iowrite_nested_reshape_uses_normal_path(da_arr, tmp_path):
    """reshape NOT outermost (abs on top) -> normal region writer, still correct."""
    da, arr = da_arr
    out = str(tmp_path / "ra.zarr")
    io.write(ops.abs(ops.reshape(da, (80, 72))), out, zarr_format=2, chunks=(16, 3))
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], np.abs(arr).reshape(80, 72))
