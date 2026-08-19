"""What `DynamicArray` accepts as a source, and what it refuses.

zarr and numpy are wrappable; **dask is refused loudly**. A numpy source is staged through
an in-memory zarr array rather than wrapped raw, so it arrives with a real chunk grid -
chunks are load-bearing downstream (tile defaults snap to them, and the writer infers its
region from them), and a raw wrap left `.chunks` as None. Wrapping a dask array half-works,
which is worse than not working: most ops still succeed (scipy calls `np.asarray` on the
block internally) while `np.asarray(arr[1:3])` raises an opaque
"object __array__ method not producing an array", and `.chunks` reports dask's
tuple-of-tuples where every consumer expects a per-axis shape.
"""

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray


@pytest.fixture
def data():
    return (np.arange(4 * 8 * 8) % 251).reshape(4, 8, 8).astype("int32")


def test_accepts_numpy(data):
    arr = DynamicArray(data)
    assert arr.shape == data.shape
    assert arr.dtype == data.dtype
    np.testing.assert_array_equal(np.asarray(arr), data)


def test_numpy_gets_a_chunk_grid(data):
    """Staging through an in-memory zarr means numpy sources have real chunks."""
    arr = DynamicArray(data)
    assert arr.chunks is not None
    assert len(arr.chunks) == data.ndim
    assert all(isinstance(c, int) and c > 0 for c in arr.chunks)


def test_numpy_explicit_chunks(data):
    assert DynamicArray(data, chunks=(2, 4, 4)).chunks == (2, 4, 4)


def test_numpy_chunks_are_bounded():
    """The auto grid targets a chunk size, so a big array is not one giant chunk."""
    big = np.zeros((256, 256, 256), dtype="int32")
    arr = DynamicArray(big)
    assert int(np.prod(arr.chunks)) < int(np.prod(big.shape))


def test_accepts_numpy_slices(data):
    """The case that breaks for dask must work for numpy."""
    arr = DynamicArray(data)
    np.testing.assert_array_equal(np.asarray(arr[1:3]), data[1:3])


def test_accepts_zarr(tmp_path, data):
    p = tmp_path / "z.zarr"
    z = zarr.create_array(store=str(p), shape=data.shape, chunks=(2, 4, 4), dtype=data.dtype)
    z[:] = data
    arr = DynamicArray(z)
    assert arr.chunks == (2, 4, 4)          # per-axis shape, not dask's tuple-of-tuples
    np.testing.assert_array_equal(np.asarray(arr), data)


def test_accepts_dynamicarray(data):
    np.testing.assert_array_equal(np.asarray(DynamicArray(DynamicArray(data))), data)


def test_rejects_dask(data):
    da = pytest.importorskip("dask.array")
    with pytest.raises(TypeError, match="cannot wrap"):
        DynamicArray(da.from_array(data, chunks=(2, 4, 4)))


def test_dask_rejection_names_the_alternative(data):
    da = pytest.importorskip("dask.array")
    with pytest.raises(TypeError) as exc:
        DynamicArray(da.from_array(data, chunks=(2, 4, 4)))
    assert "dask backend" in str(exc.value)
