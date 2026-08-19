"""`np.asarray(a)` must materialize a DynamicArray, not wrap it.

Without `__array__`, NumPy treats a DynamicArray as an opaque object and builds a **0-d
object array**: `np.asarray(a[10:20]).shape` was `()` and downstream code silently computed
nonsense instead of raising. `.compute()` always worked; the point of `__array__` is that
the ordinary NumPy spelling works too, on both backends.
"""

import numpy as np
import pytest
import zarr

from dyna_zarr import operations as ops
from dyna_zarr.io import io


@pytest.fixture
def src(tmp_path):
    data = (np.arange(32 * 32 * 32) % 251).reshape(32, 32, 32).astype("uint8")
    p = tmp_path / "a.zarr"
    z = zarr.create_array(store=str(p), shape=data.shape, chunks=(8, 8, 8), dtype="uint8")
    z[:] = data
    return io.read(str(p)), data


def test_asarray_materializes_full(src):
    arr, data = src
    np.testing.assert_array_equal(np.asarray(arr), data)


def test_asarray_materializes_slice(src):
    """The original bug: a sliced DynamicArray came back as a 0-d object array."""
    arr, data = src
    sl = (slice(10, 20), slice(5, 15), slice(0, 8))
    out = np.asarray(arr[sl])
    assert out.shape == (10, 10, 8)
    np.testing.assert_array_equal(out, data[sl])


def test_np_array_also_works(src):
    arr, data = src
    np.testing.assert_array_equal(np.array(arr[2:6]), data[2:6])


def test_dtype_argument(src):
    arr, _ = src
    assert np.asarray(arr[:4], dtype=np.int32).dtype == np.dtype(np.int32)


def test_copy_false_raises(src):
    """The data does not exist until read, so a no-copy view is impossible."""
    arr, _ = src
    with pytest.raises(ValueError, match="without copying"):
        np.array(arr[:4], copy=False)


def test_transform_chain_materializes(src):
    arr, data = src
    np.testing.assert_array_equal(np.asarray(ops.abs(arr[:4]) + 1), data[:4] + 1)


def test_ops_stay_lazy(src):
    """__array__ must not make the op surface eager."""
    arr, _ = src
    assert type(ops.abs(arr)).__name__ == "DynamicArray"
    assert type(arr + 1).__name__ == "DynamicArray"
    assert type(np.sqrt(arr.astype("float32"))).__name__ == "DynamicArray"


def test_asarray_matches_compute(src):
    arr, _ = src
    sl = (slice(3, 9), slice(1, 7), slice(2, 5))
    np.testing.assert_array_equal(np.asarray(arr[sl]), arr[sl].compute())
