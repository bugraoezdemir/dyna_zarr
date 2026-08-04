"""
Prefix-scan ops (cumsum/cumprod/cummax/cummin). A scan is not chunk-local: output[..p..]
depends on the whole prefix along the scan axis, so read(key) reads [0, stop) on that axis.
The sub-slice tests are the real check that the prefix math + squeeze are correct and
chunk-invariant.
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, operations as ops

SHAPE = (5, 7, 6)
REF = {
    "cumsum": lambda a, ax: np.cumsum(a, axis=ax),
    "cumprod": lambda a, ax: np.cumprod(a, axis=ax),
    "cummax": lambda a, ax: np.maximum.accumulate(a, axis=ax),
    "cummin": lambda a, ax: np.minimum.accumulate(a, axis=ax),
}
OP = {"cumsum": ops.cumsum, "cumprod": ops.cumprod, "cummax": ops.cummax, "cummin": ops.cummin}


@pytest.fixture
def data():
    arr = (np.random.default_rng(0).random(SHAPE).astype(np.float32) * 4) - 2
    return arr, DynamicArray(zarr.array(arr, chunks=(2, 3, 4)))


@pytest.mark.parametrize("name", list(OP))
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_full_compute_matches_numpy(data, name, axis):
    arr, d = data
    np.testing.assert_allclose(OP[name](d, axis).compute(), REF[name](arr, axis % 3), atol=1e-4)


@pytest.mark.parametrize("name", list(OP))
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("key", [
    np.s_[1:4, 2:6, :], np.s_[2, :, 1:5], np.s_[::2, 1:6:2, 3], np.s_[:, 0, :], np.s_[4, 6, 5],
])
def test_subslice_chunk_invariant(data, name, axis, key):
    """op(d, axis)[key] == numpy(arr, axis)[key] -- prefix read + squeeze, any chunking."""
    arr, d = data
    np.testing.assert_allclose(OP[name](d, axis)[key].compute(), REF[name](arr, axis)[key], atol=1e-4)


def test_int_dtype_matches_numpy(data):
    arr, _ = data
    di = DynamicArray(zarr.array(arr.astype("int16"), chunks=(2, 3, 4)))
    assert ops.cumsum(di, 0).dtype == np.cumsum(arr.astype("int16"), axis=0).dtype


@pytest.mark.parametrize("name", list(OP))
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_scan_write_streams_correctly(data, name, axis, tmp_path):
    """io.write routes an outermost scan to the bounded-carry streaming writer; a tiny
    region budget forces multiple cross-section tiles AND multiple scan-axis strips."""
    from dyna_zarr import io
    arr, d = data
    out = str(tmp_path / f"{name}{axis}.zarr")
    io.write(OP[name](d, axis), out, zarr_format=2, chunks=(2, 3, 4), region_size_mb=0.001)
    np.testing.assert_allclose(zarr.open(out, mode="r")[:], REF[name](arr, axis), atol=1e-4)


def test_scan_write_multi_strip_matches(tmp_path):
    """A tiny budget forces many cross-section tiles AND many scan-axis strips, so the running
    carry is exercised across strip boundaries (where a naive per-block scan would be wrong)."""
    from dyna_zarr import io
    rng = np.random.default_rng(1)
    arr = rng.random((40, 12, 10)).astype("float32")
    d = DynamicArray(zarr.array(arr, chunks=(8, 5, 4)))
    out = str(tmp_path / "cs.zarr")
    io.write(ops.cumsum(d, 0), out, zarr_format=2, chunks=(8, 5, 4), region_size_mb=0.002)
    np.testing.assert_allclose(zarr.open(out, mode="r")[:], np.cumsum(arr, axis=0), atol=1e-3)


def test_axis_out_of_range(data):
    _, d = data
    with pytest.raises(ValueError):
        ops.cumsum(d, 3)
