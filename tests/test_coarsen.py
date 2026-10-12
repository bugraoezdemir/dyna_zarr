"""operations.coarsen: bit-identical to tensorstore.downsample, lazy, memory-budgeted.

The reference is TensorStore itself, so every rule it has (ceil-sized levels with partial
edge windows, integer means rounded half to even, the lower median, mode ties to the
smallest value, min/max ignoring NaN, the float-mean summation order) is pinned without
restating it here. The one deliberate difference - NaN in a float median/mode window,
where TensorStore depends on element order - has its own test.
"""
import itertools

import numpy as np
import pytest
import tensorstore as ts
import zarr

import dyna_zarr
from dyna_zarr import operations as ops
from dyna_zarr.io import io

METHODS = ["mean", "median", "min", "max", "mode"]
DTYPES = ["uint8", "int8", "uint16", "int16", "int32", "uint64", "int64",
          "float16", "float32", "float64", "bool"]
CASES = [((7,), (2,)), ((9, 10), (2, 3)), ((5, 11, 13), (2, 2, 2)), ((4, 6, 9), (1, 4, 3)),
         ((3, 3), (4, 4)), ((20, 20, 20), (4, 4, 4)), ((30, 30), (9, 9))]


def _ts(a, f, m):
    return ts.downsample(ts.array(a), list(f), method=m).read().result()


def _data(dtype, shape, rng, ties=False):
    if dtype == "bool":
        return rng.integers(0, 2, shape).astype(bool)
    if dtype.startswith("float"):
        a = rng.normal(100, 50, shape)
        return (np.round(a / 40) if ties else a).astype(dtype)
    info = np.iinfo(dtype)
    a = rng.integers(max(int(info.min), -1000), min(int(info.max), 1000), shape)
    return (a % 4 if ties else a).astype(dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("method", METHODS)
def test_matches_tensorstore(dtype, method):
    rng = np.random.default_rng(0)
    for (shape, f), ties in itertools.product(CASES, (False, True)):
        a = _data(dtype, shape, rng, ties)
        got = np.asarray(ops.coarsen(dyna_zarr.from_array(
            a, chunks=tuple(max(1, s // 2) for s in shape)), f, method))
        want = _ts(a, f, method)
        assert got.dtype == want.dtype
        np.testing.assert_array_equal(got, want, err_msg=f"{shape} {f} ties={ties}")


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_64bit_mean_is_exact_at_the_extremes(dtype):
    """The window sum of 64-bit values overflows 64 bits; the mean must not."""
    info = np.iinfo(dtype)
    a = np.full((6, 6), info.max, dtype)
    a[::2, ::3] = info.min
    np.testing.assert_array_equal(np.asarray(ops.coarsen(dyna_zarr.from_array(a), (3, 2),
                                                         "mean")), _ts(a, (3, 2), "mean"))


def test_nan_min_max_mean_match_tensorstore():
    a = np.array([np.nan, np.nan, 1, np.nan, 2, 3], "float32")
    for m in ("min", "max", "mean"):
        np.testing.assert_array_equal(np.asarray(ops.coarsen(dyna_zarr.from_array(a), (2,),
                                                             m)), _ts(a, (2,), m))


def test_nan_median_and_mode_ignore_nan():
    """dyna's own rule (TensorStore's result depends on element order here)."""
    a = np.array([np.nan, 5, 1, 9,   np.nan, np.nan, np.nan, np.nan,   2, 2, np.nan, 7],
                 "float32")
    med = np.asarray(ops.coarsen(dyna_zarr.from_array(a), (4,), "median"))
    np.testing.assert_array_equal(med, [5, np.nan, 2])            # lower middle of non-NaN
    mode = np.asarray(ops.coarsen(dyna_zarr.from_array(a), (4,), "mode"))
    np.testing.assert_array_equal(mode, [1, np.nan, 2])


@pytest.mark.parametrize("chunks", [(1, 16, 64, 64), (2, 7, 33, 50), (3, 67, 301, 263)])
def test_chunk_fused_reads_and_sliced_keys(chunks):
    """Reads are cut on the source grid (odd grids too); slicing the result is exact."""
    rng = np.random.default_rng(5)
    a = rng.integers(0, 50, (3, 67, 301, 263)).astype("uint16")
    x = dyna_zarr.from_array(a, chunks=chunks)
    for f, m in [((1, 2, 2, 2), "mean"), ((1, 3, 3, 3), "median"), ((2, 1, 4, 4), "mode")]:
        want = _ts(a, f, m)
        c = ops.coarsen(x, f, m)
        assert c.shape == want.shape
        for key in [(slice(None),) * 4, (slice(1, 3), slice(5, 30, 2), 7, slice(10, 60))]:
            np.testing.assert_array_equal(np.asarray(c[key]), want[key])


def test_block_function_on_dask():
    """coarsen_block is the dask path too (window-aligned blocks via map_blocks)."""
    da = pytest.importorskip("dask.array")
    a = np.random.default_rng(1).integers(0, 9, (21, 43), dtype="uint16")
    x = da.from_array(a, chunks=(6, 10)).rechunk((6, 10))
    for m in METHODS:
        out_chunks = tuple(tuple(-(-c // f) for c in cs) for cs, f in zip(x.chunks, (3, 2)))
        y = x.map_blocks(ops.coarsen_block, (3, 2), m, chunks=out_chunks, dtype=x.dtype)
        np.testing.assert_array_equal(y.compute(), _ts(a, (3, 2), m))


def test_validation():
    x = dyna_zarr.from_array(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="method"):
        ops.coarsen(x, (2, 2), "gauss")
    with pytest.raises(ValueError, match="one entry per axis"):
        ops.coarsen(x, (2,), "mean")
    with pytest.raises(ValueError, match=">= 1"):
        ops.coarsen(x, (0, 2), "mean")
    assert ops.coarsen(x, (1, 1), "mean") is x


def test_written_and_budgeted_by_what_it_reads(tmp_path, capsys):
    """io.write shapes a coarsen region in OUTPUT elements but budgets it by what it
    READS (prod(factors) per output element): a worker is 2 x read + out."""
    a = np.random.default_rng(2).integers(0, 1000, (64, 64, 64), dtype="uint16")
    z = zarr.create_array(str(tmp_path / "s.zarr"), shape=a.shape, chunks=(16, 16, 16),
                          dtype=a.dtype)
    z[...] = a
    c = ops.coarsen(io.read(str(tmp_path / "s.zarr")), (2, 2, 2), "mean")
    # region 8^3 uint16 = 1 KiB out, reads 16^3 = 8 KiB in -> worker = 2*8 + 1 = 17 KiB
    # (counted as 1 KiB without the amplification, the budget would allow all 16)
    io.write(c, str(tmp_path / "o.zarr"), chunks=(8, 8, 8), region_shape=(8, 8, 8),
             max_threads=16, memory_budget_mb=3 * 17 / 1024)
    line = [l for l in capsys.readouterr().out.splitlines() if "Engine: fused" in l][0]
    assert "fused, 3 threads" in line
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "o.zarr"))[...],
                                  _ts(a, (2, 2, 2), "mean"))
