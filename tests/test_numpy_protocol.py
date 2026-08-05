"""
NumPy-protocol surface on DynamicArray: array METHODS (.astype/.clip/.round) and the ufunc
protocol (np.sqrt(a), np.add(a, 2), np.clip(a, lo, hi)). These make a DynamicArray a drop-in
for numpy/dask arrays (used by the ome_zarr_pro backend). Distinct from operations.* function
fuzzing (test_transform_correctness) -- e.g. np.clip goes through numpy's _wrapfunc calling
a.clip(min, max, out=...), NOT the ufunc protocol, which once silently returned wrong data.
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray


@pytest.fixture
def da_arr():
    arr = (np.random.default_rng(0).random((3, 8, 8)).astype(np.float32) * 10) - 5
    return DynamicArray(zarr.array(arr, chunks=(1, 4, 4))), arr     # multi-chunk -> block reads


# --- array methods ---

def test_astype_method(da_arr):
    d, arr = da_arr
    np.testing.assert_array_equal(d.astype("int16").compute(), arr.astype("int16"))


def test_clip_method(da_arr):
    d, arr = da_arr
    np.testing.assert_allclose(d.clip(-1, 2).compute(), np.clip(arr, -1, 2))
    np.testing.assert_allclose(d.clip(0, None).compute(), np.clip(arr, 0, None))


def test_round_method(da_arr):
    d, arr = da_arr
    np.testing.assert_allclose(d.round().compute(), np.round(arr))
    np.testing.assert_allclose(d.round(1).compute(), np.round(arr, 1))


def test_clip_round_reject_out(da_arr):
    d, _ = da_arr
    with pytest.raises(TypeError):
        d.clip(0, 1, out=np.empty(d.shape, dtype=d.dtype))
    with pytest.raises(TypeError):
        d.round(0, out=np.empty(d.shape, dtype=d.dtype))


# --- numpy top-level functions that dispatch onto the array (the np.clip _wrapfunc path) ---

def test_np_clip_and_round(da_arr):
    d, arr = da_arr
    np.testing.assert_allclose(np.clip(d, -1, 2).compute(), np.clip(arr, -1, 2))   # block-wise, was buggy
    np.testing.assert_allclose(np.round(d).compute(), np.round(arr))


# --- ufunc protocol (__array_ufunc__) ---

@pytest.mark.parametrize("npf,ref", [
    (np.sqrt, lambda a: np.sqrt(np.abs(a))),
    (np.absolute, np.abs),      # alias: absolute -> abs
    (np.abs, np.abs),
    (np.negative, np.negative),
    (np.exp, lambda a: np.exp(np.clip(a, None, 5))),
    (np.sign, np.sign),
    (np.floor, np.floor),
    (np.ceil, np.ceil),
])
def test_unary_ufunc_dispatch(da_arr, npf, ref):
    d, arr = da_arr
    # guard domain for sqrt/exp via the ref's own transform on d
    got = npf(np.abs(d)).compute() if npf is np.sqrt else (
        npf(np.clip(d, None, 5)).compute() if npf is np.exp else npf(d).compute())
    np.testing.assert_allclose(got, ref(arr), rtol=1e-5)


def test_binary_ufunc_dispatch(da_arr):
    d, arr = da_arr
    np.testing.assert_allclose(np.add(d, 2).compute(), arr + 2)
    np.testing.assert_allclose(np.add(2, d).compute(), 2 + arr)          # scalar on left
    np.testing.assert_allclose(np.multiply(d, d).compute(), arr * arr)
    np.testing.assert_array_equal(np.greater(d, 0).compute(), arr > 0)
    np.testing.assert_allclose(np.true_divide(d, 3).compute(), arr / 3)  # alias -> divide


def test_unsupported_ufunc_errors_cleanly(da_arr):
    """A ufunc dyna_zarr doesn't implement must NOT silently return wrong data."""
    d, _ = da_arr
    with pytest.raises(TypeError):
        np.sin(d).compute()


# --- operators (dunders) round-trip, incl. chaining ---

def test_operators_and_chain(da_arr):
    d, arr = da_arr
    np.testing.assert_array_equal(((d > -1) & (d < 2)).compute(), (arr > -1) & (arr < 2))
    np.testing.assert_allclose((-d).compute(), -arr)
    np.testing.assert_allclose(((d.astype("float32") + 5) / 2).clip(0, 4).compute(),
                               np.clip((arr + 5) / 2, 0, 4))


# --- dask-compat methods used by the ome_zarr_pro backend ---

def test_rechunk_persist_are_noops(da_arr):
    d, arr = da_arr
    np.testing.assert_array_equal(d.rechunk({0: d.shape[0]}).persist().compute(), arr)
    np.testing.assert_array_equal(d.rechunk(-1).compute(), arr)


def test_map_blocks_method_dask_style(da_arr):
    d, arr = da_arr
    # dask-style call: func, dtype, and a dask-only meta kwarg that must be ignored
    got = d.map_blocks(lambda b: b * 2, dtype="float32", meta=np.array((), dtype="float32"))
    np.testing.assert_allclose(got.compute(), arr * 2)
    # a bound (non-dask) kwarg is forwarded to func
    got2 = d.map_blocks(lambda b, k=1.0: b + k, dtype="float32", k=3.0)
    np.testing.assert_allclose(got2.compute(), arr + 3.0)
    with pytest.raises(NotImplementedError):
        d.map_blocks(lambda b: b, dtype="float32", drop_axis=0)


def test_map_overlap_method_dask_style(da_arr):
    from scipy import ndimage as ndi
    d, arr = da_arr
    got = d.map_overlap(lambda b: ndi.uniform_filter(b, 3, mode="reflect"),
                        depth=1, boundary="reflect", trim=True, dtype="float32",
                        meta=np.array((), dtype="float32"))
    np.testing.assert_allclose(got.compute(), ndi.uniform_filter(arr, 3, mode="reflect"), atol=1e-5)
    with pytest.raises(NotImplementedError):
        d.map_overlap(lambda b: b, depth=1, trim=False)
