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
        np.modf(d)          # multi-output ufunc: unsupported -> clean TypeError, not wrong data


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


# --- numpy high-level function protocol (__array_function__) ---

def test_array_function_dispatch(da_arr):
    """np.<func>(DynamicArray) dispatches to the lazy op (like dask), for the functions
    ome_zarr_pro calls uniformly across backends."""
    d, arr = da_arr
    e = DynamicArray(zarr.array(arr[::-1].copy(), chunks=(1, 4, 4)))
    er = arr[::-1]
    np.testing.assert_array_equal(np.stack([d, e], 1).compute(), np.stack([arr, er], 1))
    np.testing.assert_array_equal(np.concatenate([d, e], 0).compute(), np.concatenate([arr, er], 0))
    np.testing.assert_allclose(np.where(d > 0, d, e).compute(), np.where(arr > 0, arr, er))
    np.testing.assert_array_equal(np.transpose(d, (2, 0, 1)).compute(), np.transpose(arr, (2, 0, 1)))
    np.testing.assert_array_equal(np.transpose(d).compute(), np.transpose(arr))
    np.testing.assert_array_equal(np.flip(d).compute(), np.flip(arr))          # all axes (chained)
    np.testing.assert_array_equal(np.flip(d, (0, 2)).compute(), np.flip(arr, (0, 2)))
    np.testing.assert_array_equal(np.expand_dims(d, 1).compute(), np.expand_dims(arr, 1))
    np.testing.assert_allclose(np.max(d, axis=0).compute(), np.max(arr, axis=0))
    np.testing.assert_allclose(np.mean(d, axis=1, keepdims=True).compute(), np.mean(arr, axis=1, keepdims=True))
    np.testing.assert_allclose(np.std(d, axis=0).compute(), np.std(arr, axis=0), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.cumsum(d, axis=0).compute(), np.cumsum(arr, axis=0))
    np.testing.assert_allclose(np.diff(d, axis=2).compute(), np.diff(arr, axis=2))


def test_array_function_unsupported_returns_notimplemented(da_arr):
    d, _ = da_arr
    with pytest.raises(TypeError):
        np.linalg.norm(d)     # not in the registry -> NotImplemented -> numpy raises
