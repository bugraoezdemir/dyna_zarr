"""
Correctness + chunk-invariance for the map_overlap neighbourhood primitive.

Each neighbourhood filter must equal scipy applied to the *whole* array, no matter what
region size drives the read. We check two things against a scipy reference:

1. **Full compute** ``op(da).compute() == scipy(arr)``
2. **Random sub-slice** ``op(da)[k] == scipy(arr)[k]`` over many random keys AND several
   chunkings -- the real test that each read pulls its own halo correctly (chunk/region
   invariance) and that boundaries are handled exactly.
"""

import numpy as np
import pytest
import zarr
from scipy import ndimage as ndi

from dyna_zarr import DynamicArray, operations as ops


SHAPE = (12, 16, 20)


def da_from(arr, chunks):
    return DynamicArray(zarr.array(arr, chunks=chunks))


def random_key(rng, shape):
    key = []
    for size in shape:
        c = rng.integers(0, 4)
        if c == 0:
            key.append(int(rng.integers(0, size)))
        elif c == 1:
            key.append(slice(None))
        elif c == 2:
            a, b = sorted(rng.integers(0, size + 1, size=2))
            key.append(slice(int(a), int(b)))
        else:
            a, b = sorted(rng.integers(0, size + 1, size=2))
            key.append(slice(int(a), int(b), int(rng.integers(1, 3))))
    return tuple(key)


_KERNEL = np.random.default_rng(7).random((3, 3, 3)).astype(np.float32)   # fixed conv kernel


# name -> (dyna_op, scipy_reference); reference uses mode='reflect' to match default boundary
OPS = {
    "convolve_3":       (lambda da: ops.convolve(da, _KERNEL),
                         lambda a: ndi.convolve(a, _KERNEL, mode="reflect")),
    "correlate_3":      (lambda da: ops.correlate(da, _KERNEL),
                         lambda a: ndi.correlate(a, _KERNEL, mode="reflect")),
    "gaussian_s2":      (lambda da: ops.gaussian_filter(da, 2.0),
                         lambda a: ndi.gaussian_filter(a, 2.0, mode="reflect")),
    "gaussian_aniso":   (lambda da: ops.gaussian_filter(da, (1.0, 2.0, 0.5)),
                         lambda a: ndi.gaussian_filter(a, (1.0, 2.0, 0.5), mode="reflect")),
    "uniform_5":        (lambda da: ops.uniform_filter(da, 5),
                         lambda a: ndi.uniform_filter(a, 5, mode="reflect")),
    "uniform_even_4":   (lambda da: ops.uniform_filter(da, 4),
                         lambda a: ndi.uniform_filter(a, 4, mode="reflect")),
    "median_3":         (lambda da: ops.median_filter(da, 3),
                         lambda a: ndi.median_filter(a, 3, mode="reflect")),
    "minimum_3":        (lambda da: ops.minimum_filter(da, 3),
                         lambda a: ndi.minimum_filter(a, 3, mode="reflect")),
    "maximum_3":        (lambda da: ops.maximum_filter(da, 3),
                         lambda a: ndi.maximum_filter(a, 3, mode="reflect")),
    "grey_erosion_3":   (lambda da: ops.grey_erosion(da, 3),
                         lambda a: ndi.grey_erosion(a, size=3, mode="reflect")),
    "grey_dilation_3":  (lambda da: ops.grey_dilation(da, 3),
                         lambda a: ndi.grey_dilation(a, size=3, mode="reflect")),
}


@pytest.fixture(scope="module")
def arr():
    return np.random.default_rng(0).random(SHAPE).astype(np.float32)


@pytest.mark.parametrize("op_name", list(OPS))
def test_full_compute_matches_scipy(arr, op_name):
    dyna_op, ref_op = OPS[op_name]
    da = da_from(arr, chunks=(4, 4, 5))
    np.testing.assert_allclose(dyna_op(da).compute(), ref_op(arr), atol=1e-5,
                               err_msg=f"{op_name}: full compute != scipy")


@pytest.mark.parametrize("chunks", [(12, 16, 20), (3, 4, 5), (5, 7, 6), (1, 16, 20)])
def test_random_subslice_chunk_invariant(arr, chunks):
    """op(da)[k] == scipy(arr)[k] over random keys and several chunkings."""
    rng = np.random.default_rng(1234)
    failures = []
    for op_name, (dyna_op, ref_op) in OPS.items():
        da = da_from(arr, chunks=chunks)
        ref = ref_op(arr)
        result = dyna_op(da)
        for _ in range(15):
            k = random_key(rng, SHAPE)
            try:
                got = result[k].compute()
                np.testing.assert_allclose(got, ref[k], atol=1e-5)
            except Exception as e:
                failures.append(f"{op_name} chunks={chunks} key={k}: "
                                f"{type(e).__name__}: {str(e).splitlines()[-1][:80]}")
                break
    assert not failures, "map_overlap chunk-invariance failures:\n" + "\n".join(failures)


def test_write_readback_matches_scipy(arr, tmp_path):
    """End-to-end: gaussian through io.write (region-streamed) == scipy(whole)."""
    from dyna_zarr import io
    da = da_from(arr, chunks=(4, 4, 5))
    out = str(tmp_path / "gauss.zarr")
    io.write(ops.gaussian_filter(da, 2.0), out, zarr_format=2, chunks=(4, 4, 5))
    written = zarr.open(out, mode="r")[:]
    np.testing.assert_allclose(written, ndi.gaussian_filter(arr, 2.0, mode="reflect"), atol=1e-5)
