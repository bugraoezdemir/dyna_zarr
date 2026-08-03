"""
Correctness + chunk-invariance for the ops added to match ome_zarr_pro's dask surface:
Tier 1 (isin, digitize, rot90) and Tier 2 (var, std, argmin, argmax, diff, gradient).

Each must equal numpy for full compute AND for random sub-slices across several chunkings
(the read-key math / streaming must be region-independent).
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, operations as ops

SHAPE = (6, 8, 10)


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


@pytest.fixture(scope="module")
def arr():
    rng = np.random.default_rng(0)
    a = rng.random(SHAPE).astype(np.float32)
    a[a < 0.1] = 0.25   # a few exact bin values for isin/digitize
    return a


# name -> (dyna_op, numpy_ref, exact?)   -- all support sub-slicing
OPS = {
    "isin":       (lambda d: ops.isin(d, [0.25, 0.5]),         lambda a: np.isin(a, [0.25, 0.5]), True),
    "digitize":   (lambda d: ops.digitize(d, [0.25, 0.5, 0.75]), lambda a: np.digitize(a, [0.25, 0.5, 0.75]), True),
    "rot90_k1":   (lambda d: ops.rot90(d, 1, (1, 2)),          lambda a: np.rot90(a, 1, (1, 2)), False),
    "rot90_k3":   (lambda d: ops.rot90(d, 3, (0, 2)),          lambda a: np.rot90(a, 3, (0, 2)), False),
    "var_a0":     (lambda d: ops.var(d, 0),                    lambda a: a.var(0), False),
    "std_a1":     (lambda d: ops.std(d, 1),                    lambda a: a.std(1), False),
    "var_ddof1":  (lambda d: ops.var(d, 2, ddof=1),            lambda a: a.var(2, ddof=1), False),
    "argmin_a0":  (lambda d: ops.argmin(d, 0),                 lambda a: a.argmin(0), True),
    "argmax_a2":  (lambda d: ops.argmax(d, 2),                 lambda a: a.argmax(2), True),
    "diff_a2":    (lambda d: ops.diff(d, axis=2),              lambda a: np.diff(a, axis=2), False),
    "diff_n2_a0": (lambda d: ops.diff(d, 2, axis=0),           lambda a: np.diff(a, 2, axis=0), False),
    "gradient_a1":(lambda d: ops.gradient(d, axis=1),          lambda a: np.gradient(a, axis=1), False),
}


@pytest.mark.parametrize("name", list(OPS))
def test_full_compute_matches_numpy(arr, name):
    dyna_op, ref_op, exact = OPS[name]
    got = np.asarray(dyna_op(da_from(arr, (2, 3, 4))).compute())
    ref = np.asarray(ref_op(arr))
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    if exact:
        np.testing.assert_array_equal(got, ref)
    else:
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunks", [SHAPE, (2, 3, 4), (1, 8, 10), (3, 1, 5)])
def test_random_subslice_chunk_invariant(arr, chunks):
    rng = np.random.default_rng(7)
    failures = []
    for name, (dyna_op, ref_op, exact) in OPS.items():
        result = dyna_op(da_from(arr, chunks))
        ref = np.asarray(ref_op(arr))
        for _ in range(12):
            k = random_key(rng, ref.shape)
            try:
                got = np.asarray(result[k].compute())
                if exact:
                    np.testing.assert_array_equal(got, ref[k])
                else:
                    np.testing.assert_allclose(got, ref[k], rtol=1e-4, atol=1e-4)
            except Exception as e:
                failures.append(f"{name} chunks={chunks} key={k}: "
                                f"{type(e).__name__}: {str(e).splitlines()[-1][:70]}")
                break
    assert not failures, "new-op chunk-invariance failures:\n" + "\n".join(failures)
