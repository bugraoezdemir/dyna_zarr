"""
Correctness + chunk-invariance for the streaming reduce primitive.

A reduction must equal numpy no matter the chunking or the region driving the read, and
stay memory-bound (the reduced axis is streamed in bounded chunks). We check full compute,
random sub-slices across several chunkings, keepdims, global (axis=None), and an
end-to-end write of a projection.
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


# name -> (dyna_op, numpy_ref) ; each reduces to a known shape
REDUCERS = {
    "sum_a0":       (lambda d: ops.sum(d, 0),                 lambda a: a.sum(0)),
    "max_a1":       (lambda d: ops.max(d, 1),                 lambda a: a.max(1)),
    "min_a2":       (lambda d: ops.min(d, 2),                 lambda a: a.min(2)),
    "mean_a0":      (lambda d: ops.mean(d, 0),                lambda a: a.mean(0)),
    "prod_a1":      (lambda d: ops.prod(d, 1),                lambda a: a.prod(1)),
    "sum_a02":      (lambda d: ops.sum(d, (0, 2)),            lambda a: a.sum(axis=(0, 2))),
    "max_a0_keep":  (lambda d: ops.max(d, 0, keepdims=True),  lambda a: a.max(0, keepdims=True)),
    "mean_a12_keep":(lambda d: ops.mean(d, (1, 2), keepdims=True),
                     lambda a: a.mean(axis=(1, 2), keepdims=True)),
    "median_a0":    (lambda d: ops.median(d, 0),              lambda a: np.median(a, 0)),
    "median_a12":   (lambda d: ops.median(d, (1, 2)),         lambda a: np.median(a, axis=(1, 2))),
}


@pytest.fixture(scope="module")
def arr():
    return np.random.default_rng(0).random(SHAPE).astype(np.float32)


@pytest.mark.parametrize("name", list(REDUCERS))
def test_full_compute_matches_numpy(arr, name):
    dyna_op, ref_op = REDUCERS[name]
    got = dyna_op(da_from(arr, (2, 3, 4))).compute()
    np.testing.assert_allclose(got, ref_op(arr), rtol=1e-5, atol=1e-5,
                               err_msg=f"{name}: full compute")


@pytest.mark.parametrize("chunks", [(6, 8, 10), (2, 3, 4), (1, 8, 10), (3, 1, 5)])
def test_random_subslice_chunk_invariant(arr, chunks):
    rng = np.random.default_rng(7)
    failures = []
    for name, (dyna_op, ref_op) in REDUCERS.items():
        result = dyna_op(da_from(arr, chunks))
        ref = ref_op(arr)
        for _ in range(15):
            k = random_key(rng, ref.shape)
            try:
                np.testing.assert_allclose(result[k].compute(), ref[k], rtol=1e-5, atol=1e-5)
            except Exception as e:
                failures.append(f"{name} chunks={chunks} key={k}: "
                                f"{type(e).__name__}: {str(e).splitlines()[-1][:80]}")
                break
    assert not failures, "reduce chunk-invariance failures:\n" + "\n".join(failures)


def test_global_reductions(arr):
    da = da_from(arr, (2, 3, 4))
    for name, npf in [("sum", np.sum), ("max", np.max), ("min", np.min), ("mean", np.mean)]:
        got = getattr(ops, name)(da).compute()
        assert np.ndim(got) == 0
        np.testing.assert_allclose(got, npf(arr), rtol=1e-5, atol=1e-5, err_msg=name)


def test_any_all(arr):
    mask = (arr > 0.5)
    dm = da_from(mask, (2, 3, 4))
    np.testing.assert_array_equal(ops.any(dm, 0).compute(), mask.any(0))
    np.testing.assert_array_equal(ops.all(dm, 1).compute(), mask.all(1))
    assert bool(ops.any(dm).compute()) == bool(mask.any())


def test_streaming_is_memory_bounded(arr):
    """A tiny strip budget forces many reduced-axis chunks but the result is unchanged
    (proves the associative streaming path, independent of strip size)."""
    from dyna_zarr.operations.reductions import ReduceTransform
    da = da_from(arr, (2, 3, 4))
    t = ReduceTransform(da, "sum", axis=0, strip_bytes=64)  # forces chunk_len == 1
    streamed = da._with_transform(t).compute()
    np.testing.assert_allclose(streamed, arr.sum(0), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("chunks", [(6, 8, 10), (2, 3, 4), (1, 8, 10)])
def test_histogram_matches_numpy(arr, chunks):
    """Streaming histogram == numpy exactly, across chunkings (associativity) and for
    auto-range, fixed-range, and explicit-edge bins."""
    da = da_from((arr * 255).astype(np.float32), chunks)
    a = (arr * 255).astype(np.float32)
    # auto range
    c, e = ops.histogram(da, bins=64)
    cn, en = np.histogram(a, bins=64)
    np.testing.assert_array_equal(c, cn)
    np.testing.assert_allclose(e, en)
    assert c.sum() == a.size and c.dtype == np.int64
    # fixed range
    c2, _ = ops.histogram(da, bins=32, range=(0, 255))
    np.testing.assert_array_equal(c2, np.histogram(a, bins=32, range=(0, 255))[0])
    # explicit edges + per-plane slice
    edges = np.linspace(0, 255, 40)
    np.testing.assert_array_equal(ops.histogram(da, bins=edges)[0], np.histogram(a, bins=edges)[0])
    np.testing.assert_array_equal(da[2].histogram(64, (0, 255))[0],
                                  np.histogram(a[2], bins=64, range=(0, 255))[0])


def test_write_projection(arr, tmp_path):
    """End-to-end: max-projection over axis 0 through io.write == numpy."""
    from dyna_zarr import io
    da = da_from(arr, (2, 3, 4))
    out = str(tmp_path / "proj.zarr")
    io.write(ops.max(da, 0), out, zarr_format=2, chunks=(4, 5))
    np.testing.assert_allclose(zarr.open(out, mode="r")[:], arr.max(0), rtol=1e-5, atol=1e-5)
