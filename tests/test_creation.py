"""
Creation ops (nullary generative sources): zeros/ones/full/empty/random + *_like.

These synthesize each region lazily (no underlying array), compose with other ops, and
stream to disk. ``random`` must be position-deterministic: any sub-slice equals the whole
array sliced, and io.write matches compute.
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations as ops


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


def test_deterministic_creation():
    for op, npf in [(ops.zeros, np.zeros), (ops.ones, np.ones)]:
        a = op((4, 6), dtype=np.float32)
        assert a.shape == (4, 6) and a.dtype == np.float32
        np.testing.assert_array_equal(a.compute(), npf((4, 6), np.float32))
    np.testing.assert_array_equal(ops.full((3, 5), 7.0).compute(), np.full((3, 5), 7.0))
    # full infers dtype from fill_value
    assert ops.full((2, 2), 3).dtype == np.array(3).dtype


def test_creation_subslice_and_chain():
    z = ops.ones((8, 8), dtype=np.float32)
    np.testing.assert_array_equal(z[2:5, ::2].compute(), np.ones((8, 8), np.float32)[2:5, ::2])
    np.testing.assert_allclose((ops.ones((4, 4)) * 3 + 1).compute(), np.full((4, 4), 4.0))


def test_like():
    base = ops.zeros((3, 5), dtype=np.int16)
    assert ops.full_like(base, 9).dtype == np.int16
    np.testing.assert_array_equal(ops.full_like(base, 9).compute(), np.full((3, 5), 9, np.int16))
    np.testing.assert_array_equal(ops.ones_like(base).compute(), np.ones((3, 5), np.int16))


def test_random_position_deterministic():
    r = ops.random((6, 8, 10), seed=42)
    whole = r.compute()
    assert whole.shape == (6, 8, 10) and whole.dtype == np.float32
    assert 0.0 <= whole.min() and whole.max() < 1.0
    np.testing.assert_array_equal(whole, r.compute())          # idempotent
    rng = np.random.default_rng(1)
    for _ in range(40):                                        # chunk-invariant sub-slices
        k = random_key(rng, (6, 8, 10))
        np.testing.assert_array_equal(np.asarray(r[k].compute()), whole[k])
    # separate seeds differ
    assert not np.array_equal(ops.random((4, 4), seed=1).compute(),
                              ops.random((4, 4), seed=2).compute())


def test_write_generative(tmp_path):
    for name, pipe, ref in [
        ("full", ops.full((16, 32, 32), 5.0, dtype=np.float32), np.full((16, 32, 32), 5.0, np.float32)),
        ("random", ops.random((16, 32, 32), seed=7), None),
    ]:
        out = str(tmp_path / f"{name}.zarr")
        io.write(pipe, out, zarr_format=2, chunks=(4, 16, 16))
        written = zarr.open(out, mode="r")[:]
        expected = ref if ref is not None else pipe.compute()   # write must equal compute
        np.testing.assert_array_equal(written, expected)
