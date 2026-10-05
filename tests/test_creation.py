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


MIB = 1 << 20


@pytest.mark.parametrize("shape, dtype, expected", [
    # 1 MiB exactly, power-of-two sides: a cube when the element count is 2**(3k),
    # otherwise the spare factors of 2 go to the innermost axes (x, then y).
    ((300, 1024, 1024), "float32", (64, 64, 64)),        # 2**18 elements: a cube
    ((300, 1024, 1024), "uint8", (64, 128, 128)),        # 2**20
    ((300, 1024, 1024), "uint16", (64, 64, 128)),        # 2**19
    ((300, 1024, 1024), "float64", (32, 64, 64)),        # 2**17
    ((10, 3, 64, 512, 512), "uint16", (1, 1, 64, 64, 128)),  # leading t/c axes stay 1
    ((2048, 2048), "float32", (512, 512)),               # 2-D: a square
    ((3, 1024, 1024), "float32", (3, 256, 256)),         # short axis whole; under budget
    ((16, 16, 16), "float32", (16, 16, 16)),             # smaller than the budget: whole
])
def test_default_grid_is_a_power_of_two_1mib_chunk_sized_by_dtype(shape, dtype, expected):
    for op in (lambda: ops.zeros(shape, dtype=dtype), lambda: ops.ones(shape, dtype=dtype),
               lambda: ops.full(shape, 3, dtype=dtype), lambda: ops.empty(shape, dtype=dtype),
               lambda: ops.random(shape, dtype=dtype, seed=0)):
        a = op()
        assert a.chunks == expected
        assert int(np.prod(a.chunks)) * np.dtype(dtype).itemsize <= MIB
        # every side is a power of two unless it spans its whole axis
        for c, s in zip(a.chunks, shape):
            assert c == s or (c & (c - 1)) == 0, (c, s)


def test_explicit_chunks_are_honoured():
    assert ops.zeros((300, 1024, 1024), dtype="f4", chunks=(10, 256, 256)).chunks == (10, 256, 256)
    assert ops.random((8, 8), seed=0, chunks=(3, 5)).chunks == (3, 5)


def test_like_inherits_a_grid_or_falls_back_to_the_default():
    stored = DynamicArray(zarr.array(np.zeros((32, 64, 64), "f4"), chunks=(8, 16, 16)))
    assert ops.zeros_like(stored).chunks == (8, 16, 16)             # source grid kept
    gridless = ops.reshape(stored, (32, 64 * 64))                   # reshape leaves no grid
    assert gridless.chunks is None
    # the whole (32, 4096) uint8 array is 128 KiB, under the 1 MiB budget -> one chunk
    assert ops.ones_like(gridless, dtype="uint8").chunks == (32, 4096)
    # float64 at a size that does not fit: 1 MiB = 2**17 elements, extra factor on x.
    # (A lazy zeros source reshaped: grid-less, and nothing is allocated.)
    big = ops.reshape(ops.zeros((64, 2048, 2048), dtype="u1"), (2048, 2048 * 64))
    assert big.chunks is None
    assert ops.zeros_like(big, dtype="float64").chunks == (256, 512)
    assert ops.full_like(gridless, 1, chunks=(4, 4)).chunks == (4, 4)


def test_large_generated_array_writes_with_bounded_regions(tmp_path, capsys):
    # Previously the default grid was the WHOLE array (1.2 GB here), and io.write keeps
    # the input grid, so one region had to hold all of it. Now chunk and region are small.
    z = ops.zeros((300, 1024, 1024), dtype="float32")
    io.write(z[:64, :256, :256], str(tmp_path / "z.zarr"))         # 16 MB corner
    out = zarr.open_array(str(tmp_path / "z.zarr"))
    assert out.chunks == (64, 64, 64)
    assert int(np.prod(out.chunks)) * 4 <= MIB
    assert "memory floor" not in capsys.readouterr().out            # no oversized-chunk note
    assert not out[0:2, 0:3, 0:4].any()


def test_random_stays_position_deterministic_under_the_default_grid(tmp_path):
    r = ops.random((40, 70, 90), seed=5)                             # default grid, ragged edges
    io.write(r, str(tmp_path / "r.zarr"))
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "r.zarr"))[...], r.compute())


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
