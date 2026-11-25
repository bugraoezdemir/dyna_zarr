"""concatenate and stack of inputs with different dtypes (Workplan D1).

The combined array reports the dtype NumPy gives (np.concatenate / np.stack promote),
and every read has that dtype, whichever parts it touches: inside the narrower part,
inside the wider one, across both, empty, or an integer index. A write then keeps the
wider part's values; before, the array reported the first part's dtype, and writing
uint8 + uint16 stored 1000 as 255.
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import from_array, io
from dyna_zarr import operations as ops

PAIRS = [
    ("u1", "u2"),       # the case found: uint8 first, uint16 after
    ("u2", "u1"),       # the wider part first (was already right; stays so)
    ("u1", "i1"),       # unsigned with signed -> int16
    ("u2", "f4"),       # integer with float -> float32
    ("i4", "f8"),
    ("u2", "u2"),       # same dtype: unchanged
]


def _data(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return rng.normal(scale=1000, size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)


def _pair(first, second):
    a = _data((3, 4, 5), first, 0)
    b = _data((2, 4, 5), second, 1)
    return a, b


CONCAT_KEYS = {
    "whole": np.s_[...],
    "inside first": np.s_[0:3, :, 1:4],
    "inside second": np.s_[3:5, 1:3, :],
    "across": np.s_[2:4, :, :],
    "int in first": np.s_[1, :, :],
    "int in second": np.s_[4, 2, :],
    "steps": np.s_[0:5:2, ::2, 1::2],
    "empty": np.s_[2:2, :, :],
}


@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("key", CONCAT_KEYS.values(), ids=CONCAT_KEYS.keys())
def test_concatenate_reads_with_numpys_dtype(first, second, key):
    a, b = _pair(first, second)
    joined = ops.concatenate([from_array(a), from_array(b)], axis=0)
    want = np.concatenate([a, b], axis=0)
    assert joined.dtype == want.dtype
    got = np.asarray(joined[key])
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want[key])


def test_concatenate_dtype_counts_every_part_not_only_those_read():
    # three parts, the wide one last: a read across the first two still has its dtype
    a, b, c = (np.full((2, 3), 7, "u1"), np.full((2, 3), 8, "u1"),
               np.full((2, 3), 1000, "u2"))
    joined = ops.concatenate([from_array(a), from_array(b), from_array(c)], axis=0)
    assert joined.dtype == np.uint16
    assert np.asarray(joined[1:3]).dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(joined.compute()),
                                  np.concatenate([a, b, c], axis=0))


STACK_KEYS = {
    "whole": np.s_[...],
    "first part": np.s_[0],
    "second part": np.s_[1, :, 1:3],
    "slice of parts": np.s_[0:2, 1, :],
    "one part by slice": np.s_[1:2],
    "empty": np.s_[1:1],
}


@pytest.mark.parametrize("first,second", PAIRS)
@pytest.mark.parametrize("key", STACK_KEYS.values(), ids=STACK_KEYS.keys())
def test_stack_reads_with_numpys_dtype(first, second, key):
    a = _data((4, 5), first, 0)
    b = _data((4, 5), second, 1)
    stacked = ops.stack([from_array(a), from_array(b)], axis=0)
    want = np.stack([a, b], axis=0)
    assert stacked.dtype == want.dtype
    got = np.asarray(stacked[key])
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want[key])


@pytest.mark.parametrize("op", ["concatenate", "stack"])
def test_write_keeps_the_wider_parts_values(tmp_path, op):
    a = np.full((2, 4), 200, "u1")
    b = np.full((2, 4), 1000, "u2")
    if op == "concatenate":
        combined = ops.concatenate([from_array(a), from_array(b)], axis=0)
        want = np.concatenate([a, b], axis=0)
    else:
        combined = ops.stack([from_array(a), from_array(b)], axis=0)
        want = np.stack([a, b], axis=0)
    path = str(tmp_path / "out.zarr")
    io.write(combined, path)
    written = zarr.open_array(path, mode="r")
    assert written.dtype == np.uint16
    np.testing.assert_array_equal(written[...], want)


@pytest.mark.engine
def test_mixed_dtypes_with_the_engine_on(tmp_path):
    """The engine joins parts of one dtype only; mixed parts are read by the Python
    path and give the same values and dtype as NumPy."""
    pytest.importorskip("dyna_zarr_core")
    from dyna_zarr import _engine
    saved = _engine._choice
    _engine.set_engine(True)
    try:
        a, b = _pair("u1", "u2")
        joined = ops.concatenate([from_array(a), from_array(b)], axis=0)
        want = np.concatenate([a, b], axis=0)
        assert joined.dtype == want.dtype
        for key in CONCAT_KEYS.values():
            got = np.asarray(joined[key])
            assert got.dtype == want.dtype
            np.testing.assert_array_equal(got, want[key])
        path = str(tmp_path / "engine.zarr")
        io.write(joined, path)
        np.testing.assert_array_equal(zarr.open_array(path, mode="r")[...], want)
    finally:
        _engine._choice = saved
