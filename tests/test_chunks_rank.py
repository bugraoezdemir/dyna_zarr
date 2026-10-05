"""``.chunks`` must describe THIS array, never a stale grid from its source.

The source's chunk tuple is copied down the transform chain, so an op that
changes the rank used to report the source's tuple against its own shape:
``reshape((64,128,128) -> (64,16384))`` answered ``(8,64,64)``. That is worse
than None, because the chunk grid drives the writer's chunk and region defaults
-- None is handled everywhere, a rank-3 tuple on a 2-D array is not.
"""

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io
from dyna_zarr.operations import structural as S


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "src.zarr"
    z = zarr.create_array(store=str(path), shape=(64, 128, 128),
                          chunks=(8, 64, 64), dtype="uint16")
    z[:] = np.arange(64 * 128 * 128, dtype="uint16").reshape(64, 128, 128)
    return DynamicArray(z)


def _cases(a):
    return [
        ("source", a),
        ("reshape-2d", a.reshape((64, 16384))),
        ("reshape-1d", a.reshape((64 * 128 * 128,))),
        ("flatten", a.flatten()),
        ("transpose", S.transpose(a, (2, 1, 0))),
        ("expand_dims", S.expand_dims(a, 0)),
        ("slice", a[:32]),
        ("pad", S.pad(a, ((1, 1), (0, 0), (0, 0)))),
        ("tile", S.tile(a, (2, 1, 1))),
    ]


def test_chunks_rank_always_matches_shape(source):
    """The invariant: chunks is either None or the same rank as shape."""
    for name, arr in _cases(source):
        chunks = arr.chunks
        assert chunks is None or len(chunks) == len(arr.shape), (
            f"{name}: shape {arr.shape} ({len(arr.shape)}-D) but chunks "
            f"{chunks} ({len(chunks)}-D)")


def test_reshape_drops_the_grid(source):
    """Reshape rewrites the index space, so no grid survives it."""
    assert source.reshape((64, 16384)).chunks is None
    assert source.flatten().chunks is None


def test_pad_drops_the_grid(source):
    """Padding shifts every boundary by the pad width, so the source grid no
    longer describes the output even though the rank is unchanged."""
    assert S.pad(source, ((1, 1), (0, 0), (0, 0))).chunks is None


def test_tile_keeps_the_grid(source):
    """Tiling repeats the whole array, so a source chunk still tiles the output
    exactly whenever it tiled the input."""
    assert S.tile(source, (2, 1, 1)).chunks == (8, 64, 64)


def test_transpose_permutes_the_grid(source):
    assert S.transpose(source, (2, 1, 0)).chunks == (64, 64, 8)


def test_flatten_shape_holds_plain_ints(source):
    """np.prod returns np.int64, which prints as 'np.int64(1048576)' wherever
    the writer formats a shape."""
    shape = source.flatten().shape
    assert all(type(s) is int for s in shape), shape


def test_reshape_write_works_with_default_region(source, tmp_path):
    """Regression: the specialised reshape/flatten writers read region_size_mb
    before its None default was resolved, so the documented default crashed
    with a TypeError inside int(None * 1024 * 1024)."""
    out = tmp_path / "reshaped.zarr"
    io.write(source.reshape((64, 16384)), str(out))
    assert zarr.open_array(str(out), mode="r").shape == (64, 16384)


def test_flatten_write_works_with_default_region(source, tmp_path):
    out = tmp_path / "flat.zarr"
    io.write(source.flatten(), str(out))
    assert zarr.open_array(str(out), mode="r").shape == (64 * 128 * 128,)


def test_reshape_roundtrip_is_exact(source, tmp_path):
    out = tmp_path / "r.zarr"
    io.write(source.reshape((64, 16384)), str(out))
    expected = np.arange(64 * 128 * 128, dtype="uint16").reshape(64, 16384)
    np.testing.assert_array_equal(zarr.open_array(str(out), mode="r")[:], expected)
