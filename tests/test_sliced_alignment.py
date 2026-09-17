"""Writing a SLICE of an aligned map_overlap array.

`labels[a:b]` wraps a map_overlap(align=cell) in a SliceTransform. The alignment stays
valid under the slice - just shifted by the slice start - and io.write cuts its read
bands on that shifted grid so each cell is computed exactly once.

These guard two regressions found by hand:
  * the bands must COVER the axis; an earlier attempt returned a single start while the
    band width stayed at region_shape, silently writing only the first band and leaving
    the rest of the store zero-filled.
  * a misaligned slice must still be byte-exact, even though its bands cut mid-chunk
    (two neighbouring bands then write the same boundary chunk; TensorStore serialises
    the read-modify-write, so this is safe).
"""
import numpy as np
import pytest
import zarr

from dyna_zarr.io import io as dio
from dyna_zarr import operations as dops


def _source(tmp_path, n=128, chunk=16):
    p = str(tmp_path / "src.zarr")
    z = zarr.create_array(store=p, shape=(n, n), chunks=(chunk, chunk),
                          dtype="int32", overwrite=True)
    z[:] = np.arange(n * n, dtype="int32").reshape(n, n)
    return p


def _position_sum(block, location, **_):
    """Deterministic, position-addressed: value depends only on global position."""
    (z0, _), (y0, _) = location
    out = np.empty(block.shape, dtype="int32")
    for i in range(block.shape[0]):
        for j in range(block.shape[1]):
            out[i, j] = (z0 + i) * 1000 + (y0 + j)
    return out


@pytest.mark.parametrize("sl", [
    (slice(None), slice(None)),        # full
    (slice(0, 64), slice(0, 64)),      # aligned start
    (slice(32, 96), slice(32, 96)),    # chunk-aligned, cell-offset
    (slice(13, 77), slice(5, 69)),     # arbitrary offset (bands cut mid-chunk)
    (slice(7, 20), slice(7, 20)),      # smaller than one cell
])
def test_sliced_write_matches_compute(tmp_path, sl):
    src = _source(tmp_path)
    arr = dio.read(src)
    lab = dops.map_overlap(arr, _position_sum, depth=0, boundary="constant",
                           dtype="int32", block_info=True, align=(32, 32))
    sub = lab[sl]
    expected = np.asarray(sub.compute())

    out = str(tmp_path / "out.zarr")
    dio.write(sub, out, chunks=(16, 16), max_workers=4, zarr_format=3, overwrite=True)
    got = zarr.open_array(out, mode="r")[:]

    assert got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)
    # a silent partial write shows up as leftover zeros where the source has none
    if (expected != 0).any():
        assert (got != 0).any()
