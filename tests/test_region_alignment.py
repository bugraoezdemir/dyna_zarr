"""`io.write` adopts an op chain's read-alignment grid as its region size.

A `map_overlap(..., align=cell)` transform expands every read to whole `cell`-sized cells,
so it can only produce output a whole cell at a time. Writing it in regions smaller than a
cell recomputes the cell once per region - 8x the work for half-sized regions in 3-D, 64x
for quarter-sized (measured with tilewise-ccl's Phase B). Inferring the region from the
alignment means callers no longer have to know a producer's tiling and restate it as
`region_shape=`.
"""

import io as _io
import contextlib
import os

import numpy as np
import pytest
import zarr

from dyna_zarr import operations as ops
from dyna_zarr.io import io


def _region_line(fn):
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    lines = [l for l in buf.getvalue().splitlines() if "Region:" in l]
    assert lines, "writer printed no region line"
    return lines[0]


def _src(tmp_path, shape=(128, 128, 128), chunks=(32, 32, 32)):
    p = tmp_path / "src.zarr"
    z = zarr.create_array(store=str(p), shape=shape, chunks=chunks, dtype="uint8")
    z[:] = 3
    return io.read(str(p))


def test_alignment_is_adopted_as_region(tmp_path):
    a = _src(tmp_path)
    aligned = ops.map_overlap(a, lambda b: b, depth=0, boundary="constant",
                              dtype="uint8", align=(64, 64, 64))
    line = _region_line(lambda: io.write(aligned, str(tmp_path / "o.zarr"),
                                         chunks=(32, 32, 32), overwrite=True))
    assert "Region: (64, 64, 64)" in line
    assert "alignment" in line


def test_explicit_region_shape_still_wins(tmp_path):
    """An explicit region_shape overrides the inferred one."""
    a = _src(tmp_path)
    aligned = ops.map_overlap(a, lambda b: b, depth=0, boundary="constant",
                              dtype="uint8", align=(64, 64, 64))
    line = _region_line(lambda: io.write(aligned, str(tmp_path / "o.zarr"),
                                         chunks=(32, 32, 32),
                                         region_shape=(32, 32, 32), overwrite=True))
    assert "Region: (32, 32, 32)" in line
    assert "explicit region_shape" in line


def test_unaligned_chunks_fall_back(tmp_path):
    """An alignment that is not a whole multiple of the chunks must NOT be adopted.

    Regions would then write partial chunks, and the parallel region writes would race.
    """
    a = _src(tmp_path)
    aligned = ops.map_overlap(a, lambda b: b, depth=0, boundary="constant",
                              dtype="uint8", align=(48, 48, 48))
    line = _region_line(lambda: io.write(aligned, str(tmp_path / "o.zarr"),
                                         chunks=(32, 32, 32), overwrite=True))
    assert "alignment" not in line


@pytest.mark.parametrize("build", [
    lambda a: a,                                          # no transform at all
    lambda a: ops.gaussian_filter(a, 1.0),                # map_overlap, but align=None
    lambda a: ops.abs(a),                                 # pointwise
])
def test_no_alignment_uses_size_heuristic(tmp_path, build):
    """Arrays without an alignment keep the region_size_mb behaviour unchanged."""
    a = _src(tmp_path)
    line = _region_line(lambda: io.write(build(a), str(tmp_path / "o.zarr"),
                                         chunks=(32, 32, 32), overwrite=True))
    assert "alignment" not in line
    assert "MB" in line


def test_inferred_region_round_trips(tmp_path):
    """Inference must not change the data written."""
    a = _src(tmp_path, shape=(64, 64, 64), chunks=(16, 16, 16))
    expected = np.asarray(a._read_direct((slice(None),) * 3))
    aligned = ops.map_overlap(a, lambda b: b, depth=0, boundary="constant",
                              dtype="uint8", align=(32, 32, 32))
    out = tmp_path / "o.zarr"
    with contextlib.redirect_stdout(_io.StringIO()):
        io.write(aligned, str(out), chunks=(16, 16, 16), overwrite=True)
    np.testing.assert_array_equal(zarr.open_array(str(out), mode="r")[...], expected)


def test_integer_index_is_aligned(tmp_path):
    """An INTEGER index must honour the alignment grid too.

    The axis is squeezed only after the crop, so alignment and squeezing are independent.
    An earlier version skipped alignment for integer-indexed axes ("never align a squeezed
    axis"), which handed a position-aware func a 1-voxel slab instead of a whole cell.
    """
    a = _src(tmp_path)
    seen = []

    def record(block, location):
        seen.append(tuple(location))
        return block

    aligned = ops.map_overlap(a, record, depth=0, boundary="constant",
                              dtype="uint8", block_info=True, align=(64, 64, 64))
    aligned[70].compute()
    # axis 0 was indexed with the integer 70 -> its CORE must be the whole 64-cell [64, 128)
    assert seen[0][0] == (64, 128), seen[0]


def test_integer_index_matches_full_read(tmp_path):
    """`arr[i]` must equal row i of the fully materialized array."""
    a = _src(tmp_path)
    aligned = ops.map_overlap(a, lambda b: b + 1, depth=0, boundary="constant",
                              dtype="uint8", align=(64, 64, 64))
    full = np.asarray(aligned)
    for i in (0, 1, 63, 64, 65, 127):
        np.testing.assert_array_equal(np.asarray(aligned[i]), full[i])


def test_integer_index_without_alignment_is_unchanged(tmp_path):
    """With align=None an integer index must still read a 1-voxel slab, not a padded cell."""
    a = _src(tmp_path)
    seen = []

    def record(block, location):
        seen.append(tuple(location))
        return block

    plain = ops.map_overlap(a, record, depth=0, boundary="constant",
                            dtype="uint8", block_info=True)
    plain[70].compute()
    assert seen[0][0] == (70, 71), seen[0]
