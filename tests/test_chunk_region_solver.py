"""The chunk/region contract.

The one rule under test: **exact forms are honoured verbatim, size forms are
approximate and may be adjusted.** Plus the alignment rule that is never traded
away, since violating it is a data race rather than a slowdown:

    a region always covers a WHOLE number of output chunks.

Most of these run the solver directly rather than through a write, because the
structural decisions are deterministic while write timings on this path vary up
to 2.4x run to run (see project-dyna-write-timing-variance). The end-to-end
tests at the bottom check the solver's choices actually reach the store.
"""

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io
from dyna_zarr.io import _solve_grids, _DEFAULT_REGION_MB


F32 = "float32"


def _mb(shape, dtype=F32):
    return float(np.prod(shape)) * np.dtype(dtype).itemsize / (1024 * 1024)


def assert_region_tiles_chunks(region, chunk, shape):
    """Rule 1: no two concurrent region writers may share an output chunk.

    An axis the region spans WHOLE is exempt: there is a single region on it, so
    there is no second writer to race with.
    """
    for axis, (r, c, s) in enumerate(zip(region, chunk, shape)):
        assert r % c == 0 or r >= s, (
            f"axis {axis}: region {r} is not a multiple of chunk {c} "
            f"(region={region}, chunk={chunk}, shape={shape}) -- concurrent "
            f"writers would share a chunk"
        )


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

def test_default_preserves_input_chunks():
    """Nothing supplied means the OUTPUT KEEPS THE INPUT GRID.

    Not a computed shape. The caller always knows their own input chunks, so a
    round trip is lossless and the result is predictable without reading docs.
    """
    chunk, region, _ = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32)
    assert chunk == (8, 256, 256)


def test_default_region_is_8mb():
    chunk, region, _ = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32)
    assert _mb(region) <= _DEFAULT_REGION_MB * 1.5
    assert_region_tiles_chunks(region, chunk, (512, 1024, 1024))


def test_default_with_no_source_grid_still_bounded():
    """A TensorStore source, or a transform that dropped the grid, gives None.

    There is then nothing to preserve, so the chunk is sized from the budget --
    and it must respect that budget rather than fall back to a fixed shape.
    """
    chunk, region, _ = _solve_grids((512, 1024, 1024), None, F32, F32)
    assert _mb(chunk) <= 1.5, f"{chunk} is {_mb(chunk):.2f} MiB"
    assert_region_tiles_chunks(region, chunk, (512, 1024, 1024))


# --------------------------------------------------------------------------
# Exact forms are honoured verbatim
# --------------------------------------------------------------------------

@pytest.mark.parametrize("exact", [(8, 256, 256), (4, 64, 64), (1, 1024, 1024)])
def test_exact_chunks_honoured(exact):
    chunk, _, _ = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32,
                               chunks=exact)
    assert chunk == exact


@pytest.mark.parametrize("exact", [(16, 512, 512), (8, 256, 256), (32, 1024, 1024)])
def test_exact_region_honoured(exact):
    _, region, _ = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32,
                                region_shape=exact)
    assert region == exact


def test_both_exact_honoured_verbatim():
    """Neither side bends when both are given: the caller has decided."""
    chunk, region, _ = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32,
                                    chunks=(8, 256, 256),
                                    region_shape=(16, 512, 512))
    assert chunk == (8, 256, 256)
    assert region == (16, 512, 512)


def test_exact_chunks_beat_a_tiny_region_budget():
    """The region grows past its budget rather than split a chunk.

    Rule 1 is correctness; the budget is a target. Peak memory still does not
    scale with array size, which is the guarantee that matters.
    """
    chunk, region, notes = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32,
                                        chunks=(8, 256, 256), region_mb=0.5)
    assert chunk == (8, 256, 256)
    assert region == (8, 256, 256)
    assert_region_tiles_chunks(region, chunk, (512, 1024, 1024))
    assert any("region" in n and "chunk" in n for n in notes), notes


# --------------------------------------------------------------------------
# Size forms are approximate but honest
# --------------------------------------------------------------------------

@pytest.mark.parametrize("target", [0.25, 1.0, 4.0])
def test_chunk_budget_is_respected(target):
    chunk, _, _ = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32,
                               chunk_mb=target)
    assert _mb(chunk) <= target * 1.5, f"{chunk} is {_mb(chunk):.2f} MiB > {target}"


@pytest.mark.parametrize("target", [1.0, 8.0, 64.0])
def test_region_budget_is_respected(target):
    shape, src = (512, 1024, 1024), (8, 256, 256)
    chunk, region, _ = _solve_grids(shape, src, F32, F32, chunk_mb=1.0,
                                    region_mb=target)
    assert _mb(region) <= target * 1.5
    assert_region_tiles_chunks(region, chunk, shape)


def test_a_missed_budget_is_reported():
    """Silence on a miss is the bug: the path that HONOURED a request used to
    warn while the path that quietly returned a quarter of it said nothing."""
    _, _, notes = _solve_grids((1_000_000,), (65536,), F32, F32,
                               chunk_mb=1.0, region_mb=8.0)
    assert notes, "a budget was missed with no note"


def test_size_form_opts_out_of_preserving_the_input_grid():
    """Asking for a size is asking for a different grid, so preservation stops."""
    chunk, _, _ = _solve_grids((512, 1024, 1024), (8, 256, 256), F32, F32,
                               chunk_mb=0.25)
    assert chunk != (8, 256, 256)
    assert _mb(chunk) <= 0.375


# --------------------------------------------------------------------------
# Contradictions raise
# --------------------------------------------------------------------------

@pytest.fixture
def small_source(tmp_path):
    path = tmp_path / "src.zarr"
    z = zarr.create_array(store=str(path), shape=(32, 128, 128),
                          chunks=(8, 64, 64), dtype="float32")
    z[:] = np.random.rand(32, 128, 128).astype("float32")
    return DynamicArray(z)


def test_chunks_and_chunk_size_mb_together_raise(small_source, tmp_path):
    with pytest.raises(ValueError, match="not both"):
        io.write(small_source, str(tmp_path / "o.zarr"),
                 chunks=(8, 64, 64), chunk_size_mb=1.0)


def test_region_shape_and_region_size_mb_together_raise(small_source, tmp_path):
    with pytest.raises(ValueError, match="not both"):
        io.write(small_source, str(tmp_path / "o.zarr"),
                 region_shape=(8, 64, 64), region_size_mb=1.0)


@pytest.mark.parametrize("kwargs", [{"chunk_size_mb": 0}, {"chunk_size_mb": -1},
                                    {"region_size_mb": 0}, {"region_size_mb": -1}])
def test_non_positive_budgets_raise(small_source, tmp_path, kwargs):
    with pytest.raises(ValueError, match="must be > 0"):
        io.write(small_source, str(tmp_path / "o.zarr"), **kwargs)


def test_misaligned_exact_region_raises_with_both_remedies(small_source, tmp_path):
    """The caller pinned both grids and they do not tile. Neither is silently
    rounded: rounding up multiplies peak memory, rounding down destroys the
    alignment they were usually after. Both valid shapes are offered instead."""
    with pytest.raises(ValueError) as exc:
        io.write(small_source, str(tmp_path / "o.zarr"),
                 chunks=(8, 64, 64), region_shape=(8, 100, 100))
    msg = str(exc.value)
    assert "multiple" in msg
    assert "rounding up" in msg and "rounding down" in msg


def test_wrong_rank_region_shape_raises(small_source, tmp_path):
    with pytest.raises(ValueError, match="dims but array is"):
        io.write(small_source, str(tmp_path / "o.zarr"), region_shape=(64, 64))


# --------------------------------------------------------------------------
# Rule 1 holds across the whole matrix, including pathological sources
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape,src", [
    ((512, 1024, 1024), (8, 256, 256)),      # typical
    ((256, 512, 512), (13, 127, 251)),       # prime source chunks
    ((500, 500, 500), (7, 13, 29)),          # coprime
    ((2, 3, 48, 384, 384), (1, 1, 8, 128, 128)),   # 5D with short outer axes
    ((512, 1024, 1024), (64, 512, 512)),     # source chunk over the budget
    ((1_000_000,), (65536,)),                # 1D
    ((7,), (3,)),                            # degenerate
    ((1, 1, 1), (1, 1, 1)),                  # minimal
])
@pytest.mark.parametrize("chunk_mb,region_mb", [
    (None, None), (0.25, 1.0), (1.0, 8.0), (4.0, 64.0),
])
def test_region_always_tiles_chunks(shape, src, chunk_mb, region_mb):
    """The invariant that is never traded, over every source and budget pair."""
    chunk, region, _ = _solve_grids(shape, src, F32, F32,
                                    chunk_mb=chunk_mb, region_mb=region_mb)
    assert len(chunk) == len(shape) and len(region) == len(shape)
    assert all(1 <= c <= s for c, s in zip(chunk, shape))
    assert all(1 <= r <= s for r, s in zip(region, shape))
    assert_region_tiles_chunks(region, chunk, shape)


def test_a_tiny_source_grid_is_preserved_but_reported():
    """Preserving (7,13,29) means millions of files. It is still preserved --
    the caller asked for it by saying nothing -- but never silently."""
    chunk, _, notes = _solve_grids((500, 500, 500), (7, 13, 29), F32, F32)
    assert chunk == (7, 13, 29)
    assert any("KiB" in n for n in notes), notes


def test_a_huge_source_grid_is_preserved_but_reported():
    chunk, region, notes = _solve_grids((512, 1024, 1024), (64, 512, 512), F32, F32)
    assert chunk == (64, 512, 512)
    assert any("floor" in n or "MiB each" in n for n in notes), notes


# --------------------------------------------------------------------------
# The solver does not settle for a degenerate shape
# --------------------------------------------------------------------------

def test_budgeted_chunk_is_not_a_thin_slab():
    """A chunk one voxel deep and a full row wide has the right byte count and
    the wrong shape: a Z-oriented read or a 3D halo then pulls a whole row per
    step. Regression for (1, 256, 1024) and (16, 16, 1024)."""
    shape, src = (512, 1024, 1024), (8, 256, 256)
    for region_mb in (1.0, 8.0, 64.0):
        chunk, _, _ = _solve_grids(shape, src, F32, F32, chunk_mb=1.0,
                                   region_mb=region_mb)
        free = [c for c, s in zip(chunk, shape) if c < s]
        if len(free) > 1:
            assert max(free) / min(free) <= 16, (
                f"chunk {chunk} at region_mb={region_mb} is a slab")


def test_budgeted_chunk_never_collapses_to_near_nothing():
    """A prime source grid once produced a 260-byte chunk: millions of files and
    a write that never finished."""
    chunk, _, _ = _solve_grids((256, 512, 512), (13, 127, 251), F32, F32,
                               chunk_mb=1.0)
    assert _mb(chunk) >= 1.0 / 8


# --------------------------------------------------------------------------
# End to end: the solver's choice reaches the store, and the data survives
# --------------------------------------------------------------------------

def test_roundtrip_preserves_grid_and_data(tmp_path):
    src_path, out_path = tmp_path / "s.zarr", tmp_path / "o.zarr"
    data = np.random.rand(32, 128, 128).astype("float32")
    z = zarr.create_array(store=str(src_path), shape=data.shape,
                          chunks=(8, 64, 64), dtype="float32")
    z[:] = data

    io.write(DynamicArray(z), str(out_path))

    out = zarr.open_array(str(out_path), mode="r")
    assert tuple(out.chunks) == (8, 64, 64), "the input grid was not preserved"
    np.testing.assert_allclose(out[:], data)


def test_explicit_chunks_reach_the_store(tmp_path):
    src_path, out_path = tmp_path / "s.zarr", tmp_path / "o.zarr"
    data = np.random.rand(32, 128, 128).astype("float32")
    z = zarr.create_array(store=str(src_path), shape=data.shape,
                          chunks=(8, 64, 64), dtype="float32")
    z[:] = data

    io.write(DynamicArray(z), str(out_path), chunks=(4, 32, 32))

    out = zarr.open_array(str(out_path), mode="r")
    assert tuple(out.chunks) == (4, 32, 32)
    np.testing.assert_allclose(out[:], data)


def test_chunk_size_mb_reaches_the_store_and_is_near_budget(tmp_path):
    src_path, out_path = tmp_path / "s.zarr", tmp_path / "o.zarr"
    data = np.random.rand(64, 256, 256).astype("float32")
    z = zarr.create_array(store=str(src_path), shape=data.shape,
                          chunks=(8, 64, 64), dtype="float32")
    z[:] = data

    io.write(DynamicArray(z), str(out_path), chunk_size_mb=0.25)

    out = zarr.open_array(str(out_path), mode="r")
    assert _mb(out.chunks) <= 0.375
    np.testing.assert_allclose(out[:], data)
