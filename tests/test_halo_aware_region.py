"""The region is shaped against the HALO, not only against the I/O grid.

A neighbourhood op reads its region plus ``depth`` on every side, so two regions
of identical size can cost very different amounts of work: at depth 8, an axis
8 deep triples while one 1024 wide grows 1.6%. Shaping the region for I/O alone
made a default gaussian write 2.3x slower than the same byte budget spent on a
squarer region (16.8s -> 7.4s measured on a 2 GB float32 array, at a LOWER peak).

The solver reaches the depth by walking the chain, the same walk it already does
for dtype -- the caller is not in a position to know it, and often is not even
the author of the chain (a Pyramid or tilewise-ccl hands one over).
"""

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations
from dyna_zarr.io import (_chain_halo_depth, _halo_amplification, _solve_grids)


SHAPE, SRC, DT = (512, 1024, 1024), (8, 256, 256), "float32"


# --------------------------------------------------------------------------
# Finding the depth
# --------------------------------------------------------------------------

def test_no_halo_in_chain_reports_none():
    data = np.zeros((8, 32, 32), dtype=np.float32)
    arr = DynamicArray(data)
    assert _chain_halo_depth(arr, 3) is None
    assert _chain_halo_depth(operations.add(arr, 1.0), 3) is None


def test_gaussian_depth_is_found():
    data = np.zeros((8, 32, 32), dtype=np.float32)
    got = _chain_halo_depth(operations.gaussian_filter(DynamicArray(data), 2.0), 3)
    assert got == (8, 8, 8), got


def test_depth_is_found_through_a_later_op():
    """The halo op is not always outermost."""
    data = np.zeros((8, 32, 32), dtype=np.float32)
    chain = operations.add(
        operations.gaussian_filter(DynamicArray(data), 2.0), 1.0)
    assert _chain_halo_depth(chain, 3) == (8, 8, 8)


def test_stacked_halos_take_the_widest_per_axis():
    data = np.zeros((8, 64, 64), dtype=np.float32)
    chain = operations.gaussian_filter(
        operations.gaussian_filter(DynamicArray(data), 1.0), 3.0)
    got = _chain_halo_depth(chain, 3)
    assert got == (12, 12, 12), got


def test_the_walk_survives_a_cycle():
    """It must degrade to 'no information', never hang or raise inside a write."""
    data = np.zeros((4, 8, 8), dtype=np.float32)
    arr = DynamicArray(data)
    arr._transform = type("T", (), {"array": arr, "depth": None})()
    assert _chain_halo_depth(arr, 3) is None


# --------------------------------------------------------------------------
# Amplification arithmetic
# --------------------------------------------------------------------------

def test_amplification_is_one_without_depth():
    assert _halo_amplification((8, 256, 1024), SHAPE, (0, 0, 0)) == 1.0


def test_an_axis_spanned_whole_pays_no_halo():
    """Nothing lies outside it to read, which is why completing a short axis is
    worth so much."""
    amp = _halo_amplification((512, 256, 256), SHAPE, (8, 8, 8))
    assert amp == _halo_amplification((512, 256, 256), SHAPE, (99, 8, 8))


def test_thin_axes_dominate_the_cost():
    thin = _halo_amplification((8, 256, 1024), SHAPE, (8, 8, 8))
    square = _halo_amplification((32, 256, 256), SHAPE, (8, 8, 8))
    assert thin > square, (thin, square)
    assert thin > 3.0 and square < 2.0


# --------------------------------------------------------------------------
# The solver uses it -- and only when there is a halo
# --------------------------------------------------------------------------

def test_no_halo_choice_is_unchanged():
    """The fix must cost nothing on a chain with no neighbourhood op."""
    chunk, region, _ = _solve_grids(SHAPE, SRC, DT, DT)
    assert chunk == (8, 256, 256)
    assert region == (8, 256, 1024)


def test_halo_moves_the_region_to_a_squarer_shape():
    _, plain, _ = _solve_grids(SHAPE, SRC, DT, DT)
    _, withh, _ = _solve_grids(SHAPE, SRC, DT, DT, halo=(8, 8, 8))
    assert withh != plain
    assert (_halo_amplification(withh, SHAPE, (8, 8, 8))
            < _halo_amplification(plain, SHAPE, (8, 8, 8)))


@pytest.mark.parametrize("depth", [(4, 4, 4), (8, 8, 8), (16, 16, 16),
                                   (1, 8, 8), (8, 1, 1)])
def test_halo_never_breaks_the_alignment_rule(depth):
    """Rule 1 is correctness and outranks every efficiency term."""
    chunk, region, _ = _solve_grids(SHAPE, SRC, DT, DT, halo=depth)
    for axis, (r, c, s) in enumerate(zip(region, chunk, SHAPE)):
        assert r % c == 0 or r >= s, (axis, region, chunk)


@pytest.mark.parametrize("region_mb", [8.0, 32.0, 128.0])
def test_halo_respects_the_region_budget(region_mb):
    """The halo reshapes the region; it never inflates it past the budget.

    (1.0 is excluded deliberately: the inherited chunk is 2 MiB, so the region
    must exceed that budget to hold one whole chunk -- rule 1 beating a target,
    covered by test_halo_cannot_shrink_below_one_chunk below.)
    """
    chunk, region, _ = _solve_grids(SHAPE, SRC, DT, DT, halo=(8, 8, 8),
                                    region_mb=region_mb)
    got = np.prod(region) * 4 / 1024 / 1024
    assert got <= region_mb * 1.5, f"{region} is {got:.2f} MiB"


def test_halo_cannot_shrink_below_one_chunk():
    """Rule 1 outranks the halo term as it outranks every efficiency term."""
    chunk, region, notes = _solve_grids(SHAPE, SRC, DT, DT, halo=(8, 8, 8),
                                        region_mb=1.0)
    assert region == chunk
    assert any("whole chunk" in n for n in notes), notes


def test_exact_region_still_wins_over_the_halo():
    """An exact form is honoured verbatim; the halo is an optimisation, and an
    optimisation never overrides what the caller actually asked for."""
    _, region, _ = _solve_grids(SHAPE, SRC, DT, DT, halo=(8, 8, 8),
                                region_shape=(8, 256, 1024))
    assert region == (8, 256, 1024)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_gaussian_write_is_correct_and_uses_a_squarer_region(tmp_path, capsys):
    src, out = tmp_path / "s.zarr", tmp_path / "o.zarr"
    data = np.random.rand(16, 64, 64).astype("float32")
    z = zarr.create_array(store=str(src), shape=data.shape, chunks=(4, 32, 32),
                          dtype="float32")
    z[:] = data

    io.write(operations.gaussian_filter(io.read(str(src)), 1.0), str(out))

    from scipy import ndimage
    expected = ndimage.gaussian_filter(data, 1.0, mode="reflect")
    np.testing.assert_allclose(zarr.open_array(str(out), mode="r")[:], expected,
                               rtol=1e-4, atol=1e-5)
