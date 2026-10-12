"""write_levels(memory_budget_mb=): y/x tiles for big shards appended chunk by chunk.

Untiled, the open output chunks are about one chunk row per level across the whole plane,
which grows with the plane. With a budget, the planner picks a y/x tile (z swept per tile,
Morton order) and how many levels tile within it; the deeper levels come from a stored
level in a second pass. Output must be identical to the untiled write.
"""
import importlib
import re
import warnings

import numpy as np
import pytest
import tensorstore as ts
import zarr

import dyna_zarr
from dyna_zarr.io import io

dio = importlib.import_module("dyna_zarr.io")
plan = dio._plan_xy_tiles


# --- the planner -----------------------------------------------------------------------

def _levels(shape, chunks, n):
    factors = [(2 ** k,) * 3 for k in range(n)]
    shapes = [tuple(-(-s // f) for s, f in zip(shape, fac)) for fac in factors]
    return factors, [chunks] * n, shapes


def _plan(budget_mb, n=3, unit=None, workers=4, shape=(64, 512, 512), chunks=(8, 32, 32),
          src_chunks=(1, 512, 512)):
    f, c, L = _levels(shape, chunks, n)
    nbytes = lambda b: int(np.prod(b))                 # uint8
    return plan(shape, f, c, L, 1, budget_mb * 2**20, unit, workers=workers,
                block_cost=lambda b: 3 * nbytes(b), read_cost=nbytes,
                region_bytes=4 * 2**20, src_chunks=src_chunks)


def test_the_whole_plane_when_it_fits():
    r = _plan(256)
    assert r["fits"] and r["n"] == 3 and r["tile"] == (512, 512) and r["workers"] == 4


def test_the_total_stays_within_a_tighter_budget():
    r = _plan(8)
    assert r["fits"] and r["chunks"] + r["blocks"] <= 8 * 2**20
    assert r["tile"] != (512, 512)


def test_full_width_tiles_come_first():
    r = _plan(8)
    assert r["tile"][1] == 512                         # rows of the plane, not squares


def test_several_tiles_go_slab_by_slab():
    # every tile of a slab before the next slab: a stack of files reads each plane in one
    # stretch; the slab is a multiple of level 0's chunk depth (8) and the windows (4)
    r = _plan(8)
    assert r["tile"] != (512, 512) and r["slab"] < 64 and r["slab"] % 8 == 0
    assert _plan(256)["slab"] == 64                     # one tile: the whole depth


def test_small_deep_levels_are_held_whole_not_deferred():
    # level 3's chunk footprint (8 x 32 = 256) would force tiles of 256 rows; the level
    # itself is 32 KiB, so it is held whole and written in the same pass, with all
    # workers, in full-width tiles of 64 rows
    r = _plan(7, n=4)
    assert r["fits"] and r["workers"] == 4 and r["n"] == 4 and r["tile"] == (64, 512)


def test_below_the_smallest_pass_says_so():
    r = _plan(0.01)
    assert not r["fits"] and r["need_minimal"] > 0.01 * 2**20


def test_source_chunks_coprime_with_the_window_do_not_inflate_blocks():
    # stored level 0 with 101-deep chunks under a window of 16: aligning the block to
    # both made the smallest block 1616 planes deep (12 GB at 1616 x 1616)
    r = _plan(64, n=5, shape=(2347, 1616, 1616), chunks=(101, 101, 101),
              src_chunks=(101, 101, 101))
    assert r["fits"] and r["bz"] <= 64 and r["chunks"] + r["blocks"] <= 64 * 2**20


def test_a_whole_plane_unit_split_by_tiles_is_reported():
    r = _plan(8, unit=(1, 512, 512))
    assert r["amp"] > 1 and r["need_units"] > r["chunks"] + r["blocks"]


# --- end to end (zarrista-bounded) ---------------------------------------------------------

SHAPE = (64, 512, 512)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    return rng.integers(0, 250, size=SHAPE, dtype="uint8")


def _expected(data, f, method):
    if method == "simple":
        return data[tuple(slice(None, None, k) for k in f)]
    return ts.downsample(ts.array(data), list(f), method=method).read().result()


@pytest.mark.zarrista
@pytest.mark.parametrize("method", ["simple", "mean", "mode"])
def test_tiled_write_matches_and_stays_within_the_estimate(data, tmp_path, method):
    if not dio._bounded_partial_writes():
        pytest.skip("needs zarrista-bounded (in-place shard appends)")
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=(8, 64, 64),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    factors = [(1, 1, 1), (2, 2, 2), (4, 4, 4), (8, 8, 8)]
    outs = [dict(path=str(tmp_path / f"L{k}.zarr"), factors=f, chunks=(8, 32, 32),
                 shard_coefficients=(2, 4, 4)) for k, f in enumerate(factors)]
    done, src, src_f, notes = 0, io.read(str(tmp_path / "src.zarr")), (1, 1, 1), []
    while done < len(outs):
        rel = [dict(o, factors=tuple(a // b for a, b in zip(o["factors"], src_f)))
               for o in outs[done:]]
        with io.capture_messages() as messages:
            n = io.write_levels(src, rel, method=method, region_size_mb=0.05,
                                zarr_format=3, backend="zarrista", max_threads=4,
                                memory_budget_mb=4)
        notes += messages
        done += n
        src, src_f = io.read(outs[0]["path"]), (1, 1, 1)          # mean/mode: from L0
    for o in outs:
        np.testing.assert_array_equal(zarr.open_array(o["path"])[...],
                                      _expected(data, o["factors"], method))
    tiled = [m for m in notes if "tiles of" in m]
    assert tiled, notes
    est = float(re.search(r"open chunks ([0-9.]+) \+", tiled[0]).group(1))
    peak = float(re.search(r"peaked at ([0-9.]+) MiB",
                           next(m for m in notes if "peaked at" in m)).group(1))
    assert peak <= est + 0.5, (peak, est)                # estimate printed in whole MiB


class _PlaneSource:
    """Planes read whole for any piece (one compressed strip per plane)."""

    def __init__(self, data):
        self.data, self.shape, self.dtype = data, data.shape, data.dtype
        self.ndim = data.ndim
        self.chunks = (1,) + data.shape[1:]
        self.read_unit = (1,) + data.shape[1:]

    def __getitem__(self, key):
        return self.data[key]


@pytest.mark.zarrista
def test_splitting_whole_plane_units_warns_with_the_budget_that_avoids_it(data, tmp_path):
    if not dio._bounded_partial_writes():
        pytest.skip("needs zarrista-bounded (in-place shard appends)")
    src = dyna_zarr.from_array(_PlaneSource(data))
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 32, 32),
                 shard_coefficients=(2, 4, 4))]
    with pytest.warns(RuntimeWarning, match=r"decoded ~\d+x.*at least \d+ MiB"):
        io.write_levels(src, outs, region_size_mb=0.05, zarr_format=3, backend="zarrista",
                        memory_budget_mb=1.5)
    np.testing.assert_array_equal(zarr.open_array(outs[0]["path"])[...], data)


def test_the_budget_is_what_the_write_adds_to_the_process(data, tmp_path):
    # as in io.write: whatever the process already holds, the same budget plans the same
    # tiles (it used to subtract the process's RSS, so a big process got tiny tiles)
    ballast = np.ones(256 * 2**20, dtype="uint8")          # 256 MiB the write does not own
    if not dio._bounded_partial_writes():
        pytest.skip("needs zarrista-bounded (in-place shard appends)")
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=(8, 64, 64),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 32, 32),
                 shard_coefficients=(2, 4, 4))]
    with io.capture_messages() as messages:
        io.write_levels(io.read(str(tmp_path / "src.zarr")), outs, region_size_mb=0.05,
                        zarr_format=3, backend="zarrista", memory_budget_mb=4)
    assert ballast.sum() and any("tiles of" in m and "of 4" in m for m in messages), messages


def test_the_budget_caps_the_threads_when_shards_fit_a_region(data, tmp_path):
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 64, 64))]
    with io.capture_messages() as messages:
        io.write_levels(dyna_zarr.from_array(data), outs, zarr_format=2, max_threads=8,
                        region_size_mb=1, memory_budget_mb=2.5)
    line = next(m for m in messages if "outputs in one pass" in m)
    assert re.search(r", ([12]) threads", line), line
    np.testing.assert_array_equal(zarr.open_array(outs[0]["path"])[...], data)


def test_full_width_with_fewer_threads_before_narrower_tiles():
    # a narrower tile reads each row in pieces, once per tile across (2.6x slower on a
    # 2916-wide stack): full width with fewer threads wins, down to half the threads
    full4 = _plan(256)
    budget = None
    for mb in [x / 4 for x in range(4, 64)]:               # the tightest budgets first
        r = _plan(mb)
        if r["fits"] and r["tile"][1] == 512:
            budget = mb
            break
    assert budget is not None and full4["tile"][1] == 512
    r = _plan(budget)
    assert r["tile"][1] == 512 and r["workers"] >= 2         # never narrower at >= half


def test_a_row_spanning_slabs_is_counted_once_across_the_plane():
    # level 1's chunk rows (2 x 8 = 16 source planes) span two slabs of 8: one row open
    # across the plane, the next only over the tiles reached - not two whole-plane rows
    r = _plan(8)
    assert r["slab"] < 64
    f, c, L = _levels((64, 512, 512), (8, 32, 32), 3)
    plane_rows_2 = 2 * 8 * L[1][1] * L[1][2]                 # the old count, level 1 alone
    assert r["chunks"] < plane_rows_2 + 2 * 8 * 512 * 512    # + level 0 rows at most
