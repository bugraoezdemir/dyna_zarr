"""write_levels chooses its regions by how often they decode each input read unit
(Workplan D9).

A region-wise read decodes every source tile it touches. Regions aligned to the output's
write units only (BARD, 2026-11-09: regions of 384 x 768 over 512 x 512 TIFF tiles) decode
each tile ~2.6 times; a shape within the same budget that meets the tiles' edges decodes
them fewer times. Small budgets and sources without a read unit are planned as before.
"""
import threading

import numpy as np
import pytest
import zarr

import dyna_zarr
from dyna_zarr import io
from dyna_zarr import operations as ops
from dyna_zarr.io import _decode_aware_region, _decodes_per_unit, _reads_per_unit


@pytest.mark.parametrize("region, unit, size, expect", [
    (384, 512, 1536, 2.0),          # every tile read by two regions
    (768, 512, 1536, 4 / 3),        # the middle tile of three by two
    (1536, 512, 1536, 1.0),         # the common multiple: once
    (384, 512, 1000, 2.0),          # a partial last tile counts like the others
    (5, 1, 100, 1.0),               # a read unit of 1 (one plane per file): free
    (2000, 512, 1536, 1.0),         # the whole axis
    (512, 512, 1536, 1.0),
])
def test_reads_per_unit(region, unit, size, expect):
    assert _reads_per_unit(region, unit, size) == pytest.approx(expect)


def test_decodes_multiply_over_the_axes():
    unit, shape = (1, 1, 512, 512), (1, 1920, 14040, 24551)
    assert _decodes_per_unit((1, 384, 384, 768), unit, shape) == pytest.approx(2.62, abs=0.01)
    assert _decodes_per_unit((1, 384, 1536, 1536), unit, shape) == pytest.approx(1.0)


BARD = (1, 1920, 14040, 24551)
TILES = (1, 1, 512, 512)
ALIGNED = (1, 384, 384, 768)       # level 0 + 1 in one pass: twice his (192, 192, 384) shards


def _within(mb):
    return lambda r: int(np.prod(r, dtype=np.int64)) * 2 <= mb * 2**20      # uint16; int64: numpy<2 on Windows multiplies in int32


@pytest.mark.parametrize("mb, expect, decodes", [
    (256, ALIGNED, 2.62),                     # the budget holds only the aligned region
    (500, (1, 384, 768, 768), 1.76),
    (1000, (1, 384, 768, 1536), 1.32),
    (2000, (1, 384, 1536, 1536), 1.0),        # every tile decoded once
])
def test_the_bard_layout(mb, expect, decodes):
    today = ALIGNED                           # (what grown() gives is passed in; here
    region, got, before = _decode_aware_region(ALIGNED, BARD, TILES, _within(mb), today)
    assert region == expect
    assert got == pytest.approx(decodes, abs=0.01)
    assert before == pytest.approx(2.62, abs=0.01)


def test_today_stays_when_it_decodes_as_few_times():
    """A tie keeps today's (larger) region: (1, 384, 384, 1536) decodes as often as
    today's (1, 384, 384, 3072) when the rows cannot grow."""
    today = (1, 384, 384, 3072)               # grown x-first at 1000 MB: ~2.0
    fits = lambda r: r[2] == 384 and _within(1000)(r)
    region, got, before = _decode_aware_region(ALIGNED, BARD, TILES, fits, today)
    assert region == today and got == before


@pytest.mark.parametrize("unit", [None, (1, 1, 1, 1), (1, 512, 512)])
def test_no_read_unit_plans_as_before(unit):
    region, _, _ = _decode_aware_region(ALIGNED, BARD, unit, _within(2000), ALIGNED)
    assert region == ALIGNED


def test_nothing_fits_but_the_aligned_region():
    region, _, _ = _decode_aware_region(ALIGNED, BARD, TILES, lambda r: False, ALIGNED)
    assert region == ALIGNED


class TiledPlane:
    """A micro-reader-like one-plane file with square tiles: a read decodes every tile
    it touches (counted)."""

    def __init__(self, data, tile, counter):
        self.data, self.tile, self.counter = data, tile, counter
        self.shape, self.dtype = data.shape, data.dtype
        self.ndim = data.ndim
        self.read_unit = (1,) * (data.ndim - 2) + (tile, tile)
        self.chunks = self.read_unit

    def __getitem__(self, key):
        ys, xs = key[-2], key[-1]
        t = self.tile
        n = ((ys.stop - 1) // t - ys.start // t + 1) * ((xs.stop - 1) // t - xs.start // t + 1)
        with self.counter["lock"]:
            self.counter["n"] += n
        return self.data[key]


class InterleavedTiles(TiledPlane):
    """An RGB slide's JPEG tiles: one tile holds every channel (read unit (3, t, t)); a
    read of any channel decodes the whole tile."""

    def __init__(self, data, tile, counter):
        super().__init__(data, tile, counter)
        self.read_unit = (data.shape[0], tile, tile)
        self.chunks = self.read_unit


@pytest.mark.parametrize("aware, levels, per_tile", [("1", 2, 1.0), ("0", 3, 3.0)])
def test_fewer_levels_in_a_pass_keep_interleaved_tiles_whole(tmp_path, monkeypatch, aware,
                                                            levels, per_tile):
    """3 levels in one pass need regions on a 128 x 128 grid, one channel deep within the
    budget: each tile decoded once per channel. 2 levels (64 x 64 grid) fit all 3
    channels: each tile decoded once; the third level is the caller's next pass."""
    monkeypatch.setenv("DYNA_ZARR_DECODE_AWARE_REGIONS", aware)
    rng = np.random.default_rng(2)
    data = rng.integers(0, 255, (3, 256, 256), dtype="uint8")
    counter = {"n": 0, "lock": threading.Lock()}
    src = dyna_zarr.from_array(InterleavedTiles(data, 32, counter))
    outs = [{"path": str(tmp_path / f"{k}.zarr"), "factors": (1, 2 ** k, 2 ** k),
             "chunks": (1, 32, 32)} for k in range(3)]
    n = io.write_levels(src, outs, region_size_mb=128 * 128 / 2**20, zarr_format=2,
                        max_threads=2)
    assert n == levels
    assert counter["n"] == pytest.approx(64 * per_tile)        # 8 x 8 tiles
    np.testing.assert_array_equal(zarr.open(outs[0]["path"], mode="r")[:], data)
    np.testing.assert_array_equal(zarr.open(outs[1]["path"], mode="r")[:],
                                  data[:, ::2, ::2])


def _source(planes, counter, tile=32):
    return ops.squeeze(ops.concatenate(
        [dyna_zarr.from_array(TiledPlane(p[None, None, None], tile, counter))
         for p in planes], axis=2))


@pytest.mark.parametrize("aware, regions_decode", [("1", 16 / 9), ("0", 2.0)])
def test_write_levels_decodes_each_tile_fewer_times(tmp_path, monkeypatch, aware,
                                                    regions_decode):
    """8 planes of 96 x 96 in 32 x 32 tiles, written in (8, 24, 24) chunks with a region
    budget of 2304 pixels per plane: memory-first regions (8, 24, 96) read every tile
    twice; (8, 48, 48) 16/9 times."""
    monkeypatch.setenv("DYNA_ZARR_DECODE_AWARE_REGIONS", aware)
    rng = np.random.default_rng(1)
    planes = [rng.integers(0, 255, (96, 96), dtype="uint8") for _ in range(8)]
    counter = {"n": 0, "lock": threading.Lock()}
    src = _source(planes, counter)
    out = str(tmp_path / "out.zarr")
    io.write_levels(src, [{"path": out, "factors": (1, 1, 1), "chunks": (8, 24, 24)}],
                    region_size_mb=8 * 48 * 48 / 2**20, zarr_format=2, max_threads=2)
    assert counter["n"] == pytest.approx(8 * 9 * regions_decode)
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], np.stack(planes))
