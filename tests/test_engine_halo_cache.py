"""A write over a filter decodes each source chunk about once (Workplan DR6, 2026-11-25).

Found by the DR6 quiet-machine benchmarks: a Gaussian downscale through dyna
(gaussian_filter, then every 2nd pixel) was 1.3 to 3 times slower with the engine than on
the Python path. Two causes, both measured on real pyramids:

- the stepped slice over the filter read it in blocks of one source chunk, and every
  block read and filtered its own halo (one region: 63 ms where the filter alone took 36);
- regions decoded the chunks of their halo again for every region touching them (2.5 to
  3.5 decodes per chunk), as the engine had no chunk cache.

Now views over a filter read it in as few pieces as their limits allow
(``Source::reads_halo``), and a write over one gives its engine pool a cache of decoded
chunks, counted in its memory budget.
"""
import os
import sys

import numpy as np
import pytest
import zarr

from dyna_zarr import _engine, _scope, io
from dyna_zarr.operations.neighborhood import gaussian_filter

sys.path.insert(0, os.path.dirname(__file__))
from test_write_threads import RUNTIME_MB, _run  # noqa: E402  (fresh-process helpers)

pytestmark = pytest.mark.engine

SIGMA = (0.0, 2 / 3, 2 / 3)


@pytest.fixture(autouse=True)
def engine_on(monkeypatch):
    if not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", True)


def _source(path, sharded):
    kw = {"shards": (4, 256, 256), "chunks": (4, 64, 64)} if sharded else {"chunks": (4, 96, 128)}
    z = zarr.create_array(str(path), shape=(8, 600, 700), dtype="uint16", **kw)
    rng = np.random.default_rng(3)
    z[:] = rng.integers(0, 4000, (8, 600, 700)).astype("uint16")
    return path


@pytest.fixture(scope="module", params=[False, True], ids=["chunks", "sharded"])
def source(request, tmp_path_factory):
    return _source(tmp_path_factory.mktemp("halo") / "s.zarr", request.param), request.param


def _units(sharded):
    unit = (4, 64, 64) if sharded else (4, 96, 128)
    return int(np.prod([-(-s // u) for s, u in zip((8, 600, 700), unit)]))


def _pools(monkeypatch):
    seen = []
    orig = _scope.WriteScope.pool

    def pool(self):
        p = orig(self)
        if all(p is not q for q in seen):
            seen.append(p)
        return p
    monkeypatch.setattr(_scope.WriteScope, "pool", pool)
    return seen


@pytest.mark.parametrize("threads", [1, 4])
def test_a_gaussian_downscale_write_decodes_each_chunk_once(source, tmp_path, monkeypatch,
                                                            capsys, threads):
    path, sharded = source
    pools = _pools(monkeypatch)
    v = gaussian_filter(io.read(str(path)), SIGMA, boundary="reflect")[:, ::2, ::2]
    io.write(v, str(tmp_path / "o"), max_threads=threads, region_size_mb=0.25)
    assert "chunk cache" in capsys.readouterr().out
    assert len(pools) == 1 and pools[0].cache_bytes > 0
    hits, decodes = pools[0].cache_counts()
    assert decodes == _units(sharded), (hits, decodes)
    assert hits > 0
    _engine.set_engine(False)
    want = np.asarray(v.compute())
    np.testing.assert_array_equal(np.asarray(zarr.open_array(str(tmp_path / "o"))[:]), want)


def test_a_copy_has_no_cache(source, tmp_path, monkeypatch, capsys):
    """Only chains that read a halo get one: a copy or a stride decodes each chunk once
    anyway, and keeps decoding straight into its destination."""
    path, _ = source
    a = io.read(str(path))
    from dyna_zarr import _lowering
    assert not _lowering.lowered(a[:, ::2, ::2]).reads_halo
    assert _lowering.lowered(gaussian_filter(a, SIGMA)[:, ::2, ::2]).reads_halo
    io.write(a[:, ::2, ::2], str(tmp_path / "o"), max_threads=4)
    assert "chunk cache" not in capsys.readouterr().out


def test_a_stepped_slice_reads_a_filter_whole(source):
    """One read of a strided filter decodes every chunk of its box once (before: each
    block of one chunk read its own halo, up to 9 decodes per chunk in 2-D)."""
    core = pytest.importorskip("dyna_zarr_core")
    path, sharded = source
    from dyna_zarr import _lowering
    v = _lowering.lowered(gaussian_filter(io.read(str(path)), SIGMA, boundary="reflect"))
    s = v.index((slice(None), slice(None, None, 2), slice(None, None, 2)))
    pool = core.Pool(4, cache_mb=256)
    got = s.read([0, 0, 0], list(s.shape), threads=4, pool=pool)
    assert pool.cache_counts() == (0, _units(sharded))
    plain = s.read([0, 0, 0], list(s.shape), threads=4)       # no pool: no cache
    np.testing.assert_array_equal(got, plain)


def test_cached_regions_are_the_values_of_direct_reads(source):
    """Through the cache, any region gives the same values as a direct read: edges,
    regions inside one unit, regions over several."""
    core = pytest.importorskip("dyna_zarr_core")
    path, _ = source
    a = core.Array.open(str(path))
    pool = core.Pool(3, cache_mb=8)                  # small: units are dropped and decoded again
    rng = np.random.default_rng(5)
    for _ in range(40):
        start = [int(rng.integers(0, s)) for s in a.shape]
        stop = [int(rng.integers(b + 1, s + 1)) for b, s in zip(start, a.shape)]
        np.testing.assert_array_equal(a.read(start, stop, threads=3, pool=pool),
                                      a.read(start, stop, threads=3))
    hits, decodes = pool.cache_counts()
    assert hits > 0 and decodes > 0


def test_the_write_with_its_cache_stays_within_the_budget(tmp_path):
    """The cache gets what memory_budget_mb leaves once the regions are planned."""
    src = _source(tmp_path / "s.zarr", False)
    budget_mb = 40
    grew = _run("""
        import json, sys
        from dyna_zarr import _engine, io
        from dyna_zarr.operations.neighborhood import gaussian_filter
        _engine.set_engine(True)
        v = gaussian_filter(io.read(sys.argv[1]), (0.0, 2 / 3, 2 / 3))[:, ::2, ::2]
        io.write(v[:, :64, :64], sys.argv[2] + "/warm", max_threads=8)
        before, _ = rss_and_peak()
        io.write(v, sys.argv[2] + "/o", max_threads=8, memory_budget_mb=float(sys.argv[3]),
                 region_size_mb=0.25)
        _, peak = rss_and_peak()
        print(json.dumps((peak - before) / 2**20))
    """, src, tmp_path, budget_mb)
    assert grew <= budget_mb + RUNTIME_MB, grew


def test_without_a_budget_the_cache_is_capped(source):
    """Without memory_budget_mb a write's cache takes at most io._HALO_CACHE_MAX (64 MiB,
    measured 2026-11-27: as fast as the whole cache a Gaussian downscale asks for, with
    90-220 MB less at 16 threads); with a budget, what the plan leaves (``room``)."""
    path, _ = source
    mod = sys.modules["dyna_zarr.io"]
    v = gaussian_filter(io.read(str(path)), SIGMA)[:, ::2, ::2]
    big = mod._halo_cache_bytes(v, (8, 300, 350), 100, 16, 1 << 62)   # uncapped
    assert big > mod._HALO_CACHE_MAX == 64 << 20
    assert mod._halo_cache_bytes(v, (8, 300, 350), 100, 16, None) == mod._HALO_CACHE_MAX
    assert mod._halo_cache_bytes(v, (8, 300, 350), 100, 16, 10 << 20) == 10 << 20
    assert mod._halo_cache_bytes(v, (8, 300, 350), 100, 16, 0) == 0
    small = mod._halo_cache_bytes(v, (4, 64, 64), 2, 1, 1 << 62)
    assert 0 < small < mod._HALO_CACHE_MAX
    assert mod._halo_cache_bytes(v, (4, 64, 64), 2, 1, None) == small


@pytest.mark.parametrize("budget_mb, whole", [(2000, True), (12, False)])
def test_the_budget_plans_the_regions_first_then_the_cache(source, tmp_path, monkeypatch,
                                                          budget_mb, whole):
    """Bugra 2026-11-26: the fast path when it fits the budget, a smaller cache when it
    does not. The regions and threads are planned with the whole budget, as without a
    cache; the cache gets what that plan leaves, whole when it fits (2000 MB), smaller
    when the budget is tight (12 MB), never at the threads' expense."""
    path, sharded = source
    if sharded:
        pytest.skip("written chunk by chunk through write_levels' tiles: the next test")
    mod = sys.modules["dyna_zarr.io"]
    plans, sizes = [], []
    plan0, size0 = mod._plan_regions, mod._halo_cache_bytes

    def plan(*a):
        r = plan0(*a)
        plans.append((a[4], r))
        return r

    def size(array, extent, along_last, threads, room):
        got = size0(array, extent, along_last, threads, room)
        sizes.append((room, got, size0(array, extent, along_last, threads, 1 << 62)))
        return got
    monkeypatch.setattr(mod, "_plan_regions", plan)
    monkeypatch.setattr(mod, "_halo_cache_bytes", size)
    v = gaussian_filter(io.read(str(path)), SIGMA, boundary="reflect")[:, ::2, ::2]
    io.write(v, str(tmp_path / "o"), max_threads=8, region_size_mb=0.25,
             memory_budget_mb=budget_mb)
    (budget, (workers, _, region_bytes, fits)), = plans
    assert budget == budget_mb * 2**20 and fits     # the whole budget, not minus a cache
    (room, got, want), = sizes
    assert room == budget - workers * region_bytes
    assert (got == want) if whole else (0 < got == int(room) < want), (room, got, want)
    _engine.set_engine(False)
    np.testing.assert_array_equal(np.asarray(zarr.open_array(str(tmp_path / "o"))[:]),
                                  np.asarray(v.compute()))


@pytest.mark.parametrize("budget_mb", [2000, 12])
def test_the_tiled_pass_plans_the_tiles_first_then_the_cache(tmp_path, monkeypatch, budget_mb):
    """The same rule in write_levels' tiled pass (a sharded output written chunk by
    chunk): the tiles are planned with the whole budget and smaller shares of it; the
    plan with the most levels per pass and then the most threads wins (a whole cache
    beside it only breaks ties), and the cache gets at most what that plan leaves."""
    src = _source(tmp_path / "s.zarr", True)
    mod = sys.modules["dyna_zarr.io"]
    plans, rooms = [], []
    plan0, size0 = mod._plan_xy_tiles, mod._halo_cache_bytes

    def plan(*a, **k):
        r = plan0(*a, **k)
        plans.append((a[5], r))
        return r

    def size(array, extent, along_last, threads, room):
        if room != 1 << 62:                         # not the planner sizing a candidate
            rooms.append((threads, room))
        return size0(array, extent, along_last, threads, room)
    monkeypatch.setattr(mod, "_plan_xy_tiles", plan)
    monkeypatch.setattr(mod, "_halo_cache_bytes", size)
    v = gaussian_filter(io.read(str(src)), SIGMA, boundary="reflect")[:, ::2, ::2]
    io.write(v, str(tmp_path / "o"), max_threads=8, region_size_mb=0.25,
             memory_budget_mb=budget_mb)
    budget = budget_mb * 2**20
    assert plans[0][0] == budget                    # the whole budget first
    fit = [p for _, p in plans if p["fits"]]
    most_n = max(p["n"] for p in fit)
    (threads, room), = rooms
    # the threads are not cut for the cache
    assert threads == max(p["workers"] for p in fit if p["n"] == most_n)
    kept = [p for p in fit if (p["n"], p["workers"]) == (most_n, threads)]
    assert room is not None and room + min(p["chunks"] + p["blocks"] for p in kept) <= budget + 1
    _engine.set_engine(False)
    np.testing.assert_array_equal(np.asarray(zarr.open_array(str(tmp_path / "o"))[:]),
                                  np.asarray(v.compute()))
