"""write_levels keeps memory_budget_mb with the engine too (decision DR-5), and views over
a sharded array read in blocks of its inner chunk, not of its shard.

Found by the DR6 quiet-machine benchmarks (2026-11-25): a strided copy of a sharded
array (shards of 8 MiB, inner chunks of 1 MiB) at 16 threads grew the process by 277 MB
with the engine against 103 MB on the Python path. The engine's own buffers peaked at
142 MiB: every block read a whole shard's input span, and io.write sends a sharded
output through write_levels, whose planner counted a block with the Python path's
estimate, not the engine's cost.
"""
import os
import sys

import numpy as np
import pytest
import zarr

from dyna_zarr import _engine, io
from dyna_zarr import operations as ops

sys.path.insert(0, os.path.dirname(__file__))
from test_write_threads import RUNTIME_MB, _run  # noqa: E402  (fresh-process helpers)


@pytest.fixture(params=[False, True], ids=["python", "engine"])
def engine(request, monkeypatch):
    if request.param and not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", request.param)
    return request.param


@pytest.fixture(scope="module")
def sharded(tmp_path_factory):
    """A 64 MiB uint16 zarr v3 array: shards of (4, 512, 512) = 2 MiB, inner chunks of
    (4, 128, 128) = 128 KiB."""
    path = tmp_path_factory.mktemp("sharded") / "s.zarr"
    z = zarr.create_array(str(path), shape=(32, 1024, 1024), shards=(4, 512, 512),
                          chunks=(4, 128, 128), dtype="uint16")
    rng = np.random.default_rng(1)
    for i in range(0, 32, 4):
        z[i:i + 4] = (np.arange(4 << 20).reshape(4, 1024, 1024) % 4000
                      + rng.integers(0, 64, (4, 1024, 1024))).astype("uint16")
    return path


@pytest.mark.engine
def test_a_stride_over_a_sharded_array_reads_inner_chunks(sharded):
    """The engine's peak for a strided read on one thread is about one inner chunk's span
    (with the shard as the block: a shard's span, 16x larger here), and its cost says so."""
    core = pytest.importorskip("dyna_zarr_core")
    a = core.Array.open(str(sharded))
    v = a.index((slice(None), slice(None, None, 2), slice(None, None, 2)))
    stop = [4, 256, 256]                         # one shard's worth of output
    core.reset_memory_peak()
    base = core.memory_in_use()
    got = v.read([0, 0, 0], stop, threads=1)
    peak = core.memory_peak() - base - got.nbytes
    inner_span = 4 * 128 * 128 * 2
    assert peak <= 4 * inner_span, peak          # a shard's span would be 16 inner spans
    assert v.cost([0, 0, 0], stop, 1) <= 6 * inner_span
    full = np.asarray(zarr.open_array(str(sharded))[:4, :512:2, :512:2])
    np.testing.assert_array_equal(got, full)


@pytest.mark.parametrize("case", ["stride", "gauss"])
def test_write_levels_peak_within_the_budget(sharded, tmp_path, engine, case):
    """write_levels at 16 threads stays within memory_budget_mb + RUNTIME_MB on both
    engines: a strided source, and a Gaussian filter (the engine holds its box in float64
    and a filtered copy, far above the Python path's read estimate)."""
    budget_mb = 48
    grew = _run("""
        import json, os, sys
        from dyna_zarr import _engine, io
        from dyna_zarr import operations as o
        from dyna_zarr.operations.neighborhood import gaussian_filter
        _engine.set_engine(sys.argv[3] == "True")
        a = io.read(sys.argv[1])
        v = {"stride": a[:, ::2, ::2],
             "gauss": gaussian_filter(a, (0.0, 1.0, 1.0))}[sys.argv[4]]
        outs = lambda d: [{"path": os.path.join(sys.argv[2], d, "0"), "factors": (1, 1, 1)},
                          {"path": os.path.join(sys.argv[2], d, "1"), "factors": (1, 2, 2)}]
        io.write_levels(v[:4, :256, :256], outs("warm"), method="mean", max_threads=16)
        before, _ = rss_and_peak()
        io.write_levels(v, outs("o"), method="mean", max_threads=16,
                        memory_budget_mb=float(sys.argv[5]))
        _, peak = rss_and_peak()
        print(json.dumps((peak - before) / 2**20))
    """, sharded, tmp_path, engine, case, budget_mb)
    assert grew <= budget_mb + RUNTIME_MB, grew


@pytest.mark.engine
def test_write_levels_plans_threads_with_the_engine_cost(sharded, tmp_path, capsys, monkeypatch):
    """With the engine reading the chain, a worker costs its block, the engine's own cost
    for reading it on one thread, and its levels: the budget's threads follow from that
    (before, from the Python path's read estimate: 4 threads at 110 MB where the engine's
    cost allows 3)."""
    import re
    from dyna_zarr import _lowering
    from dyna_zarr.operations.neighborhood import gaussian_filter
    if not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", True)
    # TensorStore writes: the engine writer's buffers (io._writer_bytes) would add to it
    monkeypatch.setenv("DYNA_ZARR_ENGINE_WRITER", "0")
    v = gaussian_filter(io.read(str(sharded)), (0.0, 1.0, 1.0))
    budget_mb = 110
    io.write_levels(v, [{"path": str(tmp_path / "0"), "factors": (1, 1, 1)},
                        {"path": str(tmp_path / "1"), "factors": (1, 2, 2)}],
                    method="mean", max_threads=16, memory_budget_mb=budget_mb)
    line = next(l for l in capsys.readouterr().out.splitlines() if "[Levels]" in l)
    m = re.search(r"regions of \(([\d, ]+)\).*?(\d+) threads", line)
    region = [int(x) for x in m.group(1).split(",")]
    threads = int(m.group(2))
    block = int(np.prod(region)) * 2
    per_thread = block + int(_lowering.lowered(v).cost([0, 0, 0], region, 1)) + block * (1 + 1 / 4)
    # a filter's chunk cache comes off the budget first (test_engine_halo_cache)
    along_last = -(-1024 // region[-1])
    cache = sys.modules["dyna_zarr.io"]._halo_cache_bytes(v, region, along_last, 16,
                                                         budget_mb * 2**20)
    assert ("chunk cache" in line) == (cache > 0), line
    assert threads == max(1, min(16, int((budget_mb * 2**20 - cache) // per_thread))), (
        line, per_thread, cache)


def test_write_levels_values_unchanged(sharded, tmp_path, engine):
    """The levels are the same whatever the planner chose."""
    a = io.read(str(sharded))
    v = a[:, ::2, ::2]
    io.write_levels(v, [{"path": str(tmp_path / "0"), "factors": (1, 1, 1)},
                        {"path": str(tmp_path / "1"), "factors": (1, 2, 2)}],
                    method="mean", max_threads=8, memory_budget_mb=48)
    src = np.asarray(zarr.open_array(str(sharded))[:])[:, ::2, ::2]
    np.testing.assert_array_equal(np.asarray(zarr.open_array(str(tmp_path / "0"))[:]), src)
    want = np.asarray(ops.coarsen(io.read(str(tmp_path / "0")), (1, 2, 2)).compute())
    np.testing.assert_array_equal(np.asarray(zarr.open_array(str(tmp_path / "1"))[:]), want)
