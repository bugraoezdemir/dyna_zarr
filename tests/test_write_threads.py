"""A write's threads and memory (Workplan decision DR-5, task DR6b).

- ``max_threads`` counts every thread a write keeps busy: TensorStore's pools, which
  decode the sources and encode the output, are limited to it (they used every core:
  ``max_threads=1`` kept about 4 cores busy on a copy), and the engine's threads inside
  regions come out of it.
- The planner splits ``max_threads`` into regions in flight x engine threads per region;
  more regions win, few regions get more threads each, and the budget counts the
  engine's own cost of a region at the threads it gets.
- A write's peak memory, measured in a fresh process, stays within ``memory_budget_mb``
  plus a fixed allowance for the runtime (thread stacks, the allocators' per-thread
  heaps, TensorStore's encoder buffers: about 20-30 MB measured), on both engines.
"""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import tensorstore as ts
import zarr

from dyna_zarr import _engine, _scope, io
from dyna_zarr import operations as ops

io_module = sys.modules["dyna_zarr.io"]

#: what the runtime may add to a write's peak beyond memory_budget_mb (measured 20-30 MB)
RUNTIME_MB = 48


@pytest.fixture(params=[False, True], ids=["python", "engine"])
def engine(request, monkeypatch):
    """Each test on the Python path and on the engine (when installed)."""
    if request.param and not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", request.param)
    return request.param


def _source(path, shape=(16, 128, 128), chunks=(4, 32, 32)):
    data = (np.arange(np.prod(shape)).reshape(shape) % 4000).astype("uint16")
    z = zarr.create_array(str(path), shape=shape, chunks=chunks, dtype="uint16")
    z[...] = data
    return data


def _limit(handle):
    context = handle.spec(retain_context=True).to_json().get("context", {})
    return context.get("data_copy_concurrency", {}).get("limit")


# ----------------------------------------------------------------------- the scope

def test_outside_a_write_nothing_changes(tmp_path):
    _source(tmp_path / "s")
    h = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(tmp_path / "s")}},
                read=True).result()
    assert _scope.current() is None
    assert _scope.ts_source(h) is h


def test_a_write_limits_its_output_and_sources(tmp_path, monkeypatch, engine):
    """The output and every TensorStore source read during the write use TensorStore
    pools of max_threads threads; the source keeps its own data and view."""
    data = _source(tmp_path / "s")
    seen = {}
    real_open, real_source = io_module._open_output, _scope.ts_source

    def spy_open(spec):
        handle = real_open(spec)
        seen["output"] = _limit(handle)
        return handle

    def spy_source(handle):
        out = real_source(handle)
        seen.setdefault("sources", set()).add(_limit(out))
        return out
    monkeypatch.setattr(io_module, "_open_output", spy_open)
    monkeypatch.setattr(_scope, "ts_source", spy_source)
    a = io.read(str(tmp_path / "s"), backend="tensorstore")
    io.write(a[:, 5:100, ::3], str(tmp_path / "o.zarr"), max_threads=3)
    assert seen["output"] == 3
    if not engine:                        # the engine reads the store itself
        assert seen["sources"] == {3}     # the write's own pool (one for both)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "o.zarr"))[...],
                                  data[:, 5:100, ::3])


def test_a_write_inside_a_write_keeps_the_outer_threads():
    with _scope.write_scope(2) as outer:
        with _scope.write_scope(8) as inner:
            assert inner is outer and inner.max_threads == 2
    assert _scope.current() is None


def test_a_reopened_source_keeps_its_own_context(tmp_path):
    """Only the two limits change: other context entries (credentials, a cache pool)
    stay with the source."""
    _source(tmp_path / "s")
    h = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(tmp_path / "s")},
                 "context": {"cache_pool": {"total_bytes_limit": 12345}}}, read=True).result()
    with _scope.write_scope(2):
        r = _scope.ts_source(h)
        assert _scope.ts_source(h) is r                      # once per write
    context = r.spec(retain_context=True).to_json()["context"]
    assert context["cache_pool"] == {"total_bytes_limit": 12345}
    assert context["data_copy_concurrency"] == {"limit": 2}
    assert context["file_io_concurrency"] == {"limit": 2}
    np.testing.assert_array_equal(r.read().result(), h.read().result())


def test_sources_and_output_share_one_pool(tmp_path):
    """The reopened source takes the write's pools (not pools of its own of the same
    size, which would double the threads)."""
    _source(tmp_path / "s")
    h = ts.open({"driver": "zarr3", "kvstore": {"driver": "file", "path": str(tmp_path / "s")}},
                read=True).result()
    with _scope.write_scope(2) as scope:
        spec = scope.ts_spec(h.spec(retain_context=True).to_json())
    assert not set(spec["context"]) & {"data_copy_concurrency", "file_io_concurrency"}


def test_in_memory_handles_are_not_reopened():
    h = ts.array(np.arange(10))
    with _scope.write_scope(2):
        assert _scope.ts_source(h) is h


# ---------------------------------------------------------------------- the planner

def _plan(array, region, max_threads, budget_mb=None):
    shape = array.shape
    bands = [tuple((i * r, min((i + 1) * r, s)) for i, r, s in zip(idx, region, shape))
             for idx in np.ndindex(*[-(-s // r) for s, r in zip(shape, region)])]
    read, out = io_module._region_cost(array, region, shape, array.dtype)
    budget = None if budget_mb is None else budget_mb * 2**20
    return io_module._plan_regions(array, bands, shape, max_threads, budget, 2 * read + out, out)


@pytest.mark.engine
def test_the_planner_splits_max_threads(tmp_path, monkeypatch):
    """Many regions: one thread each. Few regions: the threads go to them. One region:
    all of them. Never more than max_threads in all."""
    if not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", True)
    _source(tmp_path / "s")
    a = ops.coarsen(io.read(str(tmp_path / "s")), (2, 2, 2), "mean")     # (8, 64, 64)
    for region, threads, want in [((2, 16, 16), 8, (8, 1)),   # 64 regions
                                  ((8, 32, 64), 8, (2, 4)),   # 2 regions
                                  ((8, 64, 64), 8, (1, 8)),   # 1 region
                                  ((8, 32, 64), 1, (1, 1))]:
        r, t, cost, fits = _plan(a, region, threads)
        assert (r, t) == want and fits and r * t <= threads


@pytest.mark.engine
def test_the_planner_budgets_the_engine_cost(tmp_path, monkeypatch):
    """A budget for one region with all threads gives that; a budget below one region
    with one thread says so; regions x cost never exceeds the budget."""
    if not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", True)
    _source(tmp_path / "s")
    a = ops.coarsen(io.read(str(tmp_path / "s")), (2, 2, 2), "mean")
    region = (2, 16, 16)
    r, t, one, fits = _plan(a, region, 8, budget_mb=1e-6)
    assert (r, t, fits) == (1, 1, False)
    for budget in (one, 3 * one, 100 * one):
        r, t, cost, fits = _plan(a, region, 8, budget_mb=budget / 2**20)
        assert fits and r * cost <= budget and r * t <= 8


def test_the_python_path_has_one_thread_per_region(tmp_path, monkeypatch):
    """max_threads workers of one thread each, as before DR6b."""
    monkeypatch.setattr(_engine, "_choice", False)
    _source(tmp_path / "s")
    a = io.read(str(tmp_path / "s"))
    for region in [(16, 128, 128), (4, 32, 32)]:
        r, t, _, fits = _plan(a, region, 8)
        assert (r, t, fits) == (8, 1, True)


def test_engine_threads_inside_a_region_give_the_same_values(tmp_path, engine, capsys):
    """One region read with all of max_threads (the engine's pool) or region by region:
    the same output."""
    data = _source(tmp_path / "s")
    a = ops.coarsen(io.read(str(tmp_path / "s")), (2, 2, 2), "mean")
    io.write(a, str(tmp_path / "one.zarr"), region_shape=a.shape, max_threads=4)
    if engine:
        assert "with 4 engine threads" in capsys.readouterr().out
    io.write(a, str(tmp_path / "many.zarr"), region_shape=(2, 16, 16), max_threads=4)
    one = zarr.open_array(str(tmp_path / "one.zarr"))[...]
    np.testing.assert_array_equal(one, zarr.open_array(str(tmp_path / "many.zarr"))[...])
    assert one.shape == (8, 64, 64)
    want = data.reshape(8, 2, 64, 2, 64, 2).mean(axis=(1, 3, 5))
    np.testing.assert_allclose(one, want, atol=1)


# ------------------------------------------------- busy threads and peak memory (process)

_HELPERS = r'''
import ctypes, os, sys, time
def rss_and_peak():
    """(memory now, peak memory) of this process in bytes: the working set on Windows,
    the resident set elsewhere (on macOS only the peak, so 'now' is the peak so far)."""
    if sys.platform == "win32":
        from ctypes import wintypes
        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t)] + \
                       [(n, ctypes.c_size_t) for n in ("a", "b", "c", "d", "e", "f")]
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        k = ctypes.WinDLL("kernel32"); p = ctypes.WinDLL("psapi")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        p.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
        return pmc.WorkingSetSize, pmc.PeakWorkingSetSize
    if os.path.exists("/proc/self/status"):
        with open("/proc/self/status") as f:
            d = dict(line.split(":", 1) for line in f)
        return int(d["VmRSS"].split()[0]) << 10, int(d["VmHWM"].split()[0]) << 10
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # bytes on macOS
    return peak, peak
'''


def _run(code, *args):
    """``code`` in a fresh interpreter: thread pools and peak memory are per process."""
    out = subprocess.run([sys.executable, "-c", _HELPERS + textwrap.dedent(code),
                          *map(str, args)], capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    """A 64 MiB uint16 source in (4, 256, 256) chunks."""
    path = tmp_path_factory.mktemp("big") / "s.zarr"
    z = zarr.create_array(str(path), shape=(32, 1024, 1024), chunks=(4, 256, 256),
                          dtype="uint16")
    rng = np.random.default_rng(0)
    for i in range(0, 32, 4):
        z[i:i + 4] = (np.arange(4 << 20).reshape(4, 1024, 1024) % 4000
                      + rng.integers(0, 64, (4, 1024, 1024))).astype("uint16")
    return path


@pytest.mark.parametrize("max_threads", [1, 2])
def test_a_write_keeps_max_threads_busy(big, tmp_path, engine, max_threads):
    """Busy threads (CPU time over wall time) of a copy, a strided level and a mean
    level stay near max_threads. Before DR6b, max_threads=1 kept about 4 cores busy on
    a copy (TensorStore encoded on every core). The margin covers the worker's own read
    overlapping TensorStore encoding another region.

    On a machine with no more cores than the limit, a write cannot go over it, so the case
    is skipped: there it can only fail by mismeasuring (a 3-core macOS runner once
    measured 3.35 cores busy at max_threads=2)."""
    limit = 1.5 * max_threads + 0.25
    if (os.cpu_count() or 1) <= limit:
        pytest.skip(f"{os.cpu_count()} cores cannot exceed {limit} busy")
    busy = _run("""
        import json, os, sys, time
        from dyna_zarr import _engine, io
        from dyna_zarr import operations as o
        _engine.set_engine(sys.argv[3] == "True")
        a = io.read(sys.argv[1])
        out = {}
        for name, v in {"copy": a, "stride": a[:, ::2, ::2],
                        "mean": o.coarsen(a, (2, 2, 2))}.items():
            c0, t0 = os.times(), time.perf_counter()
            io.write(v, os.path.join(sys.argv[2], name + ".zarr"),
                     max_threads=int(sys.argv[4]))
            c1, t1 = os.times(), time.perf_counter()
            out[name] = (c1.user - c0.user + c1.system - c0.system) / (t1 - t0)
        print(json.dumps(out))
    """, big, tmp_path, engine, max_threads)
    for name, cores in busy.items():
        assert cores <= limit, (name, busy, f"{os.cpu_count()} cores")


@pytest.mark.parametrize("budget_mb", [32, 64])   # one region needs ~22 MB
@pytest.mark.parametrize("case", ["copy", "mean", "float32"])
def test_peak_memory_within_the_budget(big, tmp_path, engine, budget_mb, case):
    """The write's peak, over what the process held before it, stays within
    memory_budget_mb plus RUNTIME_MB, on both engines (measured 20-30 MB over at small
    budgets, under the budget at 128 MB). On Linux this needs glibc's arenas limited and
    freed buffers handed back during the write (_release): without, a copy at 64 MB grew
    by 93-110 MB."""
    grew = _run("""
        import json, os, sys
        from dyna_zarr import _engine, io
        from dyna_zarr import operations as o
        _engine.set_engine(sys.argv[3] == "True")
        a = io.read(sys.argv[1])
        v = {"copy": a, "mean": o.coarsen(a, (2, 2, 2)), "float32": a.astype("float32")}[sys.argv[5]]
        # a small write of the same kind first: what the first write of a process starts
        # once (TensorStore's pools, thread stacks) is not the write's memory
        io.write(v[:4, :256, :256], os.path.join(sys.argv[2], "warm.zarr"), max_threads=4)
        before, _ = rss_and_peak()
        io.write(v, os.path.join(sys.argv[2], "o.zarr"), max_threads=4,
                 memory_budget_mb=float(sys.argv[4]))
        _, peak = rss_and_peak()
        print(json.dumps((peak - before) / 2**20))
    """, big, tmp_path, engine, budget_mb, case)
    assert grew <= budget_mb + RUNTIME_MB, grew
