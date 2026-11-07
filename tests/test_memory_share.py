"""io.memory_share: write_levels takes memory and threads that free up while it runs.

A caller (eubi-bridge: conversions side by side in one run) hands write_levels a share,
``share() -> (memory MB, threads)`` usable now. The pass plans with what the share grew
by since the caller planned, and starts more threads while it runs when it grows further;
it never takes less than it was asked for. The output is the same either way.
"""
import importlib
import re
import threading
import time

import numpy as np
import pytest
import zarr

import dyna_zarr
from dyna_zarr.io import io

dio = importlib.import_module("dyna_zarr.io")

SHAPE = (64, 512, 512)


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(dio, "_SHARE_POLL_S", 0.02)


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(3).integers(0, 250, size=SHAPE, dtype="uint8")


class _Slow:
    """A source whose reads take a little while (so a pass lasts long enough to grow),
    recording which threads read it."""

    def __init__(self, data, chunks=(8, 64, 64), delay=0.01):
        self.data, self.shape, self.dtype, self.ndim = data, data.shape, data.dtype, data.ndim
        self.chunks = chunks
        self.delay = delay
        self.lock = threading.Lock()
        self.names, self.reads = set(), 0

    def __getitem__(self, key):
        time.sleep(self.delay)
        with self.lock:
            self.names.add(threading.current_thread().name)
            self.reads += 1
        return self.data[key]


class _Share:
    """``before`` until the source has had ``at`` reads, then ``after``; counts calls."""

    def __init__(self, src, before, after, at):
        self.src, self.before, self.after, self.at = src, before, after, at
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.after if self.src.reads >= self.at else self.before


def _workers(src):
    return {n for n in src.names if n.startswith("LevelsWorker")}


def test_threads_grow_while_a_pass_runs(data, tmp_path):
    src = _Slow(data)
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 64, 64))]
    share = _Share(src, before=(1.5, 1), after=(64.0, 4), at=8)
    with io.memory_share(share), io.capture_messages() as messages:
        io.write_levels(dyna_zarr.from_array(src), outs, zarr_format=2, max_threads=1,
                        region_size_mb=0.25, memory_budget_mb=1.5)
    start = next(m for m in messages if "outputs in one pass" in m)
    assert re.search(r", 1 threads", start), start               # planned with one
    assert any("1 -> 4 threads" in m for m in messages), messages
    assert len(_workers(src)) > 1, src.names
    np.testing.assert_array_equal(zarr.open_array(outs[0]["path"])[...], data)


def test_a_share_grown_before_the_pass_plans_it(data, tmp_path):
    src = _Slow(data, delay=0.0)
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 64, 64))]
    calls = {"n": 0}

    def share():                     # (1.5, 1) when the caller planned, then (64, 4)
        calls["n"] += 1
        return (1.5, 1) if calls["n"] == 1 else (64.0, 4)

    with io.memory_share(share), io.capture_messages() as messages:
        io.write_levels(dyna_zarr.from_array(src), outs, zarr_format=2, max_threads=1,
                        region_size_mb=0.25, memory_budget_mb=1.5)
    assert any("memory share grew" in m and "4 threads (asked 1)" in m for m in messages)
    start = next(m for m in messages if "outputs in one pass" in m)
    assert re.search(r", 4 threads", start), start
    np.testing.assert_array_equal(zarr.open_array(outs[0]["path"])[...], data)


def test_chunk_appending_mode_grows_within_the_grown_memory(data, tmp_path):
    # big shards appended chunk by chunk (dyna's own shard writer): more threads only while
    # the planned tile's memory with them fits the grown share
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=(8, 64, 64),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    src = _Slow(io.read(str(tmp_path / "src.zarr"))[...], delay=0.005)
    factors = [(1, 1, 1), (2, 2, 2)]
    outs = [dict(path=str(tmp_path / f"L{k}.zarr"), factors=f, chunks=(8, 32, 32),
                 shard_coefficients=(2, 4, 4)) for k, f in enumerate(factors)]
    share = _Share(src, before=(2.0, 1), after=(12.0, 4), at=4)
    with io.memory_share(share), io.capture_messages() as messages:
        n = io.write_levels(dyna_zarr.from_array(src), outs, zarr_format=3,
                            backend="tensorstore", max_threads=1, region_size_mb=0.05,
                            memory_budget_mb=2.0)
    assert any("appended to their shards" in m for m in messages), messages
    grew = [m for m in messages if "memory share grew to" in m]
    assert grew, messages
    assert len(_workers(src)) > 1, src.names
    peak = float(re.search(r"peaked at ([0-9.]+) MiB",
                           next(m for m in messages if "peaked at" in m)).group(1))
    assert peak <= 12.0, (peak, messages)
    for o in outs[:n]:
        np.testing.assert_array_equal(zarr.open_array(o["path"])[...],
                                      data[tuple(slice(None, None, k) for k in o["factors"])])


def test_a_shrinking_share_never_takes_threads_away(data, tmp_path):
    src = _Slow(data, delay=0.002)
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 64, 64))]
    share = _Share(src, before=(64.0, 4), after=(0.1, 1), at=2)
    with io.memory_share(share), io.capture_messages() as messages:
        io.write_levels(dyna_zarr.from_array(src), outs, zarr_format=2, max_threads=4,
                        region_size_mb=0.25, memory_budget_mb=64.0)
    assert not any("memory share grew" in m for m in messages), messages
    start = next(m for m in messages if "outputs in one pass" in m)
    assert re.search(r", 4 threads", start), start
    np.testing.assert_array_equal(zarr.open_array(outs[0]["path"])[...], data)


def test_a_failing_share_keeps_the_plan(data, tmp_path):
    src = _Slow(data, delay=0.002)
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 64, 64))]
    calls = {"n": 0}

    def share():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("ledger gone")
        return (1.5, 1)

    with io.memory_share(share), io.capture_messages() as messages:
        io.write_levels(dyna_zarr.from_array(src), outs, zarr_format=2, max_threads=1,
                        region_size_mb=0.25, memory_budget_mb=1.5)
    assert not any("memory share grew" in m for m in messages), messages
    assert _workers(src) == {"LevelsWorker-0"}, src.names
    np.testing.assert_array_equal(zarr.open_array(outs[0]["path"])[...], data)


def test_without_a_share_nothing_is_polled(data, tmp_path, monkeypatch):
    # outside io.memory_share the pass joins its threads as before
    src = _Slow(data, delay=0.0)
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1), chunks=(8, 64, 64))]
    with io.capture_messages() as messages:
        io.write_levels(dyna_zarr.from_array(src), outs, zarr_format=2, max_threads=2,
                        region_size_mb=0.25, memory_budget_mb=64.0)
    assert not any("memory share" in m for m in messages), messages
    np.testing.assert_array_equal(zarr.open_array(outs[0]["path"])[...], data)
