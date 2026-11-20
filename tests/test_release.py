"""Giving freed memory back during a write (dyna_zarr/_release.py, decision DR-5)."""
import json
import os
import subprocess
import sys
import threading

import numpy as np
import pytest

import dyna_zarr
from dyna_zarr import _release, io


def _in_fresh_process(code, env=None):
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env=dict(os.environ, **(env or {})), timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_release_is_safe_from_many_threads_at_once(monkeypatch):
    calls = []
    monkeypatch.setattr(_release, "_RELEASE", lambda: calls.append(1))
    monkeypatch.setattr(_release, "_last_release", float("-inf"))
    barrier = threading.Barrier(8)

    def go():
        barrier.wait()
        for _ in range(50):
            _release.release()

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # rate-limited: far fewer releases than the 400 calls, at least one
    assert 1 <= len(calls) < 400


def test_release_is_rate_limited(monkeypatch):
    calls = []
    monkeypatch.setattr(_release, "_RELEASE", lambda: calls.append(1))
    monkeypatch.setattr(_release, "_last_release", float("-inf"))
    monkeypatch.setattr(_release, "RELEASE_EVERY_S", 3600.0)
    _release.release()
    _release.release()
    assert calls == [1]


def test_nothing_to_release_is_a_no_op(monkeypatch):
    monkeypatch.setattr(_release, "_RELEASE", None)
    _release.release()


def test_the_arena_limit_is_set_once_at_import():
    """In a fresh process, importing dyna limits glibc's arenas (where there are any)."""
    done = _in_fresh_process("import json, dyna_zarr; from dyna_zarr import _release; "
                             "print(json.dumps(_release._arenas_done))")
    assert done is True


class _FakeLibc:
    def __init__(self):
        self.calls = []

        class _Mallopt:
            def __call__(inner, param, value):
                self.calls.append((param, value))
                return 1
        self.mallopt = _Mallopt()


@pytest.mark.parametrize("user_set", [False, True])
def test_limit_arenas_unless_the_user_set_malloc_arena_max(monkeypatch, user_set):
    libc = _FakeLibc()
    monkeypatch.setattr(_release, "_glibc", lambda: libc)
    monkeypatch.setattr(_release, "_arenas_done", False)
    if user_set:
        monkeypatch.setenv("MALLOC_ARENA_MAX", "64")
    else:
        monkeypatch.delenv("MALLOC_ARENA_MAX", raising=False)
    _release.limit_arenas()
    _release.limit_arenas()                          # once per process
    assert libc.calls == ([] if user_set else [(_release._M_ARENA_MAX, _release.ARENAS)])


def test_writes_release_after_regions(tmp_path, monkeypatch):
    """Every write loop hands memory back as it goes (io.write fused and pipeline,
    write_levels)."""
    calls = []
    monkeypatch.setattr(_release, "_RELEASE", lambda: calls.append(1))
    monkeypatch.setattr(_release, "RELEASE_EVERY_S", 0.0)
    a = dyna_zarr.from_array(np.arange(32 * 64 * 64, dtype="uint16").reshape(32, 64, 64))
    for engine in ["fused", "pipeline"]:
        calls.clear()
        io.write(a, str(tmp_path / f"{engine}.zarr"), chunks=(8, 32, 32),
                 region_shape=(8, 64, 64), zarr_format=2, engine=engine, max_threads=4)
        assert calls, engine
    calls.clear()
    io.write_levels(a, [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1),
                             chunks=(8, 32, 32))], zarr_format=2, max_threads=4)
    assert calls
