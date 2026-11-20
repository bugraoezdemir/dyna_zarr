"""Memory and threads of dyna's engine (Workplan decision DR-5).

- Every read's real peak (the engine's heap, counted by its allocator) stays within
  ``Array.cost`` for the same region and threads: for Zarr arrays and every view, alone
  and chained, with and without splitting, on one thread and several.
- On one thread the cost is not far above the peak (it is a plan, not a guess).
- A read with ``threads=1`` keeps one thread busy (on the calling thread; it starts no
  thread on unsharded arrays). Reads that share a ``Pool`` never use more threads than
  the pool has.
- Short-lived Python threads reading and ending is safe, also when the engine was first
  used on one of them (mimalloc v3 crashed the process then; core/Cargo.toml).
"""
import subprocess
import sys
import textwrap
import threading

import numcodecs
import numpy as np
import pytest
import zarr
import zarr.codecs as zc

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")

SHAPE = (12, 96, 128)


@pytest.fixture(scope="module")
def arrays(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("memory")
    data = np.random.default_rng(0).integers(0, 4000, size=SHAPE, dtype="uint16")

    def write(name, chunks, fmt=3, **kw):
        z = zarr.create_array(str(tmp / name), shape=SHAPE, chunks=chunks, dtype="uint16",
                              zarr_format=fmt, **kw)
        z[...] = data
        return core.Array.open(str(tmp / name))

    return {
        "v2-blosc": write("v2", (4, 32, 32), 2, compressors=numcodecs.Blosc(cname="lz4")),
        "v3-zstd": write("v3", (4, 32, 32), compressors=zc.ZstdCodec(level=3)),
        "v3-none": write("v3n", (4, 32, 32), compressors=None),
        "v3-sharded": write("sh", (2, 16, 16), shards=(4, 32, 64)),
    }


def _rotation():
    m = np.eye(4)
    c, s = np.cos(0.4), np.sin(0.4)
    m[1:3, 1:3] = [[c, -s], [s, c]]
    return m.tolist()


CHAINS = {
    "zarr": lambda a: a,
    "slice": lambda a: a.index((slice(2, 11), slice(5, 90), slice(3, 120))),
    "transpose": lambda a: a.transpose([2, 0, 1]),
    "astype": lambda a: a.astype("float32"),
    "concatenate": lambda a: core.concatenate([a, a], axis=1),
    "stride": lambda a: a.downsample([2, 2, 2], "stride"),
    "mean": lambda a: a.downsample([2, 2, 2], "mean"),
    "mean-122": lambda a: a.downsample([1, 2, 2], "mean"),
    "gaussian": lambda a: a.downsample([2, 2, 2], "gaussian"),
    "resample": lambda a: a.resample(_rotation(), list(a.shape), "linear"),
    "mean-of-mean": lambda a: a.downsample([2, 2, 2]).downsample([2, 2, 2]),
    "gaussian-of-resample": lambda a: a.resample(_rotation(), list(a.shape), "nearest")
    .downsample([2, 2, 2], "gaussian"),
    "stride-of-transpose": lambda a: a.transpose([0, 2, 1]).downsample([1, 3, 2], "stride"),
    # split into pieces by a tiny read limit
    "mean-split": lambda a: a.downsample([2, 2, 2], "mean", max_read_mb=0.02),
    "gaussian-split": lambda a: a.downsample([2, 2, 2], "gaussian", max_read_mb=0.05),
    "resample-split": lambda a: a.resample(_rotation(), list(a.shape), "linear", max_read_mb=0.02),
}


def _regions(v):
    sh = list(v.shape)
    yield [0] * len(sh), sh
    yield [x // 4 for x in sh], [max(x // 4 + 1, 3 * x // 4) for x in sh]
    yield [x - 1 for x in sh], sh                       # the last element


def _peak(v, s0, s1, threads, **kw):
    out = np.empty([e - b for b, e in zip(s0, s1)], dtype=v.dtype)
    core.reset_memory_peak()
    base = core.memory_in_use()
    v.read(s0, s1, out=out, threads=threads, **kw)
    return core.memory_peak() - base, out


@pytest.mark.parametrize("threads", [1, 4, 16])
@pytest.mark.parametrize("chain", list(CHAINS))
@pytest.mark.parametrize("name", ["v2-blosc", "v3-zstd", "v3-none", "v3-sharded"])
def test_peak_within_cost(arrays, name, chain, threads):
    v = CHAINS[chain](arrays[name])
    for s0, s1 in _regions(v):
        peak, _ = _peak(v, s0, s1, threads)
        cost = v.cost(s0, s1, threads=threads)
        assert peak <= cost, (s0, s1, peak, cost)


@pytest.mark.parametrize("chain", list(CHAINS))
def test_one_thread_cost_is_close(arrays, chain):
    """A plan, not a guess: on one thread the cost is at most a few times the peak."""
    v = CHAINS[chain](arrays["v3-zstd"])
    s0, s1 = [0] * v.ndim, list(v.shape)
    peak, _ = _peak(v, s0, s1, 1)
    assert v.cost(s0, s1, threads=1) <= 4 * peak + (64 << 10)


def test_threads_raise_the_cost_not_past_it(arrays):
    v = arrays["v2-blosc"].downsample([2, 2, 2], "mean")
    s0, s1 = [0] * 3, list(v.shape)
    costs = [v.cost(s0, s1, threads=t) for t in (1, 2, 4, 8)]
    assert costs[1] <= costs[2] <= costs[3]
    for t, c in zip((2, 4, 8), costs[1:]):
        assert _peak(v, s0, s1, t)[0] <= c


def test_cost_needs_a_valid_region_and_threads(arrays):
    a = arrays["v3-zstd"]
    with pytest.raises(core.ReadError):
        a.cost([0, 0, 0], [13, 1, 1])
    with pytest.raises(ValueError):
        a.cost([0, 0, 0], [1, 1, 1], threads=0)


def test_memory_counters(arrays):
    assert core.memory_in_use() >= 0
    now = core.reset_memory_peak()
    assert core.memory_peak() == now
    keep = arrays["v3-zstd"].transpose([2, 1, 0])
    keep.read([0, 0, 0], list(keep.shape))
    assert core.memory_peak() > now


def test_pool(arrays):
    pool = core.Pool(3)
    assert pool.threads == 3 and "3 threads" in repr(pool)
    with pytest.raises(ValueError):
        core.Pool(0)
    v = arrays["v3-zstd"].downsample([2, 2, 2], "gaussian")
    s0, s1 = [0] * 3, list(v.shape)
    want = v.read(s0, s1, threads=1)
    np.testing.assert_array_equal(v.read(s0, s1, threads=8, pool=pool), want)
    np.testing.assert_array_equal(v.read(s0, s1, threads=1, pool=pool), want)


def test_reads_sharing_a_pool_from_many_threads(arrays):
    """Many callers, one pool: every result right, every peak within its cost."""
    pool = core.Pool(2)
    v = arrays["v2-blosc"].downsample([2, 2, 2], "mean")
    want = v.read([0, 0, 0], list(v.shape), threads=1)
    errors = []

    def one(k):
        try:
            z = k % v.shape[0]
            got = v.read([z, 0, 0], [z + 1] + list(v.shape[1:]), threads=4, pool=pool)
            np.testing.assert_array_equal(got, want[z:z + 1])
        except Exception as e:                              # noqa: BLE001 - reported below
            errors.append(e)

    workers = [threading.Thread(target=one, args=(k,)) for k in range(12)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert not errors


def _run(code, *args):
    """`code` in a fresh interpreter (thread counts of this process are shared)."""
    out = subprocess.run([sys.executable, "-c", textwrap.dedent(code), *map(str, args)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return out.stdout.split()


@pytest.mark.parametrize("sharded", [False, True])
def test_one_thread_reads_keep_one_thread_busy(tmp_path, sharded):
    """zarrs' default (4 chunks at once) used to run a one-thread read on several
    threads. Busy threads are measured as CPU time over wall time; on an unsharded array
    no thread may be started at all (sharded, zarrs hands its one task at a time to
    rayon's global pool, which may start idle threads)."""
    pytest.importorskip("psutil")
    path = tmp_path / "many"
    kw = {"shards": (2, 64, 64)} if sharded else {}
    z = zarr.create_array(str(path), shape=(8, 256, 256), chunks=(1, 32, 32), dtype="uint8", **kw)
    z[...] = 7
    before, after, busy = _run("""
        import sys, time, psutil, numpy, dyna_zarr_core as core   # numpy first: OpenBLAS
        a = core.Array.open(sys.argv[1])
        me = psutil.Process()
        views = (a, a.downsample([2, 2, 2]), a.downsample([2, 2, 2], "gaussian"),
                 a.downsample([2, 2, 2], "stride"))
        before = me.num_threads()
        c0, t0 = sum(me.cpu_times()[:2]), time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            for v in views:
                v.read([0, 0, 0], list(v.shape), threads=1)
                v.cost([0, 0, 0], list(v.shape), threads=4)
        busy = (sum(me.cpu_times()[:2]) - c0) / (time.perf_counter() - t0)
        print(before, me.num_threads(), busy)
    """, path)
    assert float(busy) < 1.5
    if not sharded:
        assert int(after) == int(before)


def test_short_lived_threads_reading(tmp_path):
    """Python threads that read and end, as io.write's region workers do."""
    path = tmp_path / "a"
    z = zarr.create_array(str(path), shape=(8, 64, 64), chunks=(2, 32, 32), dtype="uint16",
                          zarr_format=2)
    z[...] = 3
    out = _run("""
        import sys
        from concurrent.futures import ThreadPoolExecutor
        import dyna_zarr_core as core
        a = core.Array.open(sys.argv[1])
        for _ in range(100):
            with ThreadPoolExecutor(4) as ex:
                list(ex.map(lambda k: int(a.read([k, 0, 0], [k + 1, 64, 64]).sum()), range(8)))
        print("ok")
    """, path)
    assert out == ["ok"]


def test_first_use_on_a_thread_that_ends(tmp_path):
    """The engine loaded and first used on a thread that ends (dyna loads it lazily,
    often on an io.write worker), then used from many new threads. With mimalloc v3 the
    first new thread that reused the ended thread's TEB address (Windows reuses them; it
    is mimalloc's thread id) took v3's torn-down main heap and crashed the process."""
    path = tmp_path / "a"
    z = zarr.create_array(str(path), shape=(4, 64, 64), chunks=(2, 32, 32), dtype="uint16",
                          zarr_format=2)
    z[...] = 3
    out = _run("""
        import sys, threading
        box = {}
        def teb():                                   # the thread's TEB (Windows), else 0
            if sys.platform != "win32":
                return 0
            import ctypes as C
            k32, nt = C.WinDLL("kernel32"), C.WinDLL("ntdll")
            k32.GetCurrentThread.restype = C.c_void_p
            info = (C.c_uint64 * 6)()
            nt.NtQueryInformationThread(C.c_void_p(k32.GetCurrentThread()), 0, info,
                                        C.sizeof(info), None)
            return info[1]
        def read():
            a = box["core"].Array.open(sys.argv[1])
            assert int(a.read([0, 0, 0], [1, 64, 64]).sum()) == 3 * 64 * 64
        def first():
            import dyna_zarr_core
            box["core"], box["teb"] = dyna_zarr_core, teb()
            read()
        t = threading.Thread(target=first); t.start(); t.join()
        reused = 0
        def later():
            global reused
            reused += box["teb"] != 0 and teb() == box["teb"]
            read()
        for _ in range(2000):
            t = threading.Thread(target=later); t.start(); t.join()
            if reused >= 2:
                break
        print("ok", reused)
    """, path)
    assert out[0] == "ok"


def test_a_pool_is_all_the_threads_reads_use(tmp_path):
    pytest.importorskip("psutil")
    path = tmp_path / "many"
    z = zarr.create_array(str(path), shape=(8, 256, 256), chunks=(1, 32, 32), dtype="uint8")
    z[...] = 7
    before, after = _run("""
        import sys, threading, psutil, numpy, dyna_zarr_core as core   # numpy: as above
        a = core.Array.open(sys.argv[1])
        me = psutil.Process()
        before = me.num_threads()
        pool = core.Pool(3)
        def go():
            for v in (a, a.downsample([2, 2, 2]), a.downsample([2, 2, 2], "gaussian")):
                v.read([0, 0, 0], list(v.shape), threads=16, pool=pool)
        callers = [threading.Thread(target=go) for _ in range(6)]
        for c in callers: c.start()
        for c in callers: c.join()
        print(before, me.num_threads())
    """, path)
    assert int(after) - int(before) <= 3


@pytest.mark.parametrize("env, expected", [(None, 1), ("10", 10), ("7", 7)])
def test_freed_memory_goes_back_within_a_millisecond(env, expected):
    """The engine's allocator hands freed memory back 1 ms later (mimalloc's default is
    10 ms, which kept several freed chunk buffers per read thread: core/src/lib.rs);
    MIMALLOC_PURGE_DELAY wins. With "10" this also checks that the option the engine
    sets is mimalloc's purge delay (its default is 10)."""
    import os
    environ = {k: v for k, v in os.environ.items() if k != "MIMALLOC_PURGE_DELAY"}
    if env is not None:
        environ["MIMALLOC_PURGE_DELAY"] = env
    out = subprocess.run([sys.executable, "-c",
                          "import dyna_zarr_core as c; print(c.allocator_purge_delay_ms())"],
                         capture_output=True, text=True, timeout=120, env=environ)
    assert out.returncode == 0, out.stderr
    assert int(out.stdout.split()[-1]) == expected
