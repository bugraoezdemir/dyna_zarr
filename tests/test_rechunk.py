"""
Two-phase disk-staged rechunk engine (Rechunker's algorithm, dask-free) + flatten built on
top of it. Covers all grid-alignment cases (coarsen / refine / misaligned / mixed / partial),
read-once, and the flatten bridge (rechunk-to-flat-contiguous + relabel).
"""
import os
import threading
import time

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations as ops
from dyna_zarr.rechunk import rechunk, flatten_write, reshape_write


class CountZarr:
    """Wrap a zarr array and count element reads (to assert read-once). Thread-safe so it's
    valid under the parallel copy engine."""

    def __init__(self, z):
        self.z = z
        self.reads = 0
        self._lock = threading.Lock()

    shape = property(lambda self: self.z.shape)
    chunks = property(lambda self: self.z.chunks)
    dtype = property(lambda self: self.z.dtype)

    def __getitem__(self, k):
        b = self.z[k]
        with self._lock:
            self.reads += int(np.prod(b.shape))
        return b


@pytest.fixture
def src3d():
    arr = np.random.default_rng(0).random((20, 24, 18)).astype(np.float32)
    return arr, zarr.array(arr, chunks=(8, 8, 8))


@pytest.mark.parametrize("tc", [
    (16, 16, 16),   # coarsen (source divides target) -> one-pass consolidate
    (4, 4, 4),      # refine  (target divides source)  -> one-pass split
    (5, 7, 6),      # misaligned -> two-pass via disk intermediate
    (8, 6, 18),     # mixed: divides on axis0, misaligned axis1, full axis2
    (20, 24, 18),   # single big chunk (partial edges)
])
def test_rechunk_matches_source(src3d, tc, tmp_path):
    arr, z = src3d
    out = str(tmp_path / "o.zarr")
    rechunk(z, tc, out, zarr_format=2)
    r = zarr.open(out, mode="r")
    np.testing.assert_array_equal(r[:], arr)
    assert r.chunks == tc


def test_rechunk_read_once(src3d, tmp_path):
    arr, z = src3d
    cz = CountZarr(z)
    rechunk(cz, (5, 7, 6), str(tmp_path / "o.zarr"), zarr_format=2)   # misaligned = two-pass
    assert cz.reads == arr.size          # each source element read exactly once


@pytest.mark.parametrize("max_workers", [1, 4])
def test_rechunk_parallel_correct_and_read_once(src3d, max_workers, tmp_path):
    """Small max_mem forces MANY regions -> the thread pool is actually exercised. Result must
    match numpy and each source element must still be read exactly once under concurrency."""
    arr, z = src3d
    cz = CountZarr(z)
    out = str(tmp_path / "o.zarr")
    rechunk(cz, (5, 7, 6), out, max_mem=4096, max_workers=max_workers, zarr_format=2)  # misaligned
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr)
    assert cz.reads == arr.size


def test_rechunk_dynamic_source(src3d, tmp_path):
    arr, z = src3d
    da = DynamicArray(z)
    out = str(tmp_path / "o.zarr")
    rechunk(da, (5, 7, 6), out, zarr_format=2)
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr)


@pytest.mark.parametrize("shape,chunks", [
    ((12, 10, 8, 6), (5, 4, 3, 4)),   # 4D non-cubic partial
    ((20, 24, 18), (8, 8, 8)),        # 3D cubic
    ((100,), (16,)),                  # already 1D
])
def test_flatten_write_matches_numpy(shape, chunks, tmp_path):
    arr = np.random.default_rng(2).random(shape).astype(np.float32)
    da = DynamicArray(zarr.array(arr, chunks=chunks))
    out = str(tmp_path / "f.zarr")
    flatten_write(da, out, output_chunks=(64,), zarr_format=2)
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr.ravel())


def test_flatten_write_read_once(tmp_path):
    arr = np.random.default_rng(3).random((20, 24, 18)).astype(np.float32)
    cz = CountZarr(zarr.array(arr, chunks=(8, 8, 8)))
    flatten_write(DynamicArray(cz), str(tmp_path / "f.zarr"),
                  output_chunks=(64,), max_mem=1 << 20, zarr_format=2)
    assert cz.reads == arr.size          # source read exactly once through the rechunk stage


@pytest.mark.parametrize("max_mem", [1 << 20, 4096, 512, 64])   # force split axis from 0 -> deep
def test_flatten_write_tiny_budget_deep_split(max_mem, tmp_path):
    """Small budgets push the split axis past axis 0; result must still equal ravel()."""
    arr = np.random.default_rng(5).random((6, 5, 4, 7)).astype(np.float32)
    da = DynamicArray(zarr.array(arr, chunks=(2, 2, 2, 3)))
    out = str(tmp_path / "f.zarr")
    flatten_write(da, out, output_chunks=(9,), max_mem=max_mem, zarr_format=2)
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr.ravel())


def test_iowrite_flatten_routes_to_rechunk(tmp_path):
    arr = np.random.default_rng(4).random((12, 10, 8, 6)).astype(np.float32)
    da = DynamicArray(zarr.array(arr, chunks=(5, 4, 3, 4)))
    out = str(tmp_path / "f.zarr")
    io.write(ops.flatten(da), out, zarr_format=2, chunks=(64,))
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr.ravel())


@pytest.mark.parametrize("shape,chunks,target,tchunks", [
    ((12, 10, 8, 6), (5, 4, 3, 4), (120, 48), (7, 9)),          # 4D -> 2D
    ((12, 10, 8, 6), (5, 4, 3, 4), (24, 10, 24), (7, 4, 10)),   # 4D -> 3D
    ((12, 10, 8, 6), (5, 4, 3, 4), (6, 20, 8, 6), (4, 7, 5, 4)),# 4D -> 4D
    ((12, 10, 8, 6), (5, 4, 3, 4), (5760,), (100,)),            # 4D -> 1D
    ((20, 24, 18), (8, 8, 8), (60, 144), (16, 16)),             # 3D -> 2D
    ((100,), (16,), (10, 10), (3, 3)),                          # 1D -> 2D
])
def test_reshape_write_matches_numpy(shape, chunks, target, tchunks, tmp_path):
    arr = np.random.default_rng(6).random(shape).astype(np.float32)
    da = DynamicArray(zarr.array(arr, chunks=chunks))
    out = str(tmp_path / "r.zarr")
    reshape_write(da, target, out, output_chunks=tchunks, zarr_format=2)
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr.reshape(target))


@pytest.mark.parametrize("max_mem", [1 << 20, 4096, 256])       # force deep flatten split
def test_reshape_write_tiny_budget(max_mem, tmp_path):
    arr = np.random.default_rng(7).random((6, 5, 4, 7)).astype(np.float32)
    da = DynamicArray(zarr.array(arr, chunks=(2, 2, 2, 3)))
    out = str(tmp_path / "r.zarr")
    reshape_write(da, (24, 35), out, output_chunks=(5, 8), max_mem=max_mem, zarr_format=2)
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr.reshape(24, 35))


def test_iowrite_reshape_routes_to_staged(tmp_path):
    arr = np.random.default_rng(8).random((12, 10, 8, 6)).astype(np.float32)
    da = DynamicArray(zarr.array(arr, chunks=(5, 4, 3, 4)))
    out = str(tmp_path / "r.zarr")
    io.write(ops.reshape(da, (80, 72)), out, zarr_format=2, chunks=(16, 3))
    np.testing.assert_array_equal(zarr.open(out, mode="r")[:], arr.reshape(80, 72))


@pytest.mark.skipif(os.environ.get("DYNA_LARGE") != "1",
                    reason="5GB flatten; set DYNA_LARGE=1 to run (slow, needs ~10GB disk)")
def test_flatten_5gb_5d_memory_bounded(tmp_path):
    """Real large-data check: 5D float32 ~5GB, chunks (1,1,32,32,32). Flatten via io.write must
    stay memory-bounded (peak << data size) and be correct (spot-checked, can't hold 5GB in RAM)."""
    import psutil
    shape = (10, 4, 256, 256, 512)                       # 1.34e9 elems * 4B = 5.37 GB
    chunks = (1, 1, 32, 32, 32)
    assert all(s % c == 0 for s, c in zip(shape[2:], chunks[2:]))
    data_gb = np.prod(shape) * 4 / 1e9

    src = str(tmp_path / "src.zarr")
    z = zarr.open(src, mode="w", shape=shape, chunks=chunks, dtype="float32")
    rng = np.random.default_rng(0)
    for t in range(shape[0]):                            # fill one (z,y,x) volume at a time
        for c in range(shape[1]):
            z[t, c] = rng.random((shape[2], shape[3], shape[4]), dtype=np.float32)

    out = str(tmp_path / "flat.zarr")
    proc = psutil.Process()
    base = proc.memory_info().rss
    peak = [base]
    stop = [False]

    def mon():
        while not stop[0]:
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.01)

    th = threading.Thread(target=mon)
    th.start()
    io.write(ops.flatten(DynamicArray(z)), out, zarr_format=2, chunks=(1 << 22,))
    stop[0] = True
    th.join()

    peak_gb = (peak[0] - base) / 1e9
    print(f"\n5GB flatten: data {data_gb:.2f}GB  peakDelta {peak_gb:.2f}GB")

    of = zarr.open(out, mode="r")
    N = int(np.prod(shape))
    assert of.shape == (N,)
    # spot-check correctness: random flat indices vs source at unravelled coords
    idx = np.sort(rng.integers(0, N, size=500))
    coords = np.unravel_index(idx, shape)
    expect = np.array([z[tuple(int(coords[d][i]) for d in range(len(shape)))]
                       for i in range(len(idx))], dtype=np.float32)
    np.testing.assert_array_equal(of.oindex[idx], expect)

    # memory must be bounded well under the data size (budget ~256MB + overheads, not O(5GB))
    assert peak_gb < 1.5, f"flatten peak {peak_gb:.2f}GB not bounded vs {data_gb:.2f}GB data"
