"""dyna_zarr.from_array: wrap any array-like source - numpy, zarr, TensorStore, dask,
DynamicArray, micro_reader.Image and duck-typed readers.

The contract a source must meet - shape, dtype, __getitem__ returning NumPy for a tuple
of slices - is what eubi-bridge's region sources and micro_reader.Image already offer.
These tests pin it: grids are taken from `chunks`, else micro-reader's `read_unit`; a
malformed advertised grid is ignored rather than trusted; `lock=` serialises reads of
sources that are not thread-safe; the source is never probed for zarr settings.
"""
import contextlib
import io as _stdio
import pickle
import threading
import time

import numpy as np
import pytest
import zarr

import dyna_zarr
from dyna_zarr import DynamicArray, io, from_array

DATA = np.random.default_rng(0).integers(0, 999, (2, 16, 120, 90), dtype="uint16")


class Duck:
    """A reader-style source; counts reads and the peak number of concurrent reads."""

    def __init__(self, chunks=None, read_unit=None, delay=0.0):
        self.shape, self.dtype = DATA.shape, DATA.dtype
        if chunks is not None:
            self.chunks = chunks
        if read_unit is not None:
            self.read_unit = read_unit
        self.delay = delay
        self.reads, self._active, self.peak = 0, 0, 0
        self._guard = threading.Lock()

    def __getitem__(self, key):
        with self._guard:
            self.reads += 1
            self._active += 1
            self.peak = max(self.peak, self._active)
        try:
            time.sleep(self.delay)
            return DATA[key]
        finally:
            with self._guard:
                self._active -= 1

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("_guard")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._guard = threading.Lock()


class MetadataBomb(Duck):
    """Like a micro_reader.Image: `.metadata` would read the file. Must not be touched."""

    @property
    def metadata(self):
        raise AssertionError("from_array probed .metadata")


def _write(arr, path, **kw):
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(arr, str(path), **kw)


def test_one_public_name():
    assert dyna_zarr.from_array is from_array
    assert not hasattr(DynamicArray, "from_array")          # no second spelling


@pytest.mark.parametrize("src,grid", [
    (Duck(chunks=(1, 4, 64, 64)), (1, 4, 64, 64)),                  # its own chunks
    (Duck(read_unit=(1, 1, 32, 90)), (1, 1, 32, 90)),               # micro-reader read_unit
    (Duck(chunks=((1, 1), (8, 8), (60, 60), (90,))), None),         # dask-style: ignored
    (Duck(), None),                                                 # no grid at all
    (Duck(chunks=(1, 64, 512, 512)), (1, 16, 120, 90)),             # clamped to the shape
])
def test_grid_resolution(src, grid):
    assert from_array(src).chunks == grid


def test_a_streamed_read_unit_is_not_taken_as_the_grid():
    # micro_reader decodes blocks over 16 MB in parts (streamed rows / decoded areas),
    # so such a read_unit (e.g. a 64 MB one-strip plane) is not a real read unit; taken
    # as the grid it would become a 64 MB output chunk and set every region's minimum.
    big = Duck(read_unit=(2, 16, 120, 90))                   # whole array, ~0.7 MB: fine
    assert from_array(big).chunks == (2, 16, 120, 90)

    class Huge(Duck):
        def __init__(self):
            super().__init__()
            self.shape = (1, 8192, 4096)
            self.read_unit = (1, 8192, 4096)                # 64 MB uint16: streamed
    assert from_array(Huge()).chunks is None


def test_explicit_chunks_win_and_are_validated():
    assert from_array(Duck(chunks=(1, 4, 64, 64)), chunks=(2, 8, 32, 32)).chunks == (2, 8, 32, 32)
    with pytest.raises(ValueError, match="one positive int per axis"):
        from_array(Duck(), chunks=(8, 8))


def test_reads_slices_ops_and_writes(tmp_path):
    src = Duck(chunks=(1, 8, 64, 64))
    a = from_array(src)
    np.testing.assert_array_equal(np.asarray(a[1, 3:9, 10:70, 5]), DATA[1, 3:9, 10:70, 5])
    np.testing.assert_array_equal(np.asarray((a * 2 + 1)[0, :2]), DATA[0, :2] * 2 + 1)
    _write(a, tmp_path / "out.zarr")
    z = zarr.open_array(str(tmp_path / "out.zarr"))
    np.testing.assert_array_equal(z[...], DATA)
    assert tuple(z.chunks) == (1, 8, 64, 64)               # the source grid is kept
    assert z.metadata.zarr_format == 3                     # no format of its own: defaults


def test_the_source_is_not_probed_for_zarr_settings(tmp_path):
    a = from_array(MetadataBomb(chunks=(1, 8, 64, 64)))    # would raise on .metadata
    _write(a, tmp_path / "out.zarr")
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "out.zarr"))[...], DATA)


def test_lock_serialises_reads(tmp_path):
    free, locked = Duck(chunks=(1, 4, 60, 90), delay=0.02), Duck(chunks=(1, 4, 60, 90), delay=0.02)
    _write(from_array(free), tmp_path / "free.zarr", max_workers=8, region_shape=(1, 4, 60, 90))
    _write(from_array(locked, lock=True), tmp_path / "locked.zarr", max_workers=8,
           region_shape=(1, 4, 60, 90))
    assert free.peak > 1                    # without a lock, regions are read concurrently
    assert locked.peak == 1                 # with one, never two reads at once
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "locked.zarr"))[...], DATA)


def test_a_shared_lock_spans_sources():
    shared = threading.Lock()
    a, b = from_array(Duck(), lock=shared), from_array(Duck(), lock=shared)
    assert a._zarr_array._lock is b._zarr_array._lock is shared


def test_pickles_with_a_fresh_lock():
    a = from_array(Duck(chunks=(1, 4, 64, 64)), lock=True)
    b = pickle.loads(pickle.dumps(a))
    assert b.chunks == (1, 4, 64, 64)
    assert b._zarr_array._lock is not a._zarr_array._lock
    np.testing.assert_array_equal(np.asarray(b[0, 0]), DATA[0, 0])


def test_known_types_pass_through_and_refuse_lock():
    z = zarr.array(DATA, chunks=(1, 8, 64, 64))
    a = from_array(z)
    assert a.chunks == (1, 8, 64, 64) and a.zarr_format == z.metadata.zarr_format
    assert from_array(DATA, chunks=(1, 8, 60, 45)).chunks == (1, 8, 60, 45)
    assert from_array(a) is not a and from_array(a).shape == a.shape
    with pytest.raises(ValueError, match="lock="):
        from_array(z, lock=True)


def test_not_array_like_is_refused():
    with pytest.raises(TypeError, match="not array-like"):
        from_array(object())


def test_dask_arrays_are_computed_per_region(tmp_path):
    da = pytest.importorskip("dask.array")
    x = da.from_array(DATA, chunks=(1, 8, 64, 64)) * 2 + 1
    a = from_array(x)
    assert a.chunks == (1, 8, 64, 64)                       # dask's chunksize, per axis
    region = np.asarray(a[1, 2:9, 10:70, 3])
    assert isinstance(region, np.ndarray)                   # materialized, not a dask slice
    np.testing.assert_array_equal(region, (DATA * 2 + 1)[1, 2:9, 10:70, 3])
    _write(a + 1, tmp_path / "out.zarr", max_workers=8)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "out.zarr"))[...],
                                  DATA * 2 + 2)
    np.testing.assert_array_equal(np.asarray(pickle.loads(pickle.dumps(a))[0, 0]),
                                  (DATA * 2 + 1)[0, 0])
    with pytest.raises(ValueError, match="lock="):
        from_array(x, lock=True)


def test_the_bare_constructor_still_refuses_dask():
    da = pytest.importorskip("dask.array")
    with pytest.raises(TypeError, match="from_array"):
        DynamicArray(da.zeros((4, 4)))


class SpanLogger(Duck):
    """Records the number of elements each source call spans (what a reader such as
    micro_reader allocates for a strided key: the whole contiguous span)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.spans = []

    def __getitem__(self, key):
        n = 1
        for k, s in zip(key, self.shape):
            if isinstance(k, slice):
                start, stop, _ = k.indices(s)
                n *= max(0, stop - start)        # the SPAN, steps ignored
        self.spans.append(n)
        return super().__getitem__(key)


@pytest.mark.parametrize("sel", [
    (0, slice(None), slice(None, None, 4), slice(None, None, 4)),
    (slice(None), slice(None), slice(None, None, 8), slice(None, None, 8)),
    (1, slice(3, 15, 5), slice(7, None, 3), slice(1, 89, 6)),
    (1, 7, slice(100, None, 9), slice(5, None, 4)),
])
def test_strided_reads_are_split_into_bounded_blocks(sel):
    # Before: one strided key went to the source as is, and a reader that decodes the
    # contiguous span (micro_reader) allocated step**n x the output - 4 GB for an 8 MB
    # region at 8x on z/y/x. Now each source call spans at most one grid unit per
    # strided axis (times the requested extent of the unit-step axes).
    src = SpanLogger(chunks=(1, 4, 32, 32))
    out = np.asarray(from_array(src)[sel])
    np.testing.assert_array_equal(out, DATA[sel])
    whole = int(np.prod([len(range(*s.indices(n))) if isinstance(s, slice) and (s.step or 1) == 1
                         else (1 if not isinstance(s, slice) else 32)
                         for s, n in zip(sel, DATA.shape)]))
    assert max(src.spans) <= whole, (max(src.spans), whole)


def test_micro_reader_image(tmp_path):
    micro_reader = pytest.importorskip("micro_reader")
    tifffile = pytest.importorskip("tifffile")
    plane = DATA[0, 0]
    path = tmp_path / "img.tif"
    tifffile.imwrite(str(path), plane, tile=(64, 64), compression="zlib")   # a compressed tile
    img = micro_reader.open(str(path)).images[0]
    a = from_array(img)
    assert a.shape == plane.shape
    if img.read_unit is not None:                       # the smallest block it decodes
        assert a.chunks == tuple(min(u, s) for u, s in zip(img.read_unit, plane.shape))
    np.testing.assert_array_equal(np.asarray(a[10:70, 5:80]), plane[10:70, 5:80])
    _write(a, tmp_path / "out.zarr")
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "out.zarr"))[...], plane)
