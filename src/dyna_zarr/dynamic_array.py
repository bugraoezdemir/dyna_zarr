"""
dynamic_array: A lightweight library for lazy operations on Zarr arrays
without the overhead of task graphs.
"""

import zarr
import numpy as np
from typing import Tuple, Union, List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
from pathlib import Path
import gc
import time
import threading
import json
from queue import Queue, Empty

# Import Transform base class from operations module
from .operations import Transform
from .codecs import Codecs

        


def _maybe_collect(state, interval=5):
    """Run a full GC at most once every `interval` seconds to avoid frequent small collections."""
    last = state.get('_last_gc_time', 0)
    now = time.time()
    if now - last >= interval:
        gc.collect()
        state['_last_gc_time'] = now


def _ndarray_to_memory_zarr(arr: np.ndarray, chunks=None) -> zarr.Array:
    """Stage a numpy array as an in-memory zarr array, so it has a real chunk grid.

    `chunks=None` picks a ~32 MiB chunk over the trailing (fastest-varying) axes, clamped to
    the array - big enough not to fragment a small array, small enough that a region read is
    still a partial read. Everything downstream then sees the same interface a file-backed
    zarr presents.
    """
    if chunks is None:
        target = 32 * 1024 * 1024 // max(1, arr.dtype.itemsize)
        chunks = list(arr.shape)
        for axis in range(arr.ndim):                      # shrink leading axes first
            if int(np.prod(chunks)) <= target:
                break
            others = int(np.prod(chunks[axis + 1:])) or 1
            chunks[axis] = max(1, min(chunks[axis], target // others))
        # a zero-length axis still needs a chunk side of at least 1: zarr >= 3.4 refuses 0
        chunks = tuple(max(1, int(c)) for c in chunks)
    else:
        chunks = tuple(int(c) for c in chunks)
    z = zarr.create_array(store={}, shape=arr.shape, chunks=chunks, dtype=arr.dtype)
    z[...] = arr
    return z


def _reject_dask(source) -> None:
    """Refuse a dask array as a DynamicArray source, loudly.

    Wrapping one half-works, which is worse than not working: `_read_direct` hands back a
    dask array instead of materialized data, so most ops still succeed (scipy calls
    `np.asarray` on the block internally) while two things break in confusing ways -
    `np.asarray(arr[1:3])` raises "object __array__ method not producing an array", and
    `.chunks` reports dask's tuple-of-tuples `((2,2),(4,4))` where every consumer here
    expects a per-axis shape `(2,4)`, silently misfeeding the region/tile machinery.

    Supporting it properly means materializing each region's dask slice on read and
    reporting `chunksize` as the grid, which is what `from_array` does. The bare
    constructor keeps refusing, so a dask array only ever enters through that one
    explicit, correct path.
    """
    if _is_dask(source):
        raise TypeError(
            f"DynamicArray() cannot wrap a {type(source).__module__}.{type(source).__name__} "
            "directly: it would hand back dask slices and dask-style .chunks. Use "
            "dyna_zarr.from_array(x), which computes each region's slice on read, or push "
            "the dask array into io.create_sink(...) with dask.array.store."
        )


def _is_dask(source) -> bool:
    return type(source).__module__.split(".")[0] == "dask"


class _SourceAdapter:
    """An array source as DynamicArray reads it through `from_array`.

    Exposes only ``shape`` / ``dtype`` / ``ndim`` / ``__getitem__``, so the constructor
    does not probe the source for zarr settings (a ``micro_reader.Image`` has a
    ``.metadata`` property that READS THE FILE). A dask source has each region's slice
    computed on read - synchronously, because dyna's own worker threads are the
    parallelism (a threaded compute inside each would nest thread pools). With a lock,
    every read holds it, for sources that are not safe to read from several threads at
    once. A lock cannot be pickled, so an unpickled adapter gets a fresh one (one per
    process, which is what a per-process reader needs); the source itself must pickle.
    """

    def __init__(self, source, lock=None):
        self._source = source
        self._lock = lock
        self._dask = _is_dask(source)
        self.shape = tuple(int(s) for s in source.shape)
        self.dtype = source.dtype
        self.ndim = len(self.shape)

    def _read(self, key):
        block = self._source[key]
        if self._dask:
            return np.asarray(block.compute(scheduler="synchronous"))
        return np.asarray(block)

    def __getitem__(self, key):
        if self._lock is None:
            return self._read(key)
        with self._lock:
            return self._read(key)

    def __reduce__(self):
        # the lock is created on the UNPICKLING side; a lock object in the reduce
        # arguments would itself have to be pickled, which is impossible
        return (_restore_source_adapter, (self._source, self._lock is not None))

    def __repr__(self):
        locked = ", locked" if self._lock is not None else ""
        return f"<source {self._source!r}{locked}>"


def _restore_source_adapter(source, locked):
    """Unpickle a _SourceAdapter: a fresh lock in this process if it had one."""
    import threading
    return _SourceAdapter(source, threading.Lock() if locked else None)


def _per_axis_grid(value, ndim):
    """``value`` as one positive int per axis, or None if it is not such a grid
    (absent, wrong rank, dask's tuple-of-tuples, non-positive)."""
    if value is None:
        return None
    try:
        grid = tuple(int(v) for v in value)
    except (TypeError, ValueError):
        return None
    if len(grid) != ndim or any(v < 1 for v in grid):
        return None
    return grid


class _PersistOwner:
    """Owns a persist() temp directory; deletes it when the last array holding it goes.

    Every DynamicArray derived from a persisted one (a copy, or a lazy op on it) carries
    a reference to this object - copies directly, ops through their upstream array - so
    the directory outlives every reader of it, not just the array persist() returned.
    """

    def __init__(self, tmpdir):
        import shutil
        import weakref
        self.tmpdir = tmpdir
        weakref.finalize(self, shutil.rmtree, tmpdir, True)


#: Stored as a DynamicArray's shape when it is only known once the data is read.
_DEFERRED_SHAPE = object()


def _shape_of(transform):
    """The shape to store for a transform's output: deferred when the transform says its
    shape depends on the data (``shape_deferred``), so building the array reads nothing."""
    return _DEFERRED_SHAPE if getattr(transform, "shape_deferred", False) else transform.shape


def _checked_chunks(transform):
    """``transform.chunks`` as a tuple of ints, after checking it can describe its output.

    Every lazy op's grid passes through here, so the invariant holds for every op,
    including ones added later: a grid is either None (no grid - the writer then picks
    its default chunk) or one positive integer per axis of ``transform.shape``. Anything
    else is a bug in that transform, and it raises rather than reach the writer, which
    would otherwise act on it (an invented (1, 1, 1) grid became one-voxel chunks on
    disk; a (1,) grid was reported for a 0-d squeeze).
    """
    chunks = transform.chunks
    if chunks is None:
        return None
    shape = tuple(transform.shape)
    try:
        chunks = tuple(int(c) for c in chunks)
    except (TypeError, ValueError):
        chunks = None
    if chunks is None or len(chunks) != len(shape) or any(c < 1 for c in chunks):
        raise ValueError(
            f"internal error: {type(transform).__name__} reported chunks "
            f"{transform.chunks!r} for an output of shape {shape}. A grid must be None or "
            f"one positive integer per axis; please report this as a dyna_zarr bug.")
    return chunks


def _as_numpy_dtype(dt):
    """``dt`` as a real ``numpy.dtype``, whatever produced it.

    TensorStore arrays report a ``tensorstore.dtype``: it prints as ``dtype("float32")``
    and even compares equal to ``np.float32``, but NumPy cannot interpret it, so
    ``np.zeros(shape, dt)`` / ``np.dtype(dt)`` / ``.itemsize`` all raise. It carries its
    exact NumPy equivalent as ``.numpy_dtype``; anything else goes through parse_dtype.
    """
    if dt is None or isinstance(dt, np.dtype):
        return dt
    numpy_dtype = getattr(dt, "numpy_dtype", None)
    if numpy_dtype is not None:
        return np.dtype(numpy_dtype)
    from .utils import parse_dtype
    return parse_dtype(dt)[0]


class DynamicArray:
    """
    Wrapper around Zarr arrays or TensorStore arrays that enables lazy operations.

    Supports multiple input types:
    - zarr.Array: Wraps for lazy operations
    - TensorStore arrays: Wraps for lazy operations
    - DynamicArray: Copy constructor

    To read from files, use operations.read(path) instead.

    ``dtype`` is always a ``numpy.dtype``, whatever the source.
    """

    # The dtype is normalised where it is STORED, not where it is read: every path that
    # builds a DynamicArray assigns `_dtype` (the constructors, the readers' object.__new__
    # paths, _with_transform), so a setter is the one place that covers all of them,
    # including sources added later. Per-use parse_dtype() calls elsewhere predate this.
    @property
    def _dtype(self):
        return self._np_dtype

    @_dtype.setter
    def _dtype(self, value):
        self._np_dtype = _as_numpy_dtype(value)

    # The shape is stored the same way, so ONE place handles a shape that is not known
    # until the data is read (ops.unique: its length depends on the data). Such an array
    # stores _DEFERRED_SHAPE and asks its transform on every access; the transform computes
    # its result once, so the first shape need - .shape, .size, compute, write, or building
    # another op on it - is the one pass, and every internal shape read goes through here.
    @property
    def _shape(self):
        shape = self._shape_value
        return tuple(self._transform.shape) if shape is _DEFERRED_SHAPE else shape

    @_shape.setter
    def _shape(self, value):
        self._shape_value = value if value is _DEFERRED_SHAPE else tuple(value)

    def __init__(self, source: Union[zarr.Array, np.ndarray, 'DynamicArray'],
                 chunks: Optional[Tuple[int, ...]] = None):
        _reject_dask(source)
        from_numpy = isinstance(source, np.ndarray)
        if from_numpy:
            # Route numpy through an in-memory zarr array rather than wrapping it raw.
            # Wrapping raw "works" but leaves `.chunks` as None, and chunks are load-bearing
            # here: they drive the tile/region defaults (tilewise-ccl's default_tile_shape
            # snaps to them) and the writer's alignment inference. Going through zarr gives
            # a real chunk grid and makes the numpy case behave like every other source -
            # one code path instead of a special case. Costs one copy into the MemoryStore.
            source = _ndarray_to_memory_zarr(source, chunks)
        if isinstance(source, DynamicArray):
            # Copy constructor
            self._source = source._source
            self._zarr_array = source._zarr_array
            self._ts_array = source._ts_array
            self._is_tensorstore = source._is_tensorstore
            self._shape = source._shape_value       # keeps a deferred shape deferred
            self._chunks = source._chunks
            self._dtype = source._dtype
            self._transform = source._transform
            self._zarr_format = source._zarr_format
            self._compressor = source._compressor
            self._compressors = source._compressors
            self._shards = source._shards
            self._codecs = source._codecs
            self._persist_owner = getattr(source, "_persist_owner", None)
        else:
            # Wrap a real Zarr array or TensorStore array
            # Check if it's tensorstore
            if hasattr(source, 'read') and hasattr(source, 'spec'):
                # It's a tensorstore array
                self._is_tensorstore = True
                self._ts_array = source
                self._zarr_array = None
            else:
                # It's a zarr array
                self._is_tensorstore = False
                self._ts_array = None
                self._zarr_array = source

            self._source = source
            self._shape = tuple(source.shape)
            self._chunks = getattr(source, 'chunks', None)
            self._dtype = source.dtype
            self._transform = None
            if not self._is_tensorstore and not from_numpy:
                self._extract_zarr_metadata(source)
            else:
                # A TensorStore handle carries no zarr metadata object; a numpy array has
                # no storage format at all - its in-memory staging store is an internal
                # detail (zarr's defaults, e.g. zstd level 0), not something to inherit.
                self._zarr_format = None
                self._compressor = None
                self._compressors = None
                self._shards = None
                self._codecs = None
                if self._is_tensorstore:
                    from .io import _ts_zarr_format
                    self._zarr_format = _ts_zarr_format(source)
                    if self._zarr_format is not None:
                        # a STORED zarr array (not a virtual view such as ts.downsample):
                        # its spec carries the codecs and shards, its chunk layout the
                        # grid - the same inheritable settings a zarr-python source gives
                        self._inherit_ts_storage(source)

    def _inherit_ts_storage(self, ts_array):
        """Chunk grid, codecs and shards of a stored TensorStore zarr array.

        Without them a TensorStore source reported ``chunks=None`` and no codec, so
        io.write fell back to its default chunking and blosc - a TensorStore level (e.g.
        a deferred ``downscale`` view in ome_zarr_pyramid) lost its storage settings on
        the way through. Anything unreadable stays None ("no preference").
        """
        from .io import _storage_settings_from_metadata
        try:
            md = ts_array.spec().to_json().get("metadata")
        except Exception:
            md = None
        _, codecs, shards = _storage_settings_from_metadata(md)
        self._codecs = codecs
        if shards is not None and len(shards) == len(self._shape):
            self._shards = shards
        if self._chunks is None:
            try:
                read_chunk = tuple(int(c) for c in ts_array.chunk_layout.read_chunk.shape)
            except Exception:
                read_chunk = None
            if read_chunk and len(read_chunk) == len(self._shape) and all(c > 0 for c in read_chunk):
                self._chunks = read_chunk

    def _extract_zarr_metadata(self, zarr_array):
        """Storage format, compression and sharding of a zarr source, from zarr 3's API.

        These are what io.write INHERITS when the caller does not say otherwise, so
        they must be right or absent - never guessed. The previous version probed
        `_version` / `store.path` (zarr 2 era): under zarr 3 it reported every array,
        v3 included, as v2, so a v3 input's codecs and shards were never inherited.
        Anything that cannot be read reliably is left as None, which io.write treats
        as "no preference" (format -> v3, codecs -> the default).
        """
        md = getattr(zarr_array, "metadata", None)
        fmt = getattr(md, "zarr_format", None)
        self._zarr_format = fmt if fmt in (2, 3) else None
        try:
            comps = tuple(getattr(zarr_array, "compressors", None) or ())
        except Exception:
            comps = ()
        self._compressors = comps or None
        self._compressor = comps[0] if (self._zarr_format == 2 and comps) else None
        shards = getattr(zarr_array, "shards", None) if self._zarr_format == 3 else None
        self._shards = tuple(int(s) for s in shards) if shards else None
        self._codecs = self._codecs_from_compressors(comps) if self._zarr_format else None

    def _codecs_from_compressors(self, comps):
        """``Codecs`` equivalent of a zarr array's ``compressors``, or None if unknown.

        An array stored WITHOUT compression yields ``Codecs(None)``, so an uncompressed
        input stays uncompressed; an unrecognised compressor yields None (the default).
        """
        if not comps:
            return Codecs(compressor=None)
        first = comps[0]
        to_dict = getattr(first, "to_dict", None)
        if to_dict is not None:                              # a zarr v3 codec
            return self._extract_codecs_from_v3([to_dict()])
        try:                                                 # a numcodecs (v2) codec
            return Codecs.from_numcodecs(first)
        except Exception:
            return None

    @staticmethod
    def _extract_codecs_from_v3(codecs_list):
        """Extract Codecs instance from Zarr v3 codec pipeline (a list of codec dicts)."""
        if codecs_list is None:
            return None

        # v3 serializes blosc's shuffle by NAME; Codecs uses numcodecs' ints.
        shuffle_ids = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}

        # Look for compression codec in the pipeline
        for codec in codecs_list:
            codec_name = codec.get('name', '')
            config = codec.get('configuration', {})

            if codec_name == 'blosc':
                shuffle = config.get('shuffle', 1)
                return Codecs(
                    compressor='blosc',
                    clevel=config.get('clevel', 5),
                    cname=config.get('cname', 'lz4'),
                    shuffle=shuffle_ids.get(shuffle, shuffle),
                )
            elif codec_name == 'zstd':
                return Codecs(
                    compressor='zstd',
                    clevel=config.get('level', 5)
                )
            elif codec_name == 'gzip':
                return Codecs(
                    compressor='gzip',
                    clevel=config.get('level', 5)
                )
            elif codec_name == 'lz4':
                return Codecs(compressor='lz4')
            elif codec_name == 'bz2':
                return Codecs(
                    compressor='bz2',
                    clevel=config.get('level', 5)
                )

        # No compression codec found
        return None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def chunks(self) -> Tuple[int, ...]:
        """Chunk grid of THIS array, or None when it has no meaningful one.

        A transform is authoritative about its own grid, including when it says
        None: an op that rewrites the index space (reshape, flatten) leaves no
        grid behind. The underlying zarr array is consulted only for an
        UNTRANSFORMED view of it, and even then only if the rank still matches -
        the source tuple is copied along the whole chain, so without these guards
        a 2-D `reshape((64,128,128) -> (64,16384))` reported the source's rank-3
        `(8,64,64)`. Downstream that is worse than None, because the chunk grid
        drives the writer's region and alignment defaults: None is handled, a
        wrong-rank tuple is not.
        """
        if self._chunks is not None:
            return self._chunks
        if self._transform is not None:
            return None
        if self._zarr_array is not None and hasattr(self._zarr_array, 'chunks'):
            chunks = self._zarr_array.chunks
            if chunks is not None and len(chunks) == len(self._shape):
                return tuple(int(c) for c in chunks)
        # TensorStore, unknown, or a rank that no longer matches.
        return None

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self) -> int:
        if self._shape_value is _DEFERRED_SHAPE:     # known without the data (unique: 1)
            return self._transform.ndim
        return len(self._shape)

    # numpy/dask array attributes derived from shape and dtype alone - they never read
    # data, so they are free on any lazy array.
    @property
    def size(self) -> int:
        """Number of elements (1 for a 0-d array), like ``numpy.ndarray.size``."""
        return int(np.prod(self._shape, dtype=np.int64))

    @property
    def itemsize(self) -> int:
        """Bytes per element, like ``numpy.ndarray.itemsize``."""
        return int(self.dtype.itemsize)

    @property
    def nbytes(self) -> int:
        """Bytes the materialized array occupies (uncompressed), like ``numpy.ndarray.nbytes``."""
        return self.size * self.itemsize

    @property
    def zarr_format(self) -> int:
        return self._zarr_format

    @property
    def compressor(self):
        """Compressor for Zarr v2."""
        return self._compressor

    @property
    def compressors(self):
        """Compressors for Zarr v3."""
        return self._compressors

    @property
    def shards(self):
        """Shards for Zarr v3."""
        return self._shards

    @property
    def codecs(self):
        """Unified compression configuration (Codecs instance)."""
        return self._codecs

    @property
    def is_tensorstore(self) -> bool:
        """True if backed by TensorStore, False if backed by zarr."""
        return self._is_tensorstore

    @property
    def array(self):
        """Get the underlying array (zarr.Array or tensorstore.TensorStore)."""
        if self._is_tensorstore:
            return self._ts_array
        else:
            return self._zarr_array

    def __getitem__(self, key):
        """
        Create a lazy slice of the array.
        For immediate execution, use .compute() method.
        """
        # Create a lazy slice transform
        transform = SliceTransform(self, key)
        return self._with_transform(transform)
    
    def compute(self, device=None):
        """
        Execute all lazy transforms and return the result as a numpy array.

        ``device`` sets the EXECUTION device for inherit-ops (device=None) in the chain:
        None/'cpu' runs on the CPU (default), 'cuda' runs the pipeline on the GPU. The
        result is always returned as a host numpy array.
        """
        from .operations._backend import device_context, to_device
        from .operations.reductions import evaluate_small_reductions
        with device_context(device):
            # Small reductions the chain needs are computed first, fused per input
            # (x.mean() and x.std() share one pass), then served from the cache.
            evaluate_small_reductions(self)
            if self._transform is None:
                # No transform - read entire array. `[...]`, not `[:]`: a 0-d array
                # has no axis for `:` to index (a persisted 0-d result is one).
                if self._is_tensorstore:
                    result = self._ts_array.read().result()
                else:
                    result = self._zarr_array[...]
            else:
                # Apply transformation to read all data
                full_slice = tuple(slice(None) for _ in range(len(self.shape)))
                result = self._transform.read(full_slice)
        return to_device(result, "cpu")

    # --- numpy/dask-like method surface (lazy; route to operations) ---
    def astype(self, dtype):
        """Lazily cast to ``dtype`` (like ``numpy``/``dask`` ``a.astype``)."""
        from . import operations as o
        return o.astype(self, dtype)

    def clip(self, a_min=None, a_max=None, out=None):
        """Lazily clip to ``[a_min, a_max]`` (either bound may be None). ``out`` accepted for
        NumPy-method compatibility (``np.clip`` calls ``a.clip(min, max, out=...)``); a non-None
        ``out`` isn't supported on a lazy array."""
        if out is not None:
            raise TypeError("clip(out=...) is not supported on a lazy DynamicArray")
        from . import operations as o
        return o.clip(self, a_min, a_max)

    def round(self, decimals=0, out=None):
        """Lazily round (like ``numpy``/``dask`` ``a.round``). ``out`` accepted for NumPy-method
        compatibility; a non-None ``out`` isn't supported on a lazy array."""
        if out is not None:
            raise TypeError("round(out=...) is not supported on a lazy DynamicArray")
        from . import operations as o
        return o.round(self, decimals)

    def reshape(self, *shape):
        """Lazily reshape (C-order), like ``numpy``/``dask`` ``a.reshape``. Accepts either a
        single shape tuple or separate int args. On ``io.write`` an outermost reshape is
        streamed memory-bounded (disk-staged); see operations.reshape."""
        from . import operations as o
        if len(shape) == 1 and not np.isscalar(shape[0]):
            shape = tuple(shape[0])
        return o.reshape(self, shape)

    def flatten(self):
        """Lazily flatten to 1D (C-order), like ``numpy`` ``a.flatten``. Memory-bounded on
        ``io.write`` (disk-staged); see operations.flatten."""
        from . import operations as o
        return o.flatten(self)

    def rechunk(self, chunks=None, **kwargs):
        """No-op for the pull model (accepts dask's signature for backend compatibility).

        In dask, ``rechunk`` changes the chunk grid so cross-chunk ops behave. A DynamicArray
        is chunk-invariant: every lazy read pulls exactly the region asked for, independent of
        any chunk grid, and streaming reductions/scans already see whole axes. So a lazy
        rechunk changes nothing about correctness and returns the array unchanged. Storage
        chunking is a separate concern, set via ``io.write(chunks=...)`` or the rechunk engine.
        """
        return self

    def persist(self, path=None, **kwargs):
        """Compute this array ONCE and return a DynamicArray that reads the stored result.

        Use it on an intermediate that would otherwise be recomputed, above all a large
        partial reduction broadcast back over the array (``x - x.mean(axis=0)`` written
        region by region re-streams the mean once per region along axis 0; ``io.write``
        warns when that happens). The result is written with ``io.write``, so it is
        memory-bounded at any size.

        ``path`` - where to store it. Omitted, a result of at most 1 MiB is kept in
        memory and anything larger goes to a temporary Zarr store, deleted when the
        last array reading it is garbage-collected. Given, the store is written there
        and left in place; an existing store there raises, as in io.write. Extra
        keyword arguments (dask's ``scheduler=`` etc.) are accepted and ignored.
        """
        import tempfile
        from .io import io as _io
        from .utils import parse_dtype

        nbytes = int(np.prod(self._shape, dtype=np.int64)) * parse_dtype(self._dtype)[0].itemsize
        if path is None and nbytes <= (1 << 20):
            return DynamicArray(np.asarray(self.compute()))
        owner = None
        if path is None:
            tmpdir = tempfile.mkdtemp(prefix="dyna_persist_")
            path = str(Path(tmpdir) / "persisted.zarr")
            owner = _PersistOwner(tmpdir)
        _io.write(self, str(path))
        result = _io.read(str(path))
        if owner is not None:
            result._persist_owner = owner
        return result

    def map_blocks(self, func, *args, dtype=None, device=None, **kwargs):
        """dask-compatible ``map_blocks``: apply ``func`` blockwise (shape-preserving). Extra
        array/scalar ``args`` become additional equally-shaped operands; dask-only kwargs
        (``meta``/``chunks``/``name``/``block_info``/...) are ignored, and any remaining kwargs
        are bound to ``func``. ``drop_axis``/``new_axis`` (shape-changing) are not supported.
        ``device`` (None=inherit the execution context, 'cpu', 'cuda') runs it on that device -
        the block is moved there before ``func``, which dispatches on the block's array module."""
        from . import operations as o
        if kwargs.get("drop_axis") is not None or kwargs.get("new_axis") is not None:
            raise NotImplementedError(
                "map_blocks drop_axis/new_axis is not supported (shape-preserving only)")
        for k in ("meta", "chunks", "name", "token", "drop_axis", "new_axis",
                  "block_info", "block_id", "enforce_ndim"):
            kwargs.pop(k, None)
        f = (lambda *bs, _f=func, _kw=kwargs: _f(*bs, **_kw)) if kwargs else func
        return o.map_blocks(f, self, *args, dtype=dtype, device=device)

    def map_overlap(self, func, depth=0, boundary="reflect", trim=True, dtype=None,
                    device=None, **kwargs):
        """dask-compatible ``map_overlap``: apply a shape-preserving neighbourhood ``func``
        with a ``depth`` halo. dask-only kwargs are ignored and any remaining kwargs are bound
        to ``func``. dask's boundary names are translated to the scipy.ndimage names the
        pull-model map_overlap uses. ``trim=False`` (shrinking output) is not supported.
        ``device`` (None=inherit, 'cpu', 'cuda') runs it on that device - the haloed block is
        moved there before ``func``, which dispatches ndimage on the block's array module."""
        from . import operations as o
        if not trim:
            raise NotImplementedError("map_overlap trim=False is not supported")
        for k in ("meta", "chunks", "name"):
            kwargs.pop(k, None)
        if isinstance(boundary, (int, float)):          # dask allows a numeric fill boundary
            boundary = "constant"                       # -> constant-pad the halo (fill 0)
        boundary = _DASK_BOUNDARY_ALIASES.get(boundary, boundary)   # e.g. 'periodic' -> 'wrap'
        f = (lambda b, _f=func, _kw=kwargs: _f(b, **_kw)) if kwargs else func
        return o.map_overlap(self, f, depth, boundary=boundary, dtype=dtype, device=device)

    def __array__(self, dtype=None, copy=None):
        """Materialize to a real numpy array - what ``np.asarray(a)`` / ``np.array(a)`` call.

        Without this, NumPy falls back to treating a DynamicArray as an opaque object and
        produces a **0-d object array** rather than the data: ``np.asarray(a[10:20])`` came
        back with ``shape == ()`` and silently wrong results downstream, instead of raising.
        Materializing here is the same work as ``.compute()``, so it is deliberately EAGER
        and unbounded - the whole (possibly sliced) array is read into memory. Slice first,
        then convert, exactly as with a zarr array.

        ``dtype``/``copy`` follow the NumPy protocol: ``copy=False`` cannot be honoured,
        since the data does not exist until it is read, and NumPy 2 requires that to raise.
        """
        if copy is False:
            raise ValueError(
                "cannot return a view of a DynamicArray without copying: the data is "
                "produced lazily on read. Use np.asarray(a) or a.compute() instead.")
        result = self.compute()
        if dtype is not None:
            result = result.astype(dtype, copy=False)
        return result

    def __array_function__(self, func, types, args, kwargs):
        """NumPy high-level function protocol -> lazy ops, so ``np.stack``/``np.max``/
        ``np.where``/``np.concatenate``/... work on DynamicArrays exactly as on dask arrays.
        This is what lets pyrops call ``np.<func>`` uniformly across both backends.
        A function dyna_zarr does not implement returns ``NotImplemented`` (NumPy then raises)."""
        handler = _array_function_registry().get(func)
        if handler is None:
            return NotImplemented
        try:
            return handler(*args, **kwargs)
        except (TypeError, ValueError, NotImplementedError):
            return NotImplemented

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """NumPy ufunc protocol -> lazy ops, so ``np.sqrt(a)`` / ``np.add(a, 2)`` work on a
        DynamicArray exactly as on a dask array. Only the plain ``__call__`` form (no ``out=``,
        no reductions) is handled; anything else -- or a ufunc dyna_zarr doesn't implement --
        returns ``NotImplemented`` so NumPy can fall back / raise clearly."""
        from . import operations as o
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        name = _UFUNC_ALIASES.get(ufunc.__name__, ufunc.__name__)
        fn = getattr(o, name, None)
        if fn is None:
            return NotImplemented
        try:
            return fn(*inputs)
        except (TypeError, ValueError):
            return NotImplemented

    def _read_direct(self, key):
        """
        Internal method to read data directly without creating transforms.
        Used by Transform.read() methods to actually fetch data.
        """
        if self._transform is None:
            # Direct read from source
            if self._is_tensorstore:
                # TensorStore: call .read().result() for async operations
                return self._ts_array[key].read().result()
            else:
                # Zarr: direct indexing
                return self._zarr_array[key]
        else:
            # Apply transformation chain
            return self._transform.read(key)

    def _with_transform(self, transform):
        """
        Create a new DynamicArray with a transformation applied.
        """
        result = DynamicArray(self)
        result._transform = transform
        result._shape = _shape_of(transform)
        result._chunks = _checked_chunks(transform)
        result._dtype = transform.dtype
        # Keep zarr metadata from original
        return result

    @classmethod
    def _from_transform(cls, transform):
        """Build a SOURCELESS DynamicArray whose data is synthesized by ``transform`` (a
        generative source, e.g. the creation ops). There is no underlying zarr/tensorstore
        array; every read goes through ``transform.read(key)``."""
        self = cls.__new__(cls)
        self._source = None
        self._zarr_array = None
        self._ts_array = None
        self._is_tensorstore = False
        self._shape = _shape_of(transform)
        self._chunks = _checked_chunks(transform)
        self._dtype = np.dtype(transform.dtype)
        self._transform = transform
        self._zarr_format = None
        self._compressor = None
        self._compressors = None
        self._shards = None
        self._codecs = None
        return self
    
    # Reduction methods. LAZY, exactly like operations.<reducer> (and dask): each returns
    # a DynamicArray, so `x > x.mean()` stays one lazy chain. A small result (<= 1 MiB,
    # e.g. a 0-d statistic) is computed once and cached - see operations.reductions -
    # and converts on demand: float(x.mean()), int(x.max()), `if x.any():`.
    def min(self, axis=None, keepdims=False):
        """Minimum along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.min(self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """Maximum along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.max(self, axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims=False):
        """Sum along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        """Mean along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.mean(self, axis=axis, keepdims=keepdims)

    def prod(self, axis=None, keepdims=False):
        """Product along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.prod(self, axis=axis, keepdims=keepdims)

    def median(self, axis=None, keepdims=False):
        """Median along ``axis`` (None = all). Lazy; reads the reduced axes whole."""
        from . import operations
        return operations.median(self, axis=axis, keepdims=keepdims)

    def std(self, axis=None, keepdims=False, ddof=0):
        """Standard deviation along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.std(self, axis=axis, keepdims=keepdims, ddof=ddof)

    def var(self, axis=None, keepdims=False, ddof=0):
        """Variance along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.var(self, axis=axis, keepdims=keepdims, ddof=ddof)

    def any(self, axis=None, keepdims=False):
        """Whether any element is true along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        """Whether all elements are true along ``axis`` (None = all). Lazy."""
        from . import operations
        return operations.all(self, axis=axis, keepdims=keepdims)

    def argmin(self, axis=None, keepdims=False):
        """Index of the minimum along ``axis`` (None = flattened). Lazy."""
        from . import operations
        return operations.argmin(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis=None, keepdims=False):
        """Index of the maximum along ``axis`` (None = flattened). Lazy."""
        from . import operations
        return operations.argmax(self, axis=axis, keepdims=keepdims)

    # --- scalar conversion: compute on demand, NumPy's rules -----------------------
    def _one_element(self, what):
        size = int(np.prod(self._shape)) if self._shape else 1
        if size != 1:
            raise TypeError(
                f"only an array with exactly one element can be converted to {what}; "
                f"this one has shape {self._shape}")
        return np.asarray(self.compute()).reshape(())

    def __float__(self):
        return float(self._one_element("a Python float"))

    def __int__(self):
        return int(self._one_element("a Python int"))

    def __complex__(self):
        return complex(self._one_element("a Python complex"))

    def __index__(self):
        from .utils import parse_dtype
        if parse_dtype(self._dtype)[0].kind not in "biu":
            raise TypeError(
                f"only an integer array can be used as an index; this one is {self._dtype}")
        return int(self._one_element("an index"))

    def __bool__(self):
        # NumPy refuses to guess for more than one element, and so does this. It is
        # checked from the shape, so an ambiguous `if arr:` raises WITHOUT reading data.
        size = int(np.prod(self._shape)) if self._shape else 1
        if size != 1:
            raise ValueError(
                "the truth value of a DynamicArray with more than one element is "
                "ambiguous; use .any() or .all()")
        return bool(self._one_element("a bool"))

    def clear_cache(self):
        """Forget the cached small-reduction results in this array's chain, so the next
        read recomputes them (e.g. after the source data was rewritten in place).
        ``io.clear_cache()`` does the same for every array."""
        from .operations.reductions import clear_chain_cache
        clear_chain_cache(self)

    def histogram(self, bins=256, range=None):
        """Lazy streaming histogram over the whole array: ``(counts, bin_edges)`` like numpy,
        both lazy (see operations.histogram). Slice first for a per-channel/plane histogram:
        ``da[channel].histogram()``."""
        from . import operations
        return operations.histogram(self, bins=bins, range=range)


# Import Transform subclasses
from .operations import SliceTransform


# --------------------------------------------------------------------------- #
# Operator overloads -> lazy elementwise ops via operations.map_blocks.
# Mirrors numpy / ome_zarr_pyramid.Pyramid: arithmetic and comparisons are
# elementwise; &,|,^,~ are LOGICAL (for boolean masks), matching `~mask`. `a == b`
# returns a lazy mask, but assigning __eq__ *after* the class body leaves the
# inherited identity __hash__ intact, so a DynamicArray is still hashable.
# --------------------------------------------------------------------------- #

# dask map_overlap boundary names -> scipy.ndimage names used by operations.map_overlap.
_DASK_BOUNDARY_ALIASES = {"periodic": "wrap"}

_ARRAY_FUNCTION_REGISTRY = None


def _array_function_registry():
    """Lazily build & cache {numpy_function: handler} for __array_function__. Handlers adapt
    NumPy's call convention (extra out=/keepdims kwargs, tuple/None axis defaults) to the
    dyna_zarr.operations signatures. Not covered here (kept explicit / backend-branched by the
    caller): creation (nullary), np.pad with a mode dyna's pad lacks."""
    global _ARRAY_FUNCTION_REGISTRY
    if _ARRAY_FUNCTION_REGISTRY is not None:
        return _ARRAY_FUNCTION_REGISTRY
    import numpy as _np
    from . import operations as o

    def _red(fn):                       # reductions: tolerate out=/keepdims=_NoValue
        def h(a, axis=None, out=None, keepdims=_np._NoValue, **k):
            kd = False if keepdims is _np._NoValue else bool(keepdims)
            return fn(a, axis=axis, keepdims=kd)
        return h

    def _red_ddof(fn):                  # std/var
        def h(a, axis=None, out=None, keepdims=_np._NoValue, ddof=0, **k):
            kd = False if keepdims is _np._NoValue else bool(keepdims)
            return fn(a, axis=axis, keepdims=kd, ddof=ddof)
        return h

    def _histogram(a, bins=10, range=None, density=None, weights=None):
        # numpy's own default is bins=10. density/weights are not implemented: refused
        # (NotImplemented -> numpy raises TypeError) rather than silently ignored, which
        # would return plain counts where the caller asked for something else.
        if density or weights is not None:
            raise NotImplementedError("np.histogram(density=/weights=) on a DynamicArray")
        return o.histogram(a, bins=bins, range=range)

    def _unique(a, return_index=False, return_inverse=False, return_counts=False,
                axis=None, *, equal_nan=True, sorted=True):
        # values only (always sorted, so sorted=False is satisfied too); the rest is refused
        # rather than ignored, which would return a different thing than was asked for
        if return_index or return_inverse or return_counts or axis is not None or not equal_nan:
            raise NotImplementedError(
                "np.unique(return_index=/return_inverse=/return_counts=/axis=/"
                "equal_nan=False) on a DynamicArray")
        return o.unique(a)

    def _flip(a, axis=None):            # dyna flip is per-int-axis; chain for tuple / all-axes
        axes = range(a.ndim) if axis is None else ((axis,) if _np.isscalar(axis) else axis)
        out = a
        for ax in axes:
            out = o.flip(out, ax)
        return out

    def _roll(a, shift, axis=None):     # dyna roll is single-axis; chain for a per-axis list
        if axis is not None and not _np.isscalar(axis):
            shifts = shift if not _np.isscalar(shift) else [shift] * len(axis)
            out = a
            for s, ax in zip(shifts, axis):
                out = o.roll(out, int(s), int(ax))
            return out
        return o.roll(a, shift, axis)

    reg = {
        _np.stack: lambda arrays, axis=0, **k: o.stack(list(arrays), axis=axis),
        _np.concatenate: lambda arrays, axis=0, **k: o.concatenate(list(arrays), axis=axis),
        _np.where: lambda cond, x, y, **k: o.where(cond, x, y),
        _np.expand_dims: lambda a, axis, **k: o.expand_dims(a, axis),
        _np.squeeze: lambda a, axis=None, **k: o.squeeze(a, axis),
        _np.transpose: lambda a, axes=None, **k: o.transpose(
            a, tuple(reversed(range(a.ndim))) if axes is None else axes),
        _np.flip: _flip,
        _np.roll: _roll,
        _np.pad: lambda a, pad_width, mode="constant", **k: o.pad(a, pad_width, mode=mode, **k),
        _np.rot90: lambda a, k=1, axes=(0, 1), **kw: o.rot90(a, k, axes),
        _np.diff: lambda a, n=1, axis=-1, **k: o.diff(a, n, axis),
        _np.gradient: lambda a, *ar, axis=-1, **k: o.gradient(a, axis=axis),
        _np.digitize: lambda a, bins, right=False, **k: o.digitize(a, bins, right),
        _np.isin: lambda a, test, invert=False, **k: o.isin(a, test, invert),
        _np.histogram: _histogram,
        _np.cumsum: lambda a, axis=None, **k: o.cumsum(a, axis),
        _np.cumprod: lambda a, axis=None, **k: o.cumprod(a, axis),
        _np.round: lambda a, decimals=0, **k: o.round(a, decimals),
        _np.around: lambda a, decimals=0, **k: o.round(a, decimals),
        _np.clip: lambda a, a_min=None, a_max=None, **k: o.clip(a, a_min, a_max),
        _np.unique: _unique,
        # complex-part extraction (pointwise; dispatched as array-functions, not ufuncs)
        _np.real: lambda a, **k: o.map_blocks(_np.real, a, name="real"),
        _np.imag: lambda a, **k: o.map_blocks(_np.imag, a, name="imag"),
        _np.angle: lambda a, deg=False, **k: o.map_blocks(
            lambda x: _np.angle(x, deg=deg), a, name="angle"),
    }
    for npf, ofn in [(_np.amax, o.max), (_np.max, o.max), (_np.amin, o.min), (_np.min, o.min),
                     (_np.sum, o.sum), (_np.mean, o.mean), (_np.prod, o.prod),
                     (_np.any, o.any), (_np.all, o.all),
                     (_np.argmin, o.argmin), (_np.argmax, o.argmax), (_np.median, o.median)]:
        reg[npf] = _red(ofn)
    reg[_np.std] = _red_ddof(o.std)
    reg[_np.var] = _red_ddof(o.var)
    _ARRAY_FUNCTION_REGISTRY = reg
    return reg

# numpy ufunc names that differ from the operations spelling (see __array_ufunc__).
_UFUNC_ALIASES = {
    "absolute": "abs",
    "true_divide": "divide",
}


def _binop(opname, reflected=False):
    def method(self, other):
        from . import operations as o
        fn = getattr(o, opname)
        return fn(other, self) if reflected else fn(self, other)
    method.__name__ = ("__r" if reflected else "__") + opname + "__"
    return method


def _unop(opname):
    def method(self):
        from . import operations as o
        return getattr(o, opname)(self)
    method.__name__ = "__" + opname + "__"
    return method


for _dunder, _op in {
    "add": "add", "sub": "subtract", "mul": "multiply", "truediv": "divide",
    "floordiv": "floor_divide", "mod": "mod", "pow": "power",
    "and": "bitwise_and", "or": "bitwise_or", "xor": "bitwise_xor",
    "lshift": "left_shift", "rshift": "right_shift",
}.items():
    setattr(DynamicArray, f"__{_dunder}__", _binop(_op))
    setattr(DynamicArray, f"__r{_dunder}__", _binop(_op, reflected=True))

for _dunder, _op in {
    "lt": "less", "le": "less_equal", "gt": "greater", "ge": "greater_equal",
    "eq": "equal", "ne": "not_equal",
}.items():
    setattr(DynamicArray, f"__{_dunder}__", _binop(_op))

DynamicArray.__neg__ = _unop("negative")
DynamicArray.__abs__ = _unop("abs")
DynamicArray.__invert__ = _unop("invert")


def slice_array(array: DynamicArray, key) -> DynamicArray:
    """
    Create a lazy slice of an array.
    """
    transform = SliceTransform(array, key)
    return array._with_transform(transform)





# Global memory pool for region buffers
class RegionBufferPool:
    """
    Thread-safe memory pool for reusing region buffers in TensorStore I/O.
    
    Benefits:
    - Reduces allocation overhead (malloc/free are expensive)
    - Improves memory locality and cache performance
    - Reduces GC pressure from frequent large allocations
    - Particularly valuable for TensorStore's C++ backend
    """
    
    def __init__(self, max_size: int = 64):
        self.pool = {}  # Key: (shape, dtype) -> List[buffers]
        self.lock = threading.Lock()
        self.max_per_key = 8  # Max buffers per shape/dtype combo
        self.hits = 0
        self.misses = 0
        self.max_size = max_size
        self.total_buffers = 0
    
    def get_buffer(self, shape: Tuple[int, ...], dtype) -> Optional[np.ndarray]:
        """Get a buffer from pool if available, otherwise return None."""
        key = (shape, str(dtype))
        
        with self.lock:
            if key in self.pool and self.pool[key]:
                buf = self.pool[key].pop()
                self.hits += 1
                self.total_buffers -= 1
                return buf
        
        self.misses += 1
        return None
    
    def return_buffer(self, buf: np.ndarray):
        """Return buffer to pool for reuse."""
        if buf is None:
            return
            
        key = (buf.shape, str(buf.dtype))
        
        with self.lock:
            # Don't exceed max buffers per key
            if key not in self.pool:
                self.pool[key] = []
            
            if len(self.pool[key]) < self.max_per_key and self.total_buffers < self.max_size:
                # Zero out buffer for safety (optional, can be removed for speed)
                # buf.fill(0)  # Comment out for max performance
                self.pool[key].append(buf)
                self.total_buffers += 1
    
    def clear(self):
        """Clear the entire pool and reset stats."""
        with self.lock:
            self.pool.clear()
            self.hits = 0
            self.misses = 0
            self.total_buffers = 0
    
    def get_stats(self):
        """Get pool statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'total_buffers': self.total_buffers,
                'unique_shapes': len(self.pool)
            }


# Global buffer pool instance
_REGION_BUFFER_POOL = RegionBufferPool(max_size=64)





def from_array(source, chunks=None, *, lock=False) -> DynamicArray:
    """Wrap an array-like ``source`` as a lazy DynamicArray. Nothing is read until the
    array is.

    Supported sources:

    - ``numpy.ndarray``, ``zarr.Array``, a TensorStore array, a ``DynamicArray``;
    - a ``dask.array.Array``: each region's slice is computed when dyna reads it (with
      dask's synchronous scheduler - dyna's worker threads are the parallelism). Correct
      and memory-bounded, but work the graph shares between regions (an overlap filter,
      a rechunk) is redone per region; to WRITE a dask array, pushing it into
      ``io.create_sink`` is usually faster;
    - a ``micro_reader.Image``, or any object with ``shape``, ``dtype`` and
      ``__getitem__`` returning a NumPy array for a tuple of slices (e.g. a reader's own
      region source).

    Parameters
    ----------
    chunks : tuple of int, optional
        The grid to report (it drives region alignment and io.write's default output
        chunks). Default: the source's own grid - zarr / TensorStore chunks, dask's
        ``chunksize``, otherwise a ``chunks`` attribute that is one int per axis, then
        micro-reader's ``read_unit`` - else none.
    lock : bool or lock, optional
        Serialise every read, for a source that is not safe to read from several
        threads at once (``True``: a new lock; or pass a lock to share it between
        sources). Refused for numpy, zarr, TensorStore and dask, which are safe already.

    Only zarr and TensorStore sources carry a storage format and codec for io.write to
    keep; anything else is written with its defaults (zarr v3, blosc-lz4). A source must
    pickle to be used by worker processes (a ``micro_reader.Image`` holds open files and
    does not).
    """
    is_dask = _is_dask(source)
    safe = is_dask or isinstance(source, (DynamicArray, np.ndarray, zarr.Array)) or (
        hasattr(source, 'read') and hasattr(source, 'spec'))           # TensorStore
    if lock and safe:
        raise ValueError(
            f"lock= is for sources that are not thread-safe; a {type(source).__name__} "
            f"is safe to read from several threads already. Drop lock=.")

    if safe and not is_dask:
        # the constructor's own paths: they keep the storage settings (format, codecs,
        # shards) that zarr and TensorStore sources carry
        if isinstance(source, np.ndarray):
            return DynamicArray(source, chunks=chunks)
        arr = DynamicArray(source)
        if chunks is not None:
            arr._chunks = _clamped(_require_grid(chunks, arr.shape), arr.shape)
        return arr

    missing = [a for a in ('shape', 'dtype', '__getitem__') if not hasattr(source, a)]
    if missing:
        raise TypeError(
            f"{type(source).__module__}.{type(source).__name__} is not array-like: it has "
            f"no {', '.join(missing)}. A source needs shape, dtype and __getitem__ "
            f"returning a NumPy array.")
    shape = tuple(int(s) for s in source.shape)
    if chunks is not None:
        grid = _require_grid(chunks, shape)
    elif is_dask:
        grid = _per_axis_grid(source.chunksize, len(shape))
    else:
        # advisory grids: a malformed one is ignored, not trusted
        grid = (_per_axis_grid(getattr(source, 'chunks', None), len(shape))
                or _per_axis_grid(getattr(source, 'read_unit', None), len(shape)))
    if lock is True:
        lock = threading.Lock()
    arr = DynamicArray(_SourceAdapter(source, lock or None))
    arr._chunks = _clamped(grid, shape) if grid is not None else None
    return arr


def _require_grid(chunks, shape):
    grid = _per_axis_grid(chunks, len(tuple(shape)))
    if grid is None:
        raise ValueError(f"chunks={chunks!r} must be one positive int per axis of "
                         f"shape {tuple(shape)}")
    return grid


def _clamped(grid, shape):
    return tuple(min(int(c), int(s)) for c, s in zip(grid, shape))
