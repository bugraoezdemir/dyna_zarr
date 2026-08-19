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


class DynamicArray:
    """
    Wrapper around Zarr arrays or TensorStore arrays that enables lazy operations.
    
    Supports multiple input types:
    - zarr.Array: Wraps for lazy operations
    - TensorStore arrays: Wraps for lazy operations
    - DynamicArray: Copy constructor
    
    To read from files, use operations.read(path) instead.
    """

    def __init__(self, source: Union[zarr.Array, 'DynamicArray']):
        if isinstance(source, DynamicArray):
            # Copy constructor
            self._source = source._source
            self._zarr_array = source._zarr_array
            self._ts_array = source._ts_array
            self._is_tensorstore = source._is_tensorstore
            self._shape = source._shape
            self._chunks = source._chunks
            self._dtype = source._dtype
            self._transform = source._transform
            self._zarr_format = source._zarr_format
            self._compressor = source._compressor
            self._compressors = source._compressors
            self._shards = source._shards
            self._codecs = source._codecs
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
                self._extract_zarr_metadata(source)
            
            self._source = source
            self._shape = tuple(source.shape)
            self._chunks = getattr(source, 'chunks', None)
            self._dtype = source.dtype
            self._transform = None
            if not self._is_tensorstore:
                self._extract_zarr_metadata(source)
            else:
                self._zarr_format = None
                self._compressor = None
                self._compressors = None
                self._shards = None
                self._codecs = None

    def _extract_zarr_metadata(self, zarr_array):
        """Extract metadata from Zarr array, handling both v2 and v3."""
        # Detect Zarr format version
        if hasattr(zarr_array, '_version'):
            self._zarr_format = zarr_array._version
        elif hasattr(zarr_array, 'store'):
            # Try to detect from store structure
            store = zarr_array.store
            if hasattr(store, 'path'):
                store_path = Path(store.path) if isinstance(store.path, str) else store.path
                if (store_path / 'zarr.json').exists():
                    self._zarr_format = 3
                elif (store_path / '.zarray').exists():
                    self._zarr_format = 2
                else:
                    self._zarr_format = 2  # Default to v2
            else:
                self._zarr_format = 2  # Default to v2
        else:
            self._zarr_format = 2  # Default to v2

        # Extract compressor/compressors based on format
        if self._zarr_format == 3:
            # Zarr v3
            self._compressor = None
            if hasattr(zarr_array, 'metadata'):
                metadata = zarr_array.metadata
                if 'codecs' in metadata:
                    self._compressors = metadata['codecs']
                else:
                    self._compressors = None
            elif hasattr(zarr_array, 'compressors'):
                self._compressors = zarr_array.compressors
            else:
                self._compressors = None

            # Extract shards
            if hasattr(zarr_array, 'metadata') and 'shards' in zarr_array.metadata:
                self._shards = zarr_array.metadata['shards']
            elif hasattr(zarr_array, 'shards'):
                self._shards = zarr_array.shards
            else:
                self._shards = None
            
            # Convert v3 codecs to Codecs instance
            self._codecs = self._extract_codecs_from_v3(self._compressors)
        else:
            # Zarr v2
            try:
                self._compressor = zarr_array.compressor if hasattr(zarr_array, 'compressor') else None
            except (TypeError, AttributeError):
                # Handle Zarr v3 arrays that error on compressor access
                self._compressor = None
            self._compressors = None
            self._shards = None
            
            # Convert v2 compressor to Codecs instance
            if self._compressor is not None:
                try:
                    self._codecs = Codecs.from_numcodecs(self._compressor)
                except Exception:
                    # If conversion fails, use default
                    self._codecs = None
            else:
                self._codecs = None

    def _extract_codecs_from_v3(self, codecs_list):
        """Extract Codecs instance from Zarr v3 codec pipeline."""
        if codecs_list is None:
            return None
        
        # Look for compression codec in the pipeline
        for codec in codecs_list:
            codec_name = codec.get('name', '')
            config = codec.get('configuration', {})
            
            if codec_name == 'blosc':
                return Codecs(
                    compressor='blosc',
                    clevel=config.get('clevel', 5),
                    cname=config.get('cname', 'lz4'),
                    shuffle=config.get('shuffle', 1)
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
        """Get the chunk shape of the underlying array."""
        if self._chunks is not None:
            return self._chunks
        # Fallback: try to get chunks from underlying zarr array
        if self._zarr_array is not None and hasattr(self._zarr_array, 'chunks'):
            return self._zarr_array.chunks
        # If still None, return None (TensorStore or unknown)
        return None

    @property
    def dtype(self):
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

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
        with device_context(device):
            if self._transform is None:
                # No transform - read entire array
                if self._is_tensorstore:
                    result = self._ts_array[:].read().result()
                else:
                    result = self._zarr_array[:]
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

    def persist(self, **kwargs):
        """No-op (accepts dask's signature). dask ``persist`` materializes and caches an
        intermediate; the pull model has no graph to cache, so this returns the array
        unchanged. Use ``io.write`` to stage an intermediate to disk when needed."""
        return self

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
        result._shape = transform.shape
        result._chunks = transform.chunks
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
        self._shape = tuple(transform.shape)
        self._chunks = transform.chunks
        self._dtype = np.dtype(transform.dtype)
        self._transform = transform
        self._zarr_format = None
        self._compressor = None
        self._compressors = None
        self._shards = None
        self._codecs = None
        return self
    
    # Eager reduction shortcuts. Each streams via operations.<reducer> (memory-bound)
    # and computes immediately, returning a NumPy scalar/array. Use operations.<reducer>
    # directly (e.g. operations.max(a, 0)) for a lazy, chainable/writable reduction.
    def min(self, axis=None, keepdims=False):
        """Minimum along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.min(self, axis=axis, keepdims=keepdims).compute()

    def max(self, axis=None, keepdims=False):
        """Maximum along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.max(self, axis=axis, keepdims=keepdims).compute()

    def sum(self, axis=None, keepdims=False):
        """Sum along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.sum(self, axis=axis, keepdims=keepdims).compute()

    def mean(self, axis=None, keepdims=False):
        """Mean along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.mean(self, axis=axis, keepdims=keepdims).compute()

    def prod(self, axis=None, keepdims=False):
        """Product along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.prod(self, axis=axis, keepdims=keepdims).compute()

    def median(self, axis=None, keepdims=False):
        """Median along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.median(self, axis=axis, keepdims=keepdims).compute()

    def std(self, axis=None, keepdims=False, ddof=0):
        """Standard deviation along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.std(self, axis=axis, keepdims=keepdims, ddof=ddof).compute()

    def var(self, axis=None, keepdims=False, ddof=0):
        """Variance along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.var(self, axis=axis, keepdims=keepdims, ddof=ddof).compute()

    def any(self, axis=None, keepdims=False):
        """Whether any element is true along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.any(self, axis=axis, keepdims=keepdims).compute()

    def all(self, axis=None, keepdims=False):
        """Whether all elements are true along ``axis`` (None = all). Computed immediately."""
        from . import operations
        return operations.all(self, axis=axis, keepdims=keepdims).compute()

    def argmin(self, axis=None):
        """Index of the minimum along ``axis`` (None = flattened). Computed immediately."""
        from . import operations
        return operations.argmin(self, axis=axis).compute()

    def argmax(self, axis=None):
        """Index of the maximum along ``axis`` (None = flattened). Computed immediately."""
        from . import operations
        return operations.argmax(self, axis=axis).compute()

    def histogram(self, bins=256, range=None):
        """Streaming histogram over the whole array. Returns ``(counts, bin_edges)`` like
        numpy. Slice first for a per-channel/plane histogram: ``da[channel].histogram()``."""
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
        _np.histogram: lambda a, bins=256, range=None, **k: o.histogram(a, bins=bins, range=range),
        _np.cumsum: lambda a, axis=None, **k: o.cumsum(a, axis),
        _np.cumprod: lambda a, axis=None, **k: o.cumprod(a, axis),
        _np.round: lambda a, decimals=0, **k: o.round(a, decimals),
        _np.around: lambda a, decimals=0, **k: o.round(a, decimals),
        _np.clip: lambda a, a_min=None, a_max=None, **k: o.clip(a, a_min, a_max),
        _np.unique: lambda a, **k: o.unique(a),
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



