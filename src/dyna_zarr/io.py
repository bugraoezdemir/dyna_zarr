"""
I/O utilities for reading and writing array formats.

Provides unified interfaces for:
- Reading TIFF files, Zarr arrays, and other formats with lazy loading
- Writing DynamicArray to Zarr with parallel I/O
- Supports both local and remote storage (gs://, s3://, http://, https://)
"""

import tensorstore as ts
import zarr
import numpy as np
from pathlib import Path
from urllib.parse import urlparse
from typing import Union, Tuple, Optional, Any, List
import itertools
import time
import threading
import gc
import shutil
import warnings
from queue import Queue

from .tiff_reader import open_tiff_zarr, read_tiff_lazy
from .codecs import Codecs
from .dynamic_array import DynamicArray
from .utils import parse_dtype
from .operations._backend import (
    device_context as _device_context, asnumpy as _asnumpy,
    asnumpy_pinned as _asnumpy_pinned,
    new_stream as _new_stream, use_stream as _use_stream,
)


#: Smallest workable pipeline: 1 reader + 1 queued region + 1 in-flight write. `max_workers`
#: (the total live-region budget) is never taken below this, even by `memory_budget_mb` -
#: a budget that cannot fit 3 regions warns instead of deadlocking on a zero-capacity stage.
_MIN_LIVE_REGIONS = 3

#: Writer THREADS (not the in-flight write cap). `.write()` is async - it returns a future
#: and TensorStore's C++ pool performs the I/O - so extra writer threads only submit futures
#: faster. Measured under CPU-heavy zstd: 1/2/4/8 writers all land at 3.44-3.58s while RSS
#: climbs 998->1400 MiB. Two gives a little submission headroom without paying for more.
_WRITER_THREADS = 2


def _parse_storage_location(file_path):
    """
    Parse storage location to determine if it's local or remote.
    
    Returns:
        tuple: (storage_type, parsed_path)
            storage_type: 'local', 'gcs', 's3', 'http', 'https', or 'unknown'
            parsed_path: Path object for local, string for remote
    """
    if not isinstance(file_path, str):
        file_path = str(file_path)

    # Windows drive-letter path (e.g. C:\foo or C:/foo): urlparse would mistake
    # the drive letter for a single-character URL scheme, so short-circuit to local.
    if len(file_path) >= 2 and file_path[0].isalpha() and file_path[1] == ':':
        return 'local', Path(file_path)

    # Parse URL scheme
    parsed = urlparse(file_path)
    
    if parsed.scheme in ('', 'file'):
        # Local file system
        if parsed.scheme == 'file':
            # Remove file:// prefix
            local_path = parsed.path
        else:
            local_path = file_path
        return 'local', Path(local_path)
    elif parsed.scheme == 'gs':
        return 'gcs', file_path
    elif parsed.scheme == 's3':
        return 's3', file_path
    elif parsed.scheme in ('http', 'https'):
        return parsed.scheme, file_path
    else:
        return 'unknown', file_path


def _is_tiff_path(path_str):
    """Check if path string ends with TIFF extension."""
    lower = path_str.lower()
    return lower.endswith('.tif') or lower.endswith('.tiff')


def _detect_zarr_format_local(dir_path):
    """Detect Zarr format for local directories."""
    is_zarr_v2_array = (dir_path / ".zarray").exists()
    is_zarr_v2_group = (dir_path / ".zgroup").exists()
    is_zarr_v3 = (dir_path / "zarr.json").exists()
    
    if is_zarr_v2_array or is_zarr_v2_group:
        return 'zarr_v2'
    elif is_zarr_v3:
        return 'zarr_v3'
    else:
        return None


def read_file(file_path):
    """
    Read a file (TIFF or Zarr) with a unified interface.
    
    Automatically detects the file type and storage location:
    - TIFF files: Uses tifffile's zarr bridge for thread-safe reads
    - Zarr v2/v3: Uses TensorStore's native zarr drivers for parallel I/O
    
    Supports both local and remote storage:
    - Local: /path/to/file.tif, /path/to/array.zarr
    - GCS: gs://bucket/path/to/file
    - S3: s3://bucket/path/to/file
    - HTTP: http://example.com/path/to/file
    
    Returns an object with TensorStore-compatible API (.read().result(), etc.)
    
    Args:
        file_path: Path or URL to a TIFF file or Zarr array
        
    Returns:
        A lazy reader with parallel read support
        
    Raises:
        ValueError: If file type is unsupported or cannot be detected
    """
    storage_type, parsed_path = _parse_storage_location(file_path)
    
    if storage_type == 'local':
        # Local file system - can check file type directly
        if parsed_path.is_dir():
            # Try to detect Zarr format
            zarr_format = _detect_zarr_format_local(parsed_path)
            
            if zarr_format in ('zarr_v2', 'zarr_v3'):
                # Check if this is a group or array (applies to both v2 and v3)
                zarr_obj = zarr.open(str(parsed_path), mode='r')
                
                if isinstance(zarr_obj, zarr.Group):
                    # Find first array in group
                    array_path = None
                    for key in zarr_obj:
                        item = zarr_obj[key]
                        if isinstance(item, zarr.Array):
                            array_path = parsed_path / key
                            break
                    
                    if not array_path:
                        raise ValueError(f"No arrays found in zarr group {parsed_path}")
                    
                    parsed_path = array_path
                
                # Open with appropriate driver
                driver = "zarr3" if zarr_format == 'zarr_v3' else "zarr"
                spec = {
                    "driver": driver,
                    "kvstore": {
                        "driver": "file",
                        "path": str(parsed_path.resolve())
                    }
                }
                return ts.open(spec, read=True).result()
            
            else:
                raise ValueError(
                    f"Directory {parsed_path} does not appear to be a zarr store "
                    f"(no .zarray, .zgroup, or zarr.json found)"
                )
        
        elif _is_tiff_path(str(parsed_path)):
            # Local TIFF file -> lazy zarr array (raw backend object; read_array wraps it)
            return open_tiff_zarr(parsed_path)

        else:
            raise ValueError(
                f"Unsupported file type: {parsed_path.suffix}. "
                f"Expected .tif, .tiff, or a Zarr directory"
            )
    
    elif storage_type in ('gcs', 's3'):
        # Remote cloud storage - use TensorStore
        # Try to determine if it's a TIFF or Zarr based on path
        path_str = str(parsed_path)
        
        if _is_tiff_path(path_str):
            # TIFF file on cloud storage (tifffile may support remote via fsspec)
            return open_tiff_zarr(file_path)
        
        else:
            # Assume it's a Zarr array on cloud storage
            # TensorStore will auto-detect the format
            if storage_type == 'gcs':
                # Extract bucket and path from gs://bucket/path
                parts = path_str.replace('gs://', '').split('/', 1)
                bucket = parts[0]
                object_path = parts[1] if len(parts) > 1 else ''
                
                # Try Zarr v3 first, fall back to v2
                try:
                    spec = {
                        "driver": "zarr3",
                        "kvstore": {
                            "driver": "gcs",
                            "bucket": bucket,
                            "path": object_path
                        }
                    }
                    return ts.open(spec, read=True).result()
                except:
                    # Try Zarr v2
                    spec = {
                        "driver": "zarr",
                        "kvstore": {
                            "driver": "gcs",
                            "bucket": bucket,
                            "path": object_path
                        }
                    }
                    return ts.open(spec, read=True).result()
            
            elif storage_type == 's3':
                # Extract bucket and path from s3://bucket/path
                parts = path_str.replace('s3://', '').split('/', 1)
                bucket = parts[0]
                object_path = parts[1] if len(parts) > 1 else ''
                
                # Try Zarr v3 first, fall back to v2
                try:
                    spec = {
                        "driver": "zarr3",
                        "kvstore": {
                            "driver": "s3",
                            "bucket": bucket,
                            "path": object_path
                        }
                    }
                    return ts.open(spec, read=True).result()
                except:
                    # Try Zarr v2
                    spec = {
                        "driver": "zarr",
                        "kvstore": {
                            "driver": "s3",
                            "bucket": bucket,
                            "path": object_path
                        }
                    }
                    return ts.open(spec, read=True).result()
    
    elif storage_type in ('http', 'https'):
        # HTTP/HTTPS URL
        path_str = str(parsed_path)
        
        if _is_tiff_path(path_str):
            # TIFF over HTTP - pass to tifffile (may support via fsspec)
            return open_tiff_zarr(file_path)
        else:
            # Zarr over HTTP
            spec = {
                "driver": "zarr",  # Try v2 first
                "kvstore": {
                    "driver": "http",
                    "base_url": file_path
                }
            }
            return ts.open(spec, read=True).result()
    
    else:
        raise ValueError(
            f"Unsupported storage type or file format: {file_path}. "
            f"Expected local path, gs://, s3://, http://, or https:// URL"
        )


# I/O operations for reading and writing arrays


def read_array(source: Union[str, Path]) -> 'DynamicArray':
    """
    Read array from file path (TIFF or Zarr).
    
    Supports both local and remote storage:
    - Local: /path/to/file.tif, /path/to/array.zarr
    - GCS: gs://bucket/path/to/file
    - S3: s3://bucket/path/to/file
    - HTTP: http://example.com/path/to/file
    
    Args:
        source: Path or URL to a TIFF file or Zarr array
        
    Returns:
        DynamicArray wrapping the opened array
    """
    from dyna_zarr.io import read_file
    
    source_path = Path(source) if not isinstance(source, Path) else source
    
    # TIFF -> lazy zarr array via tifffile's bridge, wrapped as a normal zarr-backed
    # DynamicArray (laziness/slicing/memory-bounded reads all come from DynamicArray).
    if str(source_path).lower().endswith(('.tif', '.tiff')):
        return read_tiff_lazy(source)
    
    elif (isinstance(source_path, Path) and source_path.is_dir() and 
          ((source_path / ".zarray").exists() or 
           (source_path / ".zgroup").exists() or 
           (source_path / "zarr.json").exists())):
        # Zarr directory - use read_file for TensorStore performance
        try:
            ts_array = read_file(source)
            dyn_array = object.__new__(DynamicArray)
            dyn_array._ts_array = ts_array
            dyn_array._is_tensorstore = True
            dyn_array._source = source
            dyn_array._shape = tuple(ts_array.shape)
            dyn_array._dtype = ts_array.dtype
            dyn_array._transform = None
            # Also open with zarr to get metadata like chunks
            dyn_array._zarr_array = zarr.open(source, mode='r')
            dyn_array._chunks = dyn_array._zarr_array.chunks
            dyn_array._extract_zarr_metadata(dyn_array._zarr_array)
            return dyn_array
        except Exception:
            # Fall back to zarr if read_file fails
            pass
    
    # Remote URL or fallback to regular zarr opening
    if '://' in str(source):  # Likely a remote URL
        ts_array = read_file(source)
        dyn_array = object.__new__(DynamicArray)
        dyn_array._ts_array = ts_array
        dyn_array._is_tensorstore = True
        dyn_array._source = source
        dyn_array._shape = tuple(ts_array.shape)
        dyn_array._dtype = ts_array.dtype
        dyn_array._transform = None
        dyn_array._chunks = None
        dyn_array._zarr_array = None
        dyn_array._zarr_format = None
        dyn_array._compressor = None
        dyn_array._compressors = None
        dyn_array._shards = None
        dyn_array._codecs = None
        return dyn_array
    else:
        # Local zarr fallback
        zarr_array = zarr.open(source, mode='r')
        dyn_array = object.__new__(DynamicArray)
        dyn_array._is_tensorstore = False
        dyn_array._ts_array = None
        dyn_array._zarr_array = zarr_array
        dyn_array._source = source
        dyn_array._shape = zarr_array.shape
        dyn_array._chunks = zarr_array.chunks
        dyn_array._dtype = zarr_array.dtype
        dyn_array._transform = None
        dyn_array._extract_zarr_metadata(zarr_array)
        return dyn_array

def _compute_region_shape(input_shape, final_chunks, region_size_mb, dtype=None, input_chunks=None):
    """
    Compute optimal region shape with simple deterministic algorithm.
    
    Algorithm:
    1. Start with a single output chunk
    2. Identify dimensions needing expansion (region < input_shape)
    3. Expand dimensions in reverse order (last → first) until region_size_mb reached
    4. For each dimension: expand until covered OR budget exhausted
    
    Example:
      Input: (50, 179, 2, 339, 415), Input chunks: (1, 1, 1, 339, 415)
      Output chunks: (1, 64, 1, 64, 64)
      Start: (1, 64, 1, 64, 64)
      Expand dim 4: (1, 64, 1, 64, 415) - complete dimension 4
      Expand dim 3: (1, 64, 1, 339, 415) - complete dimension 3
      Expand dim 2: (1, 64, 2, 339, 415) - complete dimension 2
      Expand dim 1: (1, 128, 2, 339, 415) - stop when budget reached
    """
    if dtype is None:
        element_size = 2
    else:
        try:
            element_size = int(np.dtype(dtype).itemsize)
        except Exception:
            element_size = 2

    target_bytes = region_size_mb * 1024 * 1024
    
    if input_chunks is None:
        input_chunks = tuple(final_chunks)
    
    if len(input_chunks) != len(final_chunks):
        input_chunks = tuple(final_chunks)
    
    input_arr = np.array(input_shape, dtype=np.int64)
    input_chunk_arr = np.array(input_chunks, dtype=np.int64)
    output_chunk_arr = np.array(final_chunks, dtype=np.int64)
    
    # STEP 1: Start with a single output chunk
    region_arr = output_chunk_arr.copy()
    region_arr = np.minimum(region_arr, input_arr)  # Clamp to array size
    
    current_bytes = np.prod(region_arr) * element_size
    
    # If single output chunk exceeds target, use it anyway (can't split chunks)
    if current_bytes >= target_bytes:
        return tuple(region_arr.tolist())
    
    # STEP 2: Compute expansion increments using LCM (maintains both alignments)
    expansion_increments = np.zeros(len(region_arr), dtype=np.int64)
    for i in range(len(region_arr)):
        gcd = np.gcd(input_chunk_arr[i], output_chunk_arr[i])
        lcm = (input_chunk_arr[i] * output_chunk_arr[i]) // gcd
        expansion_increments[i] = lcm
    
    # STEP 3: Expand dimensions in reverse order (last → first)
    for dim in reversed(range(len(region_arr))):
        # Expand this dimension until complete or budget exhausted
        while region_arr[dim] < input_arr[dim]:
            increment = expansion_increments[dim]
            remaining = input_arr[dim] - region_arr[dim]
            
            # Determine new size
            if remaining <= increment:
                # Remainder fits in one increment - complete the dimension
                new_size = input_arr[dim]
            else:
                # Add one increment
                new_size = region_arr[dim] + increment
                
                # Check if we should include the remainder now
                # to avoid creating a small partial region later
                future_remaining = input_arr[dim] - new_size
                if 0 < future_remaining < increment:
                    # Next time would be a small remainder - include it now
                    new_size = input_arr[dim]
            
            # Test if this fits in budget
            test_region = region_arr.copy()
            test_region[dim] = new_size
            new_bytes = np.prod(test_region) * element_size
            
            if new_bytes <= target_bytes:
                # Fits in budget - accept it
                region_arr[dim] = new_size
                current_bytes = new_bytes
            else:
                # Doesn't fit - stop expanding this dimension
                break
        
        # After completing this dimension, check if we should continue
        # to the next dimension or stop
        if current_bytes >= target_bytes:
            break
    
    # STEP 4: Verify output chunk alignment for PARTIAL dimensions only
    # Full dimensions don't need alignment (they include all chunks anyway)
    for i in range(len(region_arr)):
        # Skip if dimension is fully enclosed
        if region_arr[i] >= input_arr[i]:
            continue
        
        # For partial dimensions, ensure output chunk alignment
        if output_chunk_arr[i] > 0 and region_arr[i] % output_chunk_arr[i] != 0:
            # Round down to output chunk boundary to avoid cutting inside chunks
            aligned_size = (region_arr[i] // output_chunk_arr[i]) * output_chunk_arr[i]
            # Ensure at least one output chunk
            region_arr[i] = max(output_chunk_arr[i], aligned_size)
    
    return tuple(region_arr.tolist())

def _aligned_region_shape(array, final_chunks, input_shape):
    """The op chain's read-alignment grid, if it has one and it can be used as-is.

    A ``map_overlap(..., align=cell)`` transform expands every read to whole ``cell``-sized
    cells, so it can only produce output a whole cell at a time. Writing it in regions
    smaller than a cell therefore recomputes the cell once per region - 8x the work for
    half-sized regions in 3-D, 64x for quarter-sized. Returning the alignment here makes the
    default region match the producer, so each cell is computed exactly once.

    Returns ``None`` when there is no alignment, or when it cannot be honoured: the region
    must be a whole multiple of the output chunks per axis (otherwise a region would write
    partial chunks and the parallel writes would race).
    """
    transform = getattr(array, "_transform", None)
    align = getattr(transform, "align", None)
    if align is None:
        return None
    try:
        align = tuple(int(a) for a in align)
    except TypeError:
        return None
    if len(align) != len(input_shape):
        return None
    if any(a <= 0 for a in align):
        return None
    if any(a % c for a, c in zip(align, final_chunks)):
        return None      # would write partial chunks -> unsafe for parallel region writes
    return tuple(min(a, s) for a, s in zip(align, input_shape))


def write_array(
    array: 'DynamicArray',
    output_path: str,
    max_workers: int = 4,
    num_readers: Optional[int] = None,
    chunks: Optional[Tuple[int, ...]] = None,
    shard_coefficients: Optional[Tuple[int, ...]] = None,
    dtype: Optional[Any] = None,
    compressor: Optional[Union[Codecs, Any]] = None,
    zarr_format: Optional[int] = None,
    region_size_mb: float = 8.0,
    gc_interval: float = 15.0,
    early_quarter_timeout: Optional[float] = None,
    early_tenth_timeout: Optional[float] = None,
    device: Optional[str] = None,
    region_shape: Optional[Tuple[int, ...]] = None,
    memory_budget_mb: Optional[float] = None,
    overwrite: bool = False,
):
    """
    ASYNC VECTORIZED TensorStore write with queue-based pipeline:
    - Queue buffers work between readers and writers (good for pipeline)
    - Readers: Fast, simple reads into queue
    - Writers: Async writes via TensorStore .write() (non-blocking)
    - TensorStore handles actual write parallelism in C++ backend

    Key insight: Queue is good for buffering. Async writes prevent blocking.
    Writers call .write() which returns futures immediately, then TensorStore
    does the actual parallel I/O internally.

    MEMORY MODEL
    ------------
    The pipeline has three stages that each pin whole region buffers: readers in
    flight, the hand-off queue, and submitted-but-uncommitted TensorStore writes.
    `max_workers` is the TOTAL number of regions allowed live across all three, so

        peak RAM ~= max_workers * region_bytes

    is computable from two numbers the caller already controls (`max_workers` and
    `region_shape`/`region_size_mb`). The internal split between the three stages is
    derived from `max_workers` and `num_readers`; the queue depth and the in-flight
    write cap are NOT separately settable (they were, and being independent multiples
    of the worker count they made peak memory scale with worker count while ignoring
    region SIZE entirely - 32 live regions at the defaults, which is ~256MB at an 8MB
    region but ~36GB at a 672^3 int32 one).

    Writer threads are deliberately NOT scaled with `max_workers`: `.write()` is async
    (it returns a future and TensorStore's C++ pool does the real I/O), so writer count
    is measurably irrelevant to throughput (1 vs 8 writers: same wall time) while each
    extra writer costs a live buffer. Readers are the real driver and stay tunable via
    `num_readers`.

    Parameters:
    -----------
    array : DynamicArray
        Input array to write
    output_path : str
        Path to output Zarr array
    max_workers : int
        TOTAL regions that may be live at once, across readers + queue + in-flight
        writes. Peak RAM ~= max_workers * region_bytes.
    num_readers : int, optional
        How much of `max_workers` goes to reader threads (the reader/writer balance).
        Default: about half of `max_workers`, at least 1. More readers hide read
        latency on slow/remote stores; they also raise peak memory one region each.
    memory_budget_mb : float, optional
        Ceiling on pipeline memory. When given, `max_workers` is clamped to
        `memory_budget_mb / region_bytes` (never raised above the requested value - the
        budget is a ceiling, not a target). If even a minimal pipeline cannot fit, a
        warning naming the culprit is emitted and the pipeline runs at its floor.
    chunks : tuple of int, optional
        Chunk shape. For v3 with sharding, this is the inner chunk shape.
        Default: (256, 256, ...) for all dimensions
    shard_coefficients : tuple of int, optional
        Multipliers for chunk sizes to compute shard shape (Zarr v3 only).
        Example: chunks=(64, 64), shard_coefficients=(4, 4) → shards=(256, 256)
        If None, infers from input array. If input has no shards, writes without sharding.
    compressor : Codecs, numcodecs compressor, or None
        Compression configuration. Can be:
        - Codecs object: Unified compression config for v2/v3
        - numcodecs compressor: For backward compatibility (converted to Codecs)
        - None: Infers from input array, or uses Blosc with LZ4 as default
    zarr_format : int
        Zarr format version (2 or 3, default: 3)
    region_size_mb : float
        Target size of read regions in MB (default: 8.0). Ignored if `region_shape` is given.
    region_shape : tuple of int, optional
        Explicit read-region shape, OVERRIDING `region_size_mb`. Must be a per-axis multiple
        of `chunks` (clamped to the array shape). Use it to align the read regions to a
        producer's processing grid - e.g. pass a tilewise-ccl `tile_shape` so its position-aware
        Phase-B `map_overlap` reads each whole tile exactly once (no re-read / re-label
        amplification) while still storing small `chunks`.
    overwrite : bool
        Replace an existing array at `output_path`. Default False (today's behaviour:
        write into/over whatever is there). Only a path that already looks like a zarr
        store is removed, so a mistyped `output_path` cannot delete an unrelated tree.
    gc_interval : float
        Seconds between GC runs (default: 15.0)
    """
    # A reshape/flatten as the OUTERMOST op is a C-order flat re-index that conflicts with
    # nd chunk layout -- the region writer can't do it memory-bound. Route it to the
    # streaming reindex writer (per-output-chunk, one input chunk at a time): hard-bounded
    # to ~1 input chunk + 1 output chunk, race-free, on any chunking.
    _tr = getattr(array, "_transform", None)
    _trname = type(_tr).__name__
    _local = _parse_storage_location(str(output_path))[0] == "local"
    if _trname == "FlattenTransform" and _local:
        # disk-staged rechunk-to-flat-contiguous + relabel: read-once source. Per-worker memory
        # = region_size_mb; peak ~= max_workers * region_size_mb (same knobs as the region path).
        from .rechunk import flatten_write
        return flatten_write(_tr.array, output_path, output_chunks=chunks,
                             max_mem=int(region_size_mb * 1024 * 1024), max_workers=max_workers,
                             dtype=dtype, zarr_format=zarr_format or 2)
    if _trname == "ReshapeTransform" and _local:
        # staged: flatten source to 1D contiguous F (read-once, bounded), then reindex F ->
        # target shape (contiguous input => no chunk-amplification). Per-worker memory =
        # region_size_mb; forwards max_workers. Same knobs as the region path.
        from .rechunk import reshape_write
        return reshape_write(_tr.array, array.shape, output_path, output_chunks=chunks,
                             max_mem=int(region_size_mb * 1024 * 1024), max_workers=max_workers,
                             dtype=dtype, zarr_format=zarr_format or 2)
    if _trname == "ScanTransform" and _local:
        # bounded-carry streaming scan (tile cross-section, walk the scan axis in strips with a
        # running carry): fully memory-bounded, read-once/write-once. Per-call budget =
        # region_size_mb (the pull read(key) path stays prefix-bounded for lazy sub-slices).
        from .operations.scan import scan_write
        return scan_write(_tr.array, _tr.op, _tr.axis, output_path, output_chunks=chunks,
                          max_mem=int(region_size_mb * 1024 * 1024),
                          dtype=dtype, zarr_format=zarr_format or 2)
    import tensorstore as ts

    # Aggressive memory cleanup before starting
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()
    
    # Handle metadata
    input_shape_temp = array.shape
    input_dtype_temp = array.dtype
    final_chunks = chunks if chunks is not None else tuple([256] * len(input_shape_temp))
    final_dtype = dtype if dtype is not None else array.dtype
    
    # Parse dtype for both Zarr v2 and v3 formats
    final_dtype_obj, final_dtype_v2_str, final_dtype_v3_name = parse_dtype(final_dtype)
    
    final_format = zarr_format if zarr_format is not None else 3
    
    # Handle compressor - convert to Codecs if needed
    if compressor is None:
        # Try to infer from input array if it's zarr
        if hasattr(array, 'codecs') and array.codecs is not None:
            final_codecs = array.codecs
        else:
            # Default: Blosc with LZ4
            final_codecs = Codecs('blosc', clevel=5, cname='lz4')
    elif isinstance(compressor, Codecs):
        final_codecs = compressor
    else:
        # Assume it's a numcodecs compressor - convert it
        final_codecs = Codecs.from_numcodecs(compressor)
    
    # Handle shards - compute from shard_coefficients or infer from input array
    final_shards = None
    if shard_coefficients is not None and final_format == 3:
        # Compute shards from coefficients * chunks
        if len(shard_coefficients) != len(final_chunks):
            raise ValueError(
                f"Shard coefficients dimensionality {len(shard_coefficients)} "
                f"must match chunks dimensionality {len(final_chunks)}"
            )
        final_shards = tuple(coef * chunk for coef, chunk in zip(shard_coefficients, final_chunks))
    elif shard_coefficients is None and final_format == 3:
        # Try to infer from input array (only for v3)
        if hasattr(array, 'shards') and array.shards is not None:
            final_shards = array.shards
        # If still None, no sharding will be used
    
    if overwrite:
        # Only remove something that already IS a zarr store: a mistyped output_path must
        # never delete an unrelated tree. _detect_zarr_format_local checks for
        # .zarray/.zgroup/zarr.json.
        _out = Path(output_path)
        if _out.exists():
            if not _out.is_dir():
                raise ValueError(
                    f"overwrite=True: {output_path} exists and is not a directory")
            if _detect_zarr_format_local(_out) is None and any(_out.iterdir()):
                raise ValueError(
                    f"overwrite=True refused: {output_path} is a non-empty directory that "
                    f"does not look like a zarr store (no .zarray/.zgroup/zarr.json). "
                    f"Delete it yourself if you really mean to replace it.")
            shutil.rmtree(_out)

    Path(output_path).mkdir(parents=True, exist_ok=True)

    driver = 'zarr3' if final_format == 3 else 'zarr'
    
    ts_spec = {
        'driver': driver,
        'kvstore': {
            'driver': 'file',
            'path': str(Path(output_path).absolute()),
        },
    }
    
    if final_format == 3:
        if final_shards is not None:
            # Zarr v3 with sharding
            # Shards are the outer chunks (files), chunks are inner chunks
            ts_spec['metadata'] = {
                'shape': list(input_shape_temp),
                'chunk_grid': {
                    'name': 'regular',
                    'configuration': {'chunk_shape': list(final_shards)}
                },
                'chunk_key_encoding': {
                    'name': 'default',
                    'configuration': {'separator': '/'}
                },
                'codecs': [
                    {
                        'name': 'sharding_indexed',
                        'configuration': {
                            'chunk_shape': list(final_chunks),
                            'codecs': final_codecs.to_v3_config(final_dtype_obj),
                            'index_codecs': [
                                {'name': 'bytes', 'configuration': {'endian': 'little'}},
                                {'name': 'crc32c'}
                            ]
                        }
                    }
                ],
                'data_type': final_dtype_v3_name,
            }
        else:
            # Zarr v3 without sharding
            ts_spec['metadata'] = {
                'shape': list(input_shape_temp),
                'chunk_grid': {
                    'name': 'regular',
                    'configuration': {'chunk_shape': list(final_chunks)}
                },
                'chunk_key_encoding': {
                    'name': 'default',
                    'configuration': {'separator': '/'}
                },
                'codecs': final_codecs.to_v3_config(final_dtype_obj),
                'data_type': final_dtype_v3_name,
            }
    else:
        # Zarr v2 - use the parsed v2 dtype format
        ts_spec['metadata'] = {
            'shape': list(input_shape_temp),
            'chunks': list(final_chunks),
            'dtype': final_dtype_v2_str,
            'dimension_separator': '/',
            'compressor': final_codecs.to_v2_config(),
        }
    
    output = ts.open(ts_spec, create=True).result()
    
    # Set up input array - use DynamicArray directly
    input_array = array
    input_shape = array.shape
    input_chunks = getattr(array, 'chunks', final_chunks)
    if input_chunks is None:
        input_chunks = final_chunks

    if region_shape is not None:
        # Explicit read-region shape OVERRIDES the region_size_mb heuristic. Used to make the
        # read regions match a producer's tile grid (e.g. tilewise-ccl's tile_shape), so a
        # position-aware pull op sees whole tiles and doesn't re-read/re-compute them. Must be
        # a multiple of the storage chunks per axis (each region then writes whole chunks, so
        # the parallel region writes stay race-free), and is clamped to the array shape.
        region_shape = tuple(int(r) for r in region_shape)
        if len(region_shape) != len(input_shape):
            raise ValueError(
                f"region_shape {region_shape} has {len(region_shape)} dims but array is "
                f"{len(input_shape)}-D")
        for r, c in zip(region_shape, final_chunks):
            if r % c != 0:
                raise ValueError(
                    f"region_shape {region_shape} must be a per-axis multiple of the output "
                    f"chunks {final_chunks} (so each region writes whole chunks)")
        region_shape = tuple(min(r, s) for r, s in zip(region_shape, input_shape))
        print(f"[Optimized] Output chunks: {final_chunks}, Region: {region_shape} (explicit region_shape)")
    else:
        # No explicit region_shape. If the op chain carries a read-ALIGNMENT grid (a
        # map_overlap built with align=..., as tilewise-ccl's Phase B is), the producer can
        # only emit whole cells of that grid: a smaller region still costs a full cell, and
        # the cell is recomputed once per region landing in it. Adopt the alignment as the
        # region so each cell is read and computed exactly once - the caller no longer has
        # to know the producer's tiling and restate it here.
        aligned = _aligned_region_shape(array, final_chunks, input_shape)
        if aligned is not None:
            region_shape = aligned
            print(f"[Optimized] Output chunks: {final_chunks}, Region: {region_shape} "
                  f"(from the op chain's alignment)")
        else:
            region_shape = _compute_region_shape(
                input_shape, final_chunks, region_size_mb,
                dtype=final_dtype, input_chunks=input_chunks
            )
            print(f"[Optimized] Output chunks: {final_chunks}, Region: {region_shape}, Size: ~{region_size_mb}MB")
    
    # Generate chunk indices
    chunk_indices = list(itertools.product(
        *[range(0, s, rs) for s, rs in zip(input_shape, region_shape)]
    ))
    
    total_chunks = len(chunk_indices)
    quarter_target = max(1, total_chunks // 4)
    tenth_target = max(1, total_chunks // 10)
    
    print(f"[Optimized] Total regions to process: {total_chunks}, Shape: {input_shape}, Region: {region_shape}", flush=True)
    
    # --- Pipeline sizing: ONE budget (max_workers = total live regions) split across the
    # three stages that actually pin buffers (readers / queue / in-flight writes).
    # Peak RAM ~= max_workers * region_bytes, so the caller can compute it from the two
    # knobs they set. Previously each stage was an independent multiple of max_workers
    # (8 + 8 + 16 = 32 live regions) with region SIZE never entering the formula: fine at
    # the 8MB default (~256MB), ~36GB at a 672^3 int32 region.
    region_bytes = int(np.prod(region_shape)) * int(np.dtype(final_dtype_obj).itemsize)

    requested_workers = max(1, int(max_workers))
    if memory_budget_mb is not None:
        # Ceiling, never a target: a generous budget must not RAISE concurrency above what
        # was asked for (at an 8MB region a 2GB budget would allow ~256 live regions).
        affordable = int((float(memory_budget_mb) * 1024 * 1024) // max(1, region_bytes))
        if affordable < _MIN_LIVE_REGIONS:
            warnings.warn(
                f"memory_budget_mb={memory_budget_mb:g} cannot fit even a minimal write "
                f"pipeline: one region is {region_bytes / 1024**2:.0f} MiB and at least "
                f"{_MIN_LIVE_REGIONS} must be live (~"
                f"{_MIN_LIVE_REGIONS * region_bytes / 1024**2:.0f} MiB). Running at the "
                f"floor; reduce region_shape/region_size_mb or raise the budget.",
                RuntimeWarning, stacklevel=2)
        max_workers = max(_MIN_LIVE_REGIONS, min(requested_workers, affordable))
    else:
        max_workers = max(_MIN_LIVE_REGIONS, requested_workers)

    # Split the live-region budget. Readers are the measured memory driver (1->8 readers:
    # 507->1089 MiB, no time change), so they stay caller-tunable via num_readers; the rest
    # goes to the queue and the in-flight write cap.
    if num_readers is None:
        num_readers = max(1, max_workers // 2)
    num_readers = max(1, min(int(num_readers), max_workers - 2))  # leave >=1 queue, >=1 write

    remaining = max_workers - num_readers
    queue_size = max(1, remaining // 2)
    max_inflight_writes = max(1, remaining - queue_size)

    # Writer THREADS are not part of the memory budget knob: .write() is async (returns a
    # future; TensorStore's C++ pool does the I/O), so writer count is measurably irrelevant
    # to throughput (1 vs 8 writers = same wall time under zstd) while each extra writer
    # holds a buffer. Keep a small constant, bounded by the in-flight cap.
    n_writer_threads = max(1, min(_WRITER_THREADS, max_inflight_writes))

    print(f"[Optimized] Live-region budget: {max_workers} "
          f"(~{max_workers * region_bytes / 1024**2:.0f} MiB at "
          f"{region_bytes / 1024**2:.0f} MiB/region) -> readers={num_readers}, "
          f"queue={queue_size}, inflight={max_inflight_writes}, "
          f"writer threads={n_writer_threads}", flush=True)

    chunk_queue = Queue(maxsize=queue_size)
    sentinel_lock = threading.Lock()
    
    state = {
        'read_idx': 0,
        'completed_readers': 0,
        'writes_processed': 0,
        'error': None,
        'start_time': time.time(),
    }
    
    # Lock for atomic read_idx access
    read_idx_lock = threading.Lock()
    
    # Collect async write futures
    write_futures = []
    futures_lock = threading.Lock()
    
    # Add a shutdown flag to cleanly stop threads
    shutdown_flag = threading.Event()
    
    def reader_thread():
        """Fast producer - no overhead."""
        # Each reader thread owns a CUDA stream so region GPU pipelines overlap (one
        # region's compute runs while another's H2D/D2H copies). None on CPU.
        gpu_stream = _new_stream(device)
        try:
            while not shutdown_flag.is_set():
                if state.get('error'):  # Check for errors
                    print(f"[Reader] Exiting due to error", flush=True)
                    break
                
                # Atomically get next chunk index
                with read_idx_lock:
                    current_read_idx = state['read_idx']
                    if current_read_idx >= len(chunk_indices):
                        print(f"[Reader] Finished - read_idx={current_read_idx} >= {len(chunk_indices)}", flush=True)
                        break
                    chunk_start = chunk_indices[current_read_idx]
                    current_idx = current_read_idx
                    state['read_idx'] += 1
                    next_read_idx = state['read_idx']
                
                if current_idx % 10 == 0:
                    print(f"[Reader] About to read region {current_idx+1}/{len(chunk_indices)}, next read_idx will be {next_read_idx}", flush=True)
                
                try:
                    chunk_slice = tuple(
                        slice(start, min(start + rs, dim_size))
                        for start, rs, dim_size in zip(chunk_start, region_shape, input_shape)
                    )
                    
                    # Read actual data using _read_direct to avoid creating SliceTransform.
                    # device_context is thread-local, so set it here (in the reader thread)
                    # so the op chain runs on the requested device for this region; the
                    # per-thread stream lets regions overlap on the GPU. Then bring the
                    # region back to host (D2H) since tensorstore writes numpy.
                    with _use_stream(gpu_stream), _device_context(device):
                        data = input_array._read_direct(chunk_slice)
                        # pinned D2H: faster + avoids pinned-staging contention across
                        # the concurrent reader threads (measured ~7-18% faster on GPU).
                        data = _asnumpy_pinned(data)

                    # Convert dtype if needed (allows unsafe casting if explicitly requested)
                    if data.dtype != final_dtype_obj:
                        data = data.astype(final_dtype_obj, copy=False)

                    chunk_queue.put((chunk_slice, data))
                    
                    if current_idx % 10 == 0:
                        print(f"[Reader] Queued region {current_idx+1}/{len(chunk_indices)}", flush=True)
                    
                except Exception as chunk_error:
                    print(f"[Reader] ERROR at region {current_idx}: {chunk_error}", flush=True)
                    import traceback
                    traceback.print_exc()
                    state['error'] = chunk_error
                    raise
            
            # Sentinel coordination
            with sentinel_lock:
                state['completed_readers'] += 1
                should_send_sentinels = (state['completed_readers'] == num_readers)
                print(f"[Reader] Normal exit - completed_readers now {state['completed_readers']}/{num_readers}", flush=True)
            
            if should_send_sentinels:
                for _ in range(n_writer_threads):
                    chunk_queue.put(None)
                    
        except Exception as e:
            print(f"[Reader] FATAL ERROR in outer handler: {e}", flush=True)
            import traceback
            traceback.print_exc()
            state['error'] = e
    
    def _reap_futures():
        """Drop every completed write future (freeing its pinned region buffer) and return
        how many are still in flight. Surfaces write errors as soon as they complete."""
        with futures_lock:
            done = [f for f in write_futures if f.done()]
            for f in done:
                write_futures.remove(f)
            n_inflight = len(write_futures)
        for f in done:
            f.result()  # surface any write error early (already complete)
        return n_inflight

    def writer_thread():
        """Async writer - uses TensorStore futures for parallelism."""
        try:
            while not shutdown_flag.is_set():
                if state.get('error'):  # Check for errors using .get() to be safe
                    break
                
                try:
                    item = chunk_queue.get(block=True)
                except Exception as e:
                    print(f"[Writer] Queue get error: {e}", flush=True)
                    break
                
                if item is None:
                    chunk_queue.task_done()
                    break
                
                try:
                    chunk_slice, data = item
                    
                    # ASYNC write - returns immediately, actual write happens in parallel
                    write_future = output[chunk_slice].write(data)
                    state['writes_processed'] += 1
                    
                    # Commit futures immediately to ensure all are tracked
                    with futures_lock:
                        write_futures.append(write_future)

                    chunk_queue.task_done()

                    # Release, THEN backpressure - two separate jobs that used to share one
                    # loop. Each pending write future pins its ~region_size data buffer until
                    # it commits, and a *done* future keeps pinning it until the future object
                    # itself is dropped. So reclamation (pruning finished futures) must happen
                    # on EVERY pass regardless of the cap; only the blocking wait is governed
                    # by max_inflight_writes. Folding the two together meant a lower cap could
                    # not be reasoned about independently of when buffers were freed.
                    n_inflight = _reap_futures()
                    while (n_inflight >= max_inflight_writes
                           and not shutdown_flag.is_set() and not state.get('error')):
                        time.sleep(0.001)
                        n_inflight = _reap_futures()
                    
                except Exception as e:
                    print(f"[Writer] ERROR: {e}", flush=True)
                    if 'error' in state:  # Only set if state dict still exists
                        state['error'] = e
                    chunk_queue.task_done()
                    break
            
            # No final commit needed - futures are already added immediately
                    
        except Exception as e:
            print(f"[Writer] ERROR in outer handler: {e}", flush=True)
            if 'error' in state:  # Only set if state dict still exists
                state['error'] = e

    print(f"[Optimized] Starting: {num_readers} readers, {n_writer_threads} writers (async), queue={queue_size}")
    
    readers = [
        threading.Thread(target=reader_thread, daemon=True, name=f"Reader-{i}")
        for i in range(num_readers)
    ]
    for r in readers:
        r.start()
    
    writers = [
        threading.Thread(target=writer_thread, daemon=True, name=f"Writer-{i}")
        for i in range(n_writer_threads)
    ]
    for w in writers:
        w.start()
    
    # Progress monitoring with integrated early timeout checking
    stop_monitor = threading.Event()
    def monitor_progress():
        """Non-blocking progress monitoring with early timeout checking."""
        last_read = 0
        tenth_checked = False
        quarter_checked = False
        
        while not stop_monitor.is_set():
            # Interruptible wait: Event.wait() returns immediately when stop_monitor is
            # set, so a fast write isn't padded by a full sleep interval. Using a plain
            # time.sleep(2.0) here left monitor.join(timeout=2.0) blocking ~2s on every
            # write (a fixed latency floor that dominated small/medium writes).
            if stop_monitor.wait(2.0):
                break
            try:
                current_read = state['read_idx']
                with futures_lock:
                    total_futures = len(write_futures)
                    completed_writes = sum(1 for f in write_futures if f.done())
                
                # Early timeout check - only check ONCE when threshold is reached
                if not tenth_checked and early_tenth_timeout is not None and completed_writes >= tenth_target:
                    elapsed_from_start = time.time() - state['start_time']
                    tenth_checked = True
                    if elapsed_from_start > early_tenth_timeout:
                        msg = (
                            f"Early tenth timeout: took {elapsed_from_start:.1f}s to write "
                            f"{tenth_target} regions (limit: {early_tenth_timeout:.1f}s). Aborting."
                        )
                        print(f"[Monitor] {msg}", flush=True)
                        state['error'] = TimeoutError(msg)
                        stop_monitor.set()
                        return
                    else:
                        print(f"[Monitor] Tenth milestone reached in {elapsed_from_start:.1f}s - continuing", flush=True)
                
                if not quarter_checked and early_quarter_timeout is not None and completed_writes >= quarter_target:
                    elapsed_from_start = time.time() - state['start_time']
                    quarter_checked = True
                    if elapsed_from_start > early_quarter_timeout:
                        msg = (
                            f"Early quarter timeout: took {elapsed_from_start:.1f}s to write "
                            f"{quarter_target} regions (limit: {early_quarter_timeout:.1f}s). Aborting."
                        )
                        print(f"[Monitor] {msg}", flush=True)
                        state['error'] = TimeoutError(msg)
                        stop_monitor.set()
                        return
                    else:
                        print(f"[Monitor] Quarter milestone reached in {elapsed_from_start:.1f}s - continuing", flush=True)
                
                if current_read >= total_chunks:
                    break
                if current_read > last_read:
                    progress_pct = (current_read / total_chunks) * 100
                    q_size = chunk_queue.qsize()
                    elapsed = time.time() - state['start_time']
                    writes_proc = state.get('writes_processed', 0)
                    print(f"[Progress] Read: {current_read}/{total_chunks} ({progress_pct:.1f}%), Q:{q_size}, Writes: {writes_proc} processed, {completed_writes}/{total_futures} done, Time: {elapsed:.1f}s")
                    last_read = current_read
            except (KeyError, NameError):
                break
    
    monitor = threading.Thread(target=monitor_progress, daemon=True, name="Monitor")
    monitor.start()
    
    # Wait for all readers to finish
    for r in readers:
        r.join()
        if state.get('error'):
            shutdown_flag.set()  # Signal all threads to stop
            break
    
    # Wait for all writers to finish
    for w in writers:
        w.join()
        if state.get('error'):
            shutdown_flag.set()  # Signal all threads to stop
            break
    
    # Stop monitor thread
    stop_monitor.set()
    monitor.join(timeout=2.0)
    
    # Signal shutdown before checking error
    shutdown_flag.set()
    
    if state.get('error'):
        raise state['error']
    
    print(f"\n[Optimized] All reads/writes queued, waiting for {len(write_futures)} async writes to complete...")
    
    # Wait for all async writes to complete
    last_gc_time = time.time()
    max_wait = 30  # Maximum seconds to wait for futures
    wait_start = time.time()
    
    while True:
        if state['error']:
            raise state['error']
        
        with futures_lock:
            completed_count = sum(1 for f in write_futures if f.done())
            total_futures = len(write_futures)
        
        if completed_count >= total_futures:
            break
        
        # Timeout on waiting for futures
        if time.time() - wait_start > max_wait:
            print(f"[Main] Timeout waiting for futures: {completed_count}/{total_futures} done", flush=True)
            break
        
        # Periodic GC
        now = time.time()
        if now - last_gc_time >= gc_interval:
            gc.collect()
            last_gc_time = now
        
        time.sleep(0.05)  # Check frequently for errors
    
    # Ensure shutdown flag is set before verification
    shutdown_flag.set()
    
    # Ensure all futures succeeded (only if no error)
    if not state.get('error'):
        print(f"[Optimized] Verifying all writes succeeded...")
        for i, future in enumerate(write_futures):
            try:
                future.result()  # Will raise if write failed
            except Exception as e:
                print(f"[Main] Write future {i} failed: {e}")
                raise
    
    elapsed = time.time() - state['start_time']
    throughput = total_chunks / elapsed if elapsed > 0 else 0
    
    if not state.get('error'):
        print(f"\n[Optimized] Completed: {total_chunks} regions in {elapsed:.1f}s ({throughput:.2f} regions/s)")
        print(f"Successfully wrote array to {output_path}")
    
    # Stop monitor thread before cleanup (if not already stopped)
    stop_monitor.set()
    monitor.join(timeout=2.0)
    
    # Ensure all threads are stopped before cleanup
    shutdown_flag.set()
    for r in readers:
        if r.is_alive():
            r.join(timeout=2.0)
    for w in writers:
        if w.is_alive():
            w.join(timeout=2.0)
    
    # Cleanup
    del write_futures
    del input_array
    del output
    # Keep shutdown_flag and state until very end to avoid NameError
    temp_error = state.get('error')
    del state
    gc.collect()
    
    if gc_was_enabled:
        gc.enable()
        gc.collect()
    
    # Clean up shutdown flag last
    del shutdown_flag
    
    if temp_error:
        raise temp_error
    
    return output_path

# I/O namespace class
class io:
    """I/O operations for reading and writing arrays."""
    
    @staticmethod
    def read(source: Union[str, Path]) -> 'DynamicArray':
        """Read array from file path. See read_array() for details."""
        return read_array(source)
    
    @staticmethod
    def write(
        array: 'DynamicArray',
        output_path: str,
        max_workers: int = 4,
        num_readers: Optional[int] = None,
        chunks: Optional[Tuple[int, ...]] = None,
        shard_coefficients: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Any] = None,
        compressor: Optional[Union[Codecs, Any]] = None,
        zarr_format: Optional[int] = None,
        region_size_mb: float = 8.0,
        gc_interval: float = 15.0,
        early_quarter_timeout: Optional[float] = None,
        early_tenth_timeout: Optional[float] = None,
        device: Optional[str] = None,
        region_shape: Optional[Tuple[int, ...]] = None,
        memory_budget_mb: Optional[float] = None,
        overwrite: bool = False,
    ):
        """Write array to Zarr. See write_array() for details. ``device`` ('cpu'|'cuda')
        runs each region's op chain on that device (results are written from host).

        This wrapper mirrors ``write_array``'s signature EXACTLY and forwards BY KEYWORD.
        It used to forward positionally with a trailing ``**kwargs``, which meant a
        parameter it did not name (``region_shape``) still reached ``write_array`` - by
        accident of argument order - while ``inspect.signature`` reported it unsupported,
        and a genuine typo was swallowed in silence. ``test_write_signature_parity`` keeps
        the two in lockstep."""
        return write_array(
            array,
            output_path,
            max_workers=max_workers,
            num_readers=num_readers,
            chunks=chunks,
            shard_coefficients=shard_coefficients,
            dtype=dtype,
            compressor=compressor,
            zarr_format=zarr_format,
            region_size_mb=region_size_mb,
            gc_interval=gc_interval,
            early_quarter_timeout=early_quarter_timeout,
            early_tenth_timeout=early_tenth_timeout,
            device=device,
            region_shape=region_shape,
            memory_budget_mb=memory_budget_mb,
            overwrite=overwrite,
        )
