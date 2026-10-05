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
from dataclasses import dataclass, field
import itertools
import math
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


def _local_path_from_file_url(path):
    """A ``file://`` URL as the local path it names; anything else unchanged.

    A file URL is LOCAL storage spelled as a URL. Left as a URL it was classified as
    remote here (no directory created) while the zarrista backend treated it as a local
    path and passed it to ``Path(...).resolve()`` verbatim - on Windows that turns
    ``file:///C:/x`` into the drive-RELATIVE ``C:x``, so the write "succeeded" into a
    directory under the current working directory. Converted once, here, both writers
    see a plain path.
    """
    s = str(path)
    if not s.lower().startswith("file://"):
        return path
    from urllib.request import url2pathname
    parsed = urlparse(s)
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(
            f"{s!r} names a file on host {parsed.netloc!r}; only local file:// URLs "
            f"(file:///path) are supported. Use the path itself instead.")
    return url2pathname(parsed.path)


#: TensorStore only accepts anonymous S3 credentials ({"type": "anonymous"}) from here on.
_TS_ANONYMOUS_S3_MIN = (0, 1, 72)


def _ts_version():
    try:
        import importlib.metadata as _md
        return tuple(int(p) for p in _md.version("tensorstore").split(".")[:3])
    except Exception:
        return (0, 0, 0)


def _ts_kvstore(url, storage_options=None, *, write=False):
    """TensorStore kvstore spec for a remote ``url``, with ``storage_options`` applied.

    The SAME option names as ``backend='zarrista'`` (which hands them to
    ``obstore.store.from_url``), so switching backends does not mean rewriting options:

    - ``endpoint`` (or ``aws_endpoint``): a non-AWS S3 (MinIO, Ceph, EMBL, ...);
    - ``region`` (or ``aws_region``);
    - ``skip_signature=True``: anonymous access to a public bucket;
    - ``virtual_hosted_style_request=False``: accepted - TensorStore makes path-style
      requests to a custom endpoint anyway; ``True`` with an endpoint is refused;
    - ``client_options={'allow_http': ...}``: accepted (TensorStore needs no permission to
      use an http:// endpoint).

    Anything else is REFUSED, never ignored (fail loudly): TensorStore takes no inline
    credentials - it reads the standard AWS variables/files (S3) and Application Default
    Credentials (GCS). Also refused: writing over plain HTTP (TensorStore's HTTP store is
    read-only) and schemes it has no store for (az://).
    """
    from .backends.zarrista_backend import UnsupportedByBackend

    storage_type, _ = _parse_storage_location(str(url))
    opts = dict(storage_options or {})
    text = str(url)

    def refuse(msg):
        raise UnsupportedByBackend(msg)

    def leftover():
        if opts:
            refuse(
                f"storage_options {sorted(opts)} are not supported by the TensorStore "
                f"backend for {text!r}. It accepts endpoint, region, skip_signature, "
                f"virtual_hosted_style_request and client_options={{'allow_http': ...}}; "
                f"credentials come from the standard AWS environment variables / ~/.aws "
                f"files (S3) or Application Default Credentials (GCS). Or use "
                f"backend='zarrista', which passes storage_options to obstore.")

    if storage_type in ("s3", "gcs"):
        bucket, _, key = text.split("://", 1)[1].partition("/")
        kv = {"driver": "s3" if storage_type == "s3" else "gcs", "bucket": bucket,
              "path": key.rstrip("/") + "/" if key else ""}
        if storage_type == "s3":
            endpoint = opts.pop("endpoint", None) or opts.pop("aws_endpoint", None)
            if endpoint:
                kv["endpoint"] = endpoint
            region = opts.pop("region", None) or opts.pop("aws_region", None)
            if region:
                kv["aws_region"] = region
            if opts.pop("skip_signature", False):
                if _ts_version() < _TS_ANONYMOUS_S3_MIN:
                    refuse(
                        f"skip_signature=True (anonymous S3) needs tensorstore >= "
                        f"{'.'.join(map(str, _TS_ANONYMOUS_S3_MIN))}; this is "
                        f"{'.'.join(map(str, _ts_version()))}. Upgrade tensorstore, or "
                        f"use backend='zarrista'.")
                kv["aws_credentials"] = {"type": "anonymous"}
            if opts.pop("virtual_hosted_style_request", False) and endpoint:
                refuse("virtual_hosted_style_request=True is not supported with a "
                       "custom endpoint on the TensorStore backend (it makes path-style "
                       "requests there). Drop it, or use backend='zarrista'.")
            client = dict(opts.pop("client_options", None) or {})
            client.pop("allow_http", None)
            if client:
                opts["client_options"] = client
        leftover()
        return kv
    if storage_type in ("http", "https"):
        if write:
            refuse(
                f"cannot write to {text!r} with the TensorStore backend: its HTTP store is "
                f"read-only. Use backend='zarrista' (which writes with HTTP PUT), or the "
                f"s3:// form with storage_options={{'endpoint': ...}} for an S3 service.")
        leftover()
        return {"driver": "http", "base_url": text.rstrip("/") + "/"}
    refuse(f"{text!r}: the TensorStore backend has no store for this scheme (it supports "
           f"s3://, gs://, and http(s):// for reading). Use backend='zarrista' (obstore "
           f"also covers az://).")


def _ts_open_remote_zarr(url, storage_options=None):
    """Open a remote zarr array with TensorStore: v3 first, then v2.

    Only a MISSING-metadata failure moves on to v2; any other error (credentials,
    network, a bad option) is raised as-is instead of being masked by the v2 attempt.
    """
    kv = _ts_kvstore(url, storage_options)
    errors = []
    for driver in ("zarr3", "zarr"):
        try:
            return ts.open({"driver": driver, "kvstore": kv}, read=True).result()
        except ValueError as exc:
            errors.append(f"{driver}: {exc}")
            if "NOT_FOUND" not in str(exc) and "not found" not in str(exc).lower():
                raise
    raise ValueError(f"no zarr v3 or v2 array at {url!r}:\n  " + "\n  ".join(errors))


def _storage_settings_from_metadata(md):
    """``(zarr_format, Codecs or None, shards or None)`` from an array's metadata JSON.

    The inheritable storage settings for sources that expose their metadata as a dict
    rather than a zarr-python array: zarrista arrays (``.metadata``) and remote
    TensorStore handles (``spec().to_json()['metadata']``). Local TensorStore reads get
    the same three from zarr-python (DynamicArray._extract_zarr_metadata), so every
    backend and location inherits alike. Unknown or unreadable -> None (no preference).
    """
    if not isinstance(md, dict):
        return None, None, None
    fmt = md.get("zarr_format")
    if fmt not in (2, 3):
        fmt = 3 if "data_type" in md else (2 if "dtype" in md else None)
    codecs = shards = None
    if fmt == 2:
        comp = md.get("compressor")
        if comp is None:
            codecs = Codecs(compressor=None)
        else:
            try:
                import numcodecs
                codecs = Codecs.from_numcodecs(numcodecs.get_codec(dict(comp)))
            except Exception:
                codecs = None
    elif fmt == 3:
        chain = list(md.get("codecs") or [])
        for c in chain:
            if isinstance(c, dict) and c.get("name") == "sharding_indexed":
                try:
                    shards = tuple(int(s) for s in
                                   md["chunk_grid"]["configuration"]["chunk_shape"])
                except Exception:
                    shards = None
                chain = list(c.get("configuration", {}).get("codecs") or [])
                break
        compressors = [c for c in chain if isinstance(c, dict)
                       and c.get("name") not in ("bytes", "transpose", "crc32c")]
        codecs = (DynamicArray._extract_codecs_from_v3(compressors) if compressors
                  else Codecs(compressor=None))
    return fmt, codecs, shards


def _apply_storage_settings(arr, md):
    """Set a source DynamicArray's inheritable format/codecs/shards from metadata JSON."""
    fmt, codecs, shards = _storage_settings_from_metadata(md)
    arr._zarr_format, arr._codecs, arr._shards = fmt, codecs, shards
    arr._compressor = arr._compressors = None
    return arr


def _ts_zarr_format(ts_array):
    """Zarr format (2 or 3) of a TensorStore handle, from its spec's driver; else None."""
    try:
        driver = ts_array.spec().to_json().get("driver")
    except Exception:
        return None
    return {"zarr3": 3, "zarr": 2}.get(driver)


def _forget_storage_format(arr):
    """Drop the inheritable storage settings of a source whose store is incidental."""
    arr._zarr_format = None
    arr._compressor = None
    arr._compressors = None
    arr._shards = None
    arr._codecs = None
    return arr


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


def read_file(file_path, storage_options=None):
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
            # A zarr array on cloud storage: v3 first, then v2, with storage_options
            # (endpoint/region/anonymous) applied exactly as for writing.
            return _ts_open_remote_zarr(file_path, storage_options)

    elif storage_type in ('http', 'https'):
        # HTTP/HTTPS URL
        path_str = str(parsed_path)

        if _is_tiff_path(path_str):
            # TIFF over HTTP - pass to tifffile (may support via fsspec)
            return open_tiff_zarr(file_path)
        else:
            # Zarr over HTTP (read-only store): v3 first, then v2 - it used to try v2 only
            return _ts_open_remote_zarr(file_path, storage_options)
    
    else:
        raise ValueError(
            f"Unsupported storage type or file format: {file_path}. "
            f"Expected local path, gs://, s3://, http://, or https:// URL"
        )


# I/O operations for reading and writing arrays


def read_array(source: Union[str, Path], backend: Optional[str] = None,
               storage_options: Optional[dict] = None) -> 'DynamicArray':
    """
    Read array from file path (TIFF or Zarr).

    Supports both local and remote storage:
    - Local: /path/to/file.tif, /path/to/array.zarr
    - GCS: gs://bucket/path/to/file
    - S3: s3://bucket/path/to/file
    - HTTP: http://example.com/path/to/file

    Args:
        source: Path or URL to a TIFF file or Zarr array
        backend: Storage backend for Zarr sources. ``'tensorstore'`` (default) or
            ``'zarrista'``, the optional Rust-backed reader
            (``pip install "dyna-zarr[zarrista]"``). Remote URLs additionally need
            ``obstore``. It selects a ZARR backend only: TIFF is read by tifffile
            whatever the value, so passing ``backend`` with a TIFF source is an
            error rather than a silent no-op. Defaults to ``'tensorstore'``.
        storage_options: Options for the remote store, forwarded to
            ``obstore.store.from_url`` (``backend='zarrista'`` only). Needed for any
            non-AWS S3: e.g. ``{'endpoint': 'https://s3.example.org', 'region': ...,
            'virtual_hosted_style_request': False}``, plus ``skip_signature=True``
            for a public bucket.

    Returns:
        DynamicArray wrapping the opened array
    """
    from dyna_zarr.io import read_file

    explicit_backend = backend is not None
    if backend is None:
        backend = "tensorstore"
    if backend not in ("tensorstore", "zarrista"):
        raise ValueError(
            f"unknown backend {backend!r}; expected 'tensorstore' (default) or 'zarrista'"
        )

    source_path = Path(source) if not isinstance(source, Path) else source
    
    is_tiff = str(source_path).lower().endswith(('.tif', '.tiff'))

    if is_tiff and explicit_backend:
        # `backend=` selects a ZARR storage backend, and neither value applies to
        # TIFF: tifffile reads it under both, tensorstore included. Accepting the
        # argument silently would misreport which reader actually ran.
        from .backends.zarrista_backend import UnsupportedByBackend

        raise UnsupportedByBackend(
            f"backend={backend!r} selects a Zarr storage backend and does not apply "
            f"to the TIFF source {str(source)!r}: TIFF is read by tifffile under "
            f"every backend. Drop the backend argument, or convert first: "
            f"io.write(io.read('x.tif'), 'x.zarr')."
        )

    if backend == "zarrista":
        # ONE guard, before any work: every combination this backend cannot honour
        # is refused here rather than silently doing something else further down.
        from .backends.zarrista_backend import check_supported as _zst_check

        _zst_check(source=source, storage_options=storage_options)
    elif storage_options and '://' not in str(source):
        # Same rule as the zarrista guard: options for a remote store make no sense on
        # a local path, and ignoring them silently would hide a wrong source.
        from .backends.zarrista_backend import UnsupportedByBackend
        raise UnsupportedByBackend(
            f"storage_options are for remote stores, but the source {str(source)!r} is a "
            f"local path. Drop storage_options, or pass a remote URL.")

    # TIFF -> lazy zarr array via tifffile's bridge, wrapped as a normal zarr-backed
    # DynamicArray (laziness/slicing/memory-bounded reads all come from DynamicArray).
    if is_tiff:
        # A TIFF has no Zarr format of its own (tifffile's bridge store has one, but
        # that is its implementation, not the data's), so nothing is inherited from it:
        # io.write then applies its defaults (v3, blosc-lz4).
        return _forget_storage_format(read_tiff_lazy(source))

    # Optional zarrista backend: a Zarr-only reader (local via its sync API, object
    # stores via its async API over obstore), so it is dispatched here and everything
    # else keeps the default path. It raises rather than silently falling back to
    # tensorstore - an explicit backend= that quietly did something else would make a
    # benchmark or a bug report meaningless.
    if backend == "zarrista":
        # Local paths use zarrista's sync API; object stores (s3://, gs://, az://,
        # http://) go through its async API over obstore, which open_array handles.
        from .backends.zarrista_backend import open_array as _zarrista_open

        handle = _zarrista_open(source, storage_options=storage_options)
        # zarrista exposes the array's metadata JSON (local and remote alike): the same
        # format/codecs/shards a TensorStore read inherits.
        return _apply_storage_settings(DynamicArray(handle),
                                       getattr(handle._array, "metadata", None))

    if (isinstance(source_path, Path) and source_path.is_dir() and
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
        ts_array = read_file(source, storage_options)
        dyn_array = object.__new__(DynamicArray)
        dyn_array._ts_array = ts_array
        dyn_array._is_tensorstore = True
        dyn_array._source = source
        dyn_array._shape = tuple(ts_array.shape)
        dyn_array._dtype = ts_array.dtype
        dyn_array._transform = None
        # The array's own (inner) chunk grid, as zarrista reports it for remote arrays;
        # it used to be None here, so a remote source lost its grid.
        try:
            grid = tuple(int(c) for c in ts_array.chunk_layout.read_chunk.shape)
            dyn_array._chunks = grid if len(grid) == len(dyn_array._shape) else None
        except Exception:
            dyn_array._chunks = None
        dyn_array._zarr_array = None
        # format/codecs/shards from the store's metadata, as for every other source
        try:
            md = ts_array.spec().to_json().get("metadata")
        except Exception:
            md = None
        _apply_storage_settings(dyn_array, md)
        if dyn_array._zarr_format is None:
            dyn_array._zarr_format = _ts_zarr_format(ts_array)
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

#: Target size of a DEFAULT output chunk, in MiB. Only used when the caller passes no
#: `chunks=`. Deliberately small: a chunk is the unit that `_compute_region_shape`
#: cannot subdivide, so an oversized default silently raises the floor on peak memory
#: for every write that does not name its own chunks.
_DEFAULT_CHUNK_MB = 1.0

#: Target size of an auto-chosen READ REGION, in MiB, when `region_size_mb` is not
#: given. The region is the unit pulled through the op chain, so this (times the
#: live-region count) is what bounds peak RAM.
_DEFAULT_REGION_MB = 8.0


#: How far the chunk and region budgets may be missed to buy exact alignment. Both
#: are targets, not contracts, so trading a little size for a grid that divides
#: cleanly is worth it: a misaligned read decodes partial chunks on both sides.
_BUDGET_TOLERANCE = 0.5

#: Floor on an auto-chosen chunk, as a fraction of the chunk budget. Shrinking a chunk
#: to buy alignment is cheap up to a point and catastrophic past it: with no floor at
#: all, a prime source grid produced a 260-byte chunk, i.e. millions of files, and a
#: write that never finished. Paired with _MIN_CHUNK_BYTES so a tiny budget cannot
#: scale the floor down to nothing.
_CHUNK_FLOOR_FRACTION = 1.0 / 8.0

#: Absolute floor on an auto-chosen chunk, in bytes.
_MIN_CHUNK_BYTES = 64 * 1024

#: An axis at most this long counts as "short" and is completed by the region before
#: any budget goes to the inner axes. Short outer axes are the t/c of tczyx: splitting
#: one forces an outer-axis halo, which in C order costs a whole inner volume per
#: index. Completing it is cheap and removes that halo entirely.
_SHORT_AXIS = 16


def _axis_candidates(input_chunk, size):
    """Sizes on one axis that tile the source grid exactly.

    Divisors of the stored chunk (a stored chunk splits into a whole number of
    these) and multiples of it (each of these covers whole stored chunks). Either
    way a read never straddles a chunk boundary.
    """
    ic = max(1, min(int(input_chunk), int(size)))
    out = {d for d in range(1, ic + 1) if ic % d == 0}
    k = 1
    while ic * k <= size:
        out.add(ic * k)
        k += 1
    out.add(int(size))
    return sorted(v for v in out if 1 <= v <= size)


def _nearest_aligned(target, candidates, prefer_smaller=1.25, size=None):
    """Candidate closest to ``target``, leaning toward the SMALLER one.

    Undershooting costs a little throughput; overshooting costs memory, and memory
    boundedness is the property that must not slip. ``prefer_smaller`` > 1 makes a
    larger candidate win only when it is clearly nearer.

    When ``size`` is given, candidates that also tile the AXIS exactly are preferred
    over merely-nearer ones. Tiling the source grid is not enough on its own: 104 is
    a multiple of a source chunk of 8, but on a 256-long axis it leaves a ragged
    48-deep final chunk in every row, so writes keep touching partial chunks
    (measured ~1.6x slower than the clean 128 next to it).
    """
    if not candidates:
        return max(1, int(target))

    def _pick(pool):
        below = [c for c in pool if c <= target]
        above = [c for c in pool if c > target]
        if not below:
            return above[0]
        if not above:
            return below[-1]
        lo, hi = below[-1], above[0]
        # Compare by RATIO, not by absolute distance. Sizes are multiplicative: from
        # a target of 63, dropping to 1 is a 63x shrink while rising to 127 is only a
        # 2x growth, yet 1 is "nearer" by subtraction. That is how a prime source
        # chunk produced a (65, 1, 1) output chunk - 260 bytes, millions of files,
        # and a write that never finished. `prefer_smaller` still tilts ties toward
        # the smaller side, which is the memory-safe direction.
        return lo if (target / lo) <= (hi / target) * prefer_smaller else hi

    if size:
        clean = [c for c in candidates if size % c == 0 or c >= size]
        if clean:
            best = _pick(clean)
            # only accept the clean choice when it is not a big step from the target
            if abs(best - target) <= max(1.0, target) * 0.5:
                return best
    return _pick(candidates)


def _aligned_grid(shape, input_chunks, chunk_dtype, chunk_mb, region_mb,
                  region_dtype=None):
    """Solve for an output chunk and a region that BOTH tile the source grid.

    ``chunk_mb`` and ``region_mb`` are targets, not contracts. Holding either
    exactly usually leaves the two grids coprime, and the region satisfying both is
    then their LCM - for a (101,101,101) chunk over (8,256,256) storage that is
    515 GB, so alignment is abandoned and every read decodes partial chunks on both
    sides.

    Letting each axis move by a few voxels removes the conflict entirely: pick the
    chunk from the divisors/multiples of the stored chunk (so chunk and storage
    align), then the region from multiples of that chunk (so region and chunk
    align, hence region and storage too). Nothing is ever rounded up to a common
    multiple - a (102, 101) region against a (101, 102) chunk becomes (102, 102),
    not their 10302-voxel LCM.

    ``region_dtype`` sizes the region and defaults to ``chunk_dtype``; they differ
    when the chain narrows, and the region must be budgeted in the widest dtype it
    actually carries.

    Returns ``(chunk, region)`` or None when the source grid is unknown.
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    if input_chunks is None or len(input_chunks) != ndim or ndim == 0:
        return None

    chunk_itemsize = max(1, int(_as_np_dtype(chunk_dtype).itemsize))
    region_itemsize = max(1, int(_as_np_dtype(region_dtype or chunk_dtype).itemsize))
    chunk_budget = max(1, int(chunk_mb * 1024 * 1024) // chunk_itemsize)
    region_budget = max(1, int(region_mb * 1024 * 1024) // region_itemsize)

    candidates = [_axis_candidates(ic, s) for ic, s in zip(input_chunks, shape)]

    # --- chunk: isotropic over the last <=3 axes, leading axes at 1, each axis
    # snapped to the nearest aligned size.
    spatial = list(range(max(0, ndim - 3), ndim))
    side = max(1, int(chunk_budget ** (1.0 / max(1, len(spatial)))))
    chunk = [1] * ndim
    for axis in spatial:
        chunk[axis] = _nearest_aligned(min(side, shape[axis]), candidates[axis],
                                       size=shape[axis])

    def _prod(v):
        out = 1
        for x in v:
            out *= int(x)
        return out

    # Shrink the largest axis (to its next aligned size down) until it fits.
    # A region is at minimum ONE chunk, so the chunk must also respect the region
    # budget: otherwise a 1 MB region budget with a 4 MB chunk silently runs 4x over.
    # Shrinking is always possible (1 divides everything) and only ever costs
    # throughput, whereas overshooting costs memory - the property that must hold.
    effective_chunk_budget = chunk_budget
    if region_itemsize:
        # express the region budget in CHUNK elements, since the chunk is stored in
        # chunk_dtype but carried through the pipeline as region_dtype
        region_budget_as_chunk_elems = max(
            1, int(region_mb * 1024 * 1024) // region_itemsize)
        effective_chunk_budget = min(chunk_budget, region_budget_as_chunk_elems)

    while _prod(chunk) > effective_chunk_budget:
        axis = max(spatial, key=lambda a: chunk[a])
        smaller = [c for c in candidates[axis] if c < chunk[axis]]
        if not smaller:
            break
        # Step down to the largest smaller candidate that still TILES the axis, so
        # shrinking cannot reintroduce the ragged final chunk _nearest_aligned just
        # avoided. Fall back to the plain step only when nothing clean is available.
        clean = [c for c in smaller if shape[axis] % c == 0]
        chunk[axis] = clean[-1] if clean else smaller[-1]

    # --- region: the chunk scaled UNIFORMLY, so the proportions the chunk chose
    # are preserved. The chunk already encodes how much each axis is worth: a
    # singleton t/c says "this axis is thin", and the region must not then undo that
    # by expanding it tenfold. Growing every axis by the same multiple keeps the two
    # grids in the same shape, which is what makes halo overhead uniform instead of
    # concentrated on whichever axis the growth loop happened to reach first.
    #
    # An axis that is already whole stops early and hands its share to the others, so
    # a short axis is completed (removing its halo entirely) without the region
    # needing a special case for it.
    region = [min(c, s) for c, s in zip(chunk, shape)]
    scale = 1
    while True:
        nxt = scale + 1
        trial = [min(int(c) * nxt, s) for c, s in zip(chunk, shape)]
        if trial == region:                     # every axis already at its limit
            break
        if _prod(trial) > region_budget:
            break
        region, scale = trial, nxt

    # Spend whatever the uniform scale left over on the INNER axes, last first: they
    # are contiguous in C order, so a read there is one sequential run and their halo
    # is the cheapest per voxel.
    grew = True
    while grew:
        grew = False
        for axis in range(ndim - 1, -1, -1):
            if region[axis] >= shape[axis]:
                continue
            nxt = min(shape[axis], region[axis] + chunk[axis])
            if shape[axis] - nxt < chunk[axis]:
                nxt = shape[axis]          # take the tail rather than leave a sliver
            trial = list(region)
            trial[axis] = nxt
            if _prod(trial) <= region_budget:
                region = trial
                grew = True
                break
    return tuple(int(v) for v in chunk), tuple(int(v) for v in region)


def _solve_grids(shape, input_chunks, chunk_dtype, region_dtype,
                 chunks=None, chunk_mb=None, region_shape=None, region_mb=None,
                 halo=None):
    """Choose an output chunk and a read region together.

    The one rule: **exact forms are honoured, size forms are approximate.** A
    caller who passes a tuple gets that tuple; a caller who passes a budget gets a
    shape near it that also satisfies the alignment rules below.

    Defaults: ``chunks`` unset and ``chunk_mb`` unset means PRESERVE THE INPUT
    CHUNKS. ``region_mb`` defaults to 8 MiB.

    Alignment, in priority order:

    1. The region is a whole multiple of the output chunk. MANDATORY. Regions are
       written concurrently, so a region covering part of a chunk means two writers
       racing on that chunk. This is correctness and is never traded away.
    2. The region tiles the INPUT chunks. Strongly preferred: a misaligned read
       straddles one extra source chunk per axis.
    3. Both budgets are met, within tolerance.

    Why a JOINT search and not a sequential one. "Adjust the region, and if that
    fails adjust the chunk" misses every solution that needs a small move in both:
    a chunk one step off its budget can admit a perfectly aligned region where the
    same chunk held fixed admits none. So every feasible (chunk, region) pair is
    scored and the best one wins.

    ``halo`` is the per-axis depth of the widest ``map_overlap`` in the chain, or
    None when there is none. Given it, the region is also shaped against the work
    a neighbourhood op will do, not only against the I/O grid: see
    :func:`_joint_search`.

    Returns ``(chunk, region, notes)``. ``notes`` lists anything decided that the
    caller did not ask for, for the caller to report.
    """
    shape = tuple(int(s) for s in shape)
    ndim = len(shape)
    notes = []
    if ndim == 0:
        return (), (), notes

    c_item = max(1, int(_as_np_dtype(chunk_dtype).itemsize))
    r_item = max(1, int(_as_np_dtype(region_dtype or chunk_dtype).itemsize))

    src = None
    if input_chunks is not None and len(input_chunks) == ndim:
        src = tuple(max(1, min(int(c), s)) for c, s in zip(input_chunks, shape))

    # ---- the chunk, when given exactly -------------------------------------
    if chunks is not None:
        chunk = tuple(max(1, min(int(c), s)) for c, s in zip(chunks, shape))
        if len(chunks) != ndim:
            raise ValueError(
                f"chunks {tuple(chunks)} has {len(chunks)} dims but array is {ndim}-D")
    else:
        chunk = None

    # ---- the region, when given exactly ------------------------------------
    if region_shape is not None:
        if len(region_shape) != ndim:
            raise ValueError(
                f"region_shape {tuple(region_shape)} has {len(region_shape)} dims but "
                f"array is {ndim}-D")
        region = tuple(max(1, min(int(r), s)) for r, s in zip(region_shape, shape))
    else:
        region = None

    # ---- both exact: honour both, verify rule 1 ----------------------------
    if chunk is not None and region is not None:
        return chunk, region, notes

    # Nothing said about the chunk at all: PRESERVE THE INPUT GRID. Not a computed
    # shape - the caller always knows their own input chunks, so the output is
    # predictable without reading these docs, and a read-modify-write round trip is
    # lossless by default. The computed alternative depended on dtype, rank and the
    # source grid, hit its own budget only for float32, and could land on shapes no
    # caller could anticipate.
    if chunk is None and chunk_mb is None and src is not None:
        chunk = src
        # Inheriting the grid inherits its problems too, so say so. Neither case is
        # overridden: the caller asked for the input grid by saying nothing, and
        # silently substituting a different one would defeat the whole point of the
        # default. A note names the remedy instead.
        inherited_mb = _prod(chunk) * c_item / (1024 * 1024)
        if inherited_mb > _DEFAULT_REGION_MB:
            notes.append(
                f"the input chunks {tuple(int(c) for c in chunk)} are "
                f"{inherited_mb:.1f} MiB each and were preserved in the output. A "
                f"region holds at least one whole chunk, so this sets the memory "
                f"floor for the write. Pass chunk_size_mb= or chunks= to store "
                f"smaller chunks.")
        elif inherited_mb < _MIN_CHUNK_BYTES / (1024 * 1024):
            notes.append(
                f"the input chunks {tuple(int(c) for c in chunk)} are only "
                f"{inherited_mb * 1024:.1f} KiB each and were preserved in the "
                f"output, which means a great many small files. Pass "
                f"chunk_size_mb= to store larger chunks.")

    chunk_budget = None
    if chunk is None:
        mb = _DEFAULT_CHUNK_MB if chunk_mb is None else float(chunk_mb)
        chunk_budget = max(1, int(mb * 1024 * 1024) // c_item)
    region_budget = None
    if region is None:
        mb = _DEFAULT_REGION_MB if region_mb is None else float(region_mb)
        region_budget = max(1, int(mb * 1024 * 1024) // r_item)

    # Both now fixed (input grid preserved and an exact region given).
    if chunk is not None and region is not None:
        return chunk, region, notes

    # ---- chunk exact, region by budget -------------------------------------
    if chunk is not None:
        region = _region_for_chunk(shape, chunk, src, region_budget, notes,
                                   r_item, halo=halo)
        return chunk, region, notes

    # ---- region exact, chunk by budget -------------------------------------
    if region is not None:
        chunk = _chunk_for_region(shape, region, src, chunk_budget, notes)
        return chunk, region, notes

    # ---- both by budget: the joint search ----------------------------------
    return _joint_search(shape, src, chunk_budget, region_budget, notes,
                         c_item=c_item, r_item=r_item, halo=halo)


def _prod(values):
    out = 1
    for v in values:
        out *= int(v)
    return out


def _region_for_chunk(shape, chunk, src, region_budget, notes, r_item, halo=None):
    """Smallest region that is a whole multiple of ``chunk`` and fits the budget.

    A region is at minimum ONE chunk. When the chunk alone already exceeds the
    region budget, the REGION WINS and goes over budget, loudly: rule 1 is
    correctness and the budget is only a target. Memory boundedness is preserved
    in the sense that matters, since peak still does not scale with array size.
    """
    ndim = len(shape)
    region = [min(int(c), s) for c, s in zip(chunk, shape)]
    if _prod(region) > region_budget:
        over = _prod(region) * r_item / (1024 * 1024)
        notes.append(
            f"one output chunk {tuple(int(c) for c in chunk)} is {over:.2f} MiB, "
            f"larger than the region budget. The region must hold at least one whole "
            f"chunk (concurrent writers must not share one), so the budget is "
            f"exceeded. Pass smaller chunks= to lower peak memory.")
        return tuple(region)

    # Grow uniformly first (preserves the chunk's proportions), then spend the
    # remainder on the inner axes, which are contiguous in C order.
    scale = 1
    while True:
        nxt = scale + 1
        trial = [min(int(c) * nxt, s) for c, s in zip(chunk, shape)]
        if trial == region or _prod(trial) > region_budget:
            break
        region, scale = trial, nxt
    grew = True
    while grew:
        grew = False
        # Without a halo, spend the remainder on the INNER axes, last first: they
        # are contiguous in C order, so a read there is one sequential run.
        #
        # With a halo, spend it where it removes the most work instead. A halo op
        # processes the region plus `depth` on each side, so the relative cost of
        # an axis is (r + 2d)/r -- worst where r is smallest. Pouring the budget
        # into x while an 8-deep axis stays 8 deep is what made the default
        # gaussian 2.2x slower than the same budget spent differently.
        order = range(ndim - 1, -1, -1)
        if halo is not None:
            order = sorted(
                (a for a in range(ndim) if region[a] < shape[a]),
                key=lambda a: ((region[a] + 2 * halo[a]) / region[a], -a),
                reverse=True,
            )
        for axis in order:
            if region[axis] >= shape[axis]:
                continue
            nxt = min(shape[axis], region[axis] + int(chunk[axis]))
            if shape[axis] - nxt < int(chunk[axis]):
                nxt = shape[axis]
            trial = list(region)
            trial[axis] = nxt
            if _prod(trial) <= region_budget:
                region, grew = trial, True
                break
    return tuple(int(v) for v in region)


def _chunk_for_region(shape, region, src, chunk_budget, notes):
    """Largest chunk within budget that divides ``region`` on every axis.

    The region is exact here, so the chunk is the side that bends. Divisors of the
    region are the only feasible sizes (rule 1), and among those the ones that also
    tile the source grid come first.
    """
    ndim = len(shape)
    floor = max(int(chunk_budget * _CHUNK_FLOOR_FRACTION), 1)

    def divisors(r):
        return sorted(d for d in range(1, int(r) + 1) if int(r) % d == 0)

    per_axis = [divisors(r) for r in region]
    best = None
    for cand in _iter_combos(per_axis, chunk_budget):
        size = _prod(cand)
        if size < floor and best is not None:
            continue
        aligned = 0
        if src:
            aligned = sum(1 for c, s in zip(cand, src) if s and (c % s == 0 or s % c == 0))
        tiles = sum(1 for c, s in zip(cand, shape) if s % c == 0)
        score = (aligned, tiles, size)
        if best is None or score > best[0]:
            best = (score, cand)
    if best is None:
        # Every divisor combination overshoots: fall back to the region itself
        # capped per axis, which trivially divides it.
        chunk = tuple(int(r) for r in region)
        notes.append(
            f"no chunk within the budget divides the requested region "
            f"{tuple(int(r) for r in region)}; using the region as the chunk.")
        return chunk
    return tuple(int(c) for c in best[1])


def _iter_combos(per_axis, budget, prefix=(), running=1):
    """Cartesian product of per-axis candidates, pruned by the byte budget.

    Pruning on the RUNNING product is what keeps the joint search tractable: a
    partial chunk already over budget cannot be rescued by later axes, since every
    candidate is at least 1. Measured ~10k combinations for a typical (8,256,256)
    source and ~50k for a coprime one, which is milliseconds.
    """
    if not per_axis:
        yield prefix
        return
    head, rest = per_axis[0], per_axis[1:]
    for c in head:
        nxt = running * int(c)
        if nxt > budget:
            break          # candidates are sorted, so every later one is worse too
        yield from _iter_combos(rest, budget, prefix + (int(c),), nxt)


def _joint_search(shape, src, chunk_budget, region_budget, notes, c_item=1,
                  r_item=1, halo=None):
    """Score every feasible (chunk, region) pair and return the best.

    Candidate chunk sizes per axis are the divisors and multiples of the source
    chunk on that axis, which are exactly the sizes that tile the source grid.
    Region candidates are whole multiples of the chosen chunk, so rule 1 holds by
    construction and never has to be checked.

    The objective is lexicographic, first difference deciding:

    1. region tiles the input chunks   (no read amplification)
    2. chunk tiles the array           (no ragged final chunk)
    3. short outer axes spanned whole  (kills an expensive halo)
    4. budget fidelity                 (by RATIO, see below)
    5. halo amplification              (only when the chain HAS a halo op)
    6. isotropy                        (see below)
    7. smaller chunk                   (memory is what must not slip)

    Halo amplification sits below fidelity deliberately. A halo op processes the
    region PLUS ``depth`` on every side, so two regions of identical size can
    cost very different amounts of work: at depth 8, an 8-deep axis triples while
    a 1024-wide one grows 1.6%. It is scored only when ``halo`` is given, so a
    chain with no neighbourhood op is completely unaffected.

    It is a TIEBREAK, not a minimisation. Minimising it outright picks the
    roundest region, and measured that is slower: a pure (128,128,128) cube has
    lower amplification (1.42x) than (32,256,256) (1.69x) and still ran 8.70s
    against 7.57s, because the cube scatters its reads across 512 source chunks
    while the flatter region stays contiguous in x. Ranking amplification after
    the alignment and budget terms keeps read contiguity winning where it
    matters and only breaks ties with it.

    Fidelity is |log(actual/target)| summed over both grids, so a 2x miss counts
    the same in either direction. Ratio and not difference: from a 63-voxel target,
    1 is a 63x error while 127 is only 2x, though subtraction calls them equal.
    That exact bug produced a (65, 1, 1) chunk.

    Isotropy is the spread of the SPATIAL axes in log space. Without it the first
    four terms are all counts and all ties, so the search settles on whichever
    extreme shape it reaches first: at a 1 MiB budget over (8,256,256) storage it
    picked (1, 256, 1024), a chunk one voxel deep and a full row wide. Same bytes,
    same alignment, but a Z-oriented read or a 3D halo then pulls a whole row per
    step. Leading axes are excluded, since a singleton t/c is correct, not skew.
    """
    ndim = len(shape)
    if src is None:
        # No source grid to align to (a TensorStore source, or a transform that
        # rewrote the index space), so there is nothing to search: size the chunk
        # from the budget alone and build the region from it. The budget is passed
        # through in the REAL dtype - hardcoding uint8 here made a 1 MiB request
        # come back as a 3.93 MiB float32 chunk.
        chunk = _default_chunks(shape, np.dtype(f"u{c_item}") if c_item in (1, 2, 4, 8)
                                else np.dtype(np.uint8),
                                target_mb=chunk_budget * c_item / (1024 * 1024))
        chunk = tuple(min(int(c), s) for c, s in zip(chunk, shape))
        return chunk, _region_for_chunk(shape, chunk, None, region_budget, notes,
                                        r_item, halo=halo), notes

    floor = max(int(chunk_budget * _CHUNK_FLOOR_FRACTION), 1)
    per_axis = [_axis_candidates(ic, s) for ic, s in zip(src, shape)]
    # Only the trailing (spatial) axes are judged for isotropy, matching
    # _default_chunks: a singleton on a leading t/c axis is the right answer there,
    # not a skew to be penalised.
    spatial = list(range(max(0, ndim - 3), ndim))

    best = None
    for chunk in _iter_combos(per_axis, chunk_budget):
        csize = _prod(chunk)
        if csize < floor:
            continue
        region = _grow_region(shape, chunk, region_budget, halo=halo)
        rsize = _prod(region)

        r_aligned = sum(1 for r, s in zip(region, src) if s and r % s == 0)
        c_tiles = sum(1 for c, s in zip(chunk, shape) if s % c == 0)
        short_whole = sum(1 for a in range(ndim)
                          if shape[a] <= _SHORT_AXIS and region[a] >= shape[a])
        # QUANTISED to whole steps of 2x, so that two shapes whose sizes differ
        # only slightly count as equally good and the later terms actually get to
        # decide. Left continuous, fidelity is a float that essentially never ties
        # and isotropy below would be dead code.
        fidelity = round((abs(math.log(max(csize, 1) / chunk_budget))
                          + abs(math.log(max(rsize, 1) / region_budget)))
                         / math.log(2))
        # Spread of the spatial axes in log space. An axis is compared against its
        # own extent, not against the other axes' raw sizes: a chunk of 16 on a
        # 16-long axis is complete, while 16 on a 1024-long one is a thin slab, and
        # measuring raw sizes calls those two identical. That mistake let
        # (16, 16, 1024) score as "isotropic" because the 1024 axis was excluded for
        # being complete and the two 16s then matched each other exactly.
        cover = [math.log(max(1, chunk[a]) / shape[a]) for a in spatial]
        skew = (max(cover) - min(cover)) if len(cover) > 1 else 0.0
        # Halo cost, quantised like fidelity and to the same 2x scale, so only a
        # MEANINGFUL difference in the work a neighbourhood op does can outrank
        # isotropy. Zero (and therefore inert) when the chain has no halo op.
        amp = 0.0
        if halo is not None:
            amp = round(math.log(_halo_amplification(region, shape, halo))
                        / math.log(2), 1)
        score = (r_aligned, c_tiles, short_whole, -fidelity, -amp, -skew, -csize)
        if best is None or score > best[0]:
            best = (score, tuple(chunk), region)

    if best is None:
        # Nothing cleared the floor (a tiny budget against a coarse source grid).
        # Drop the floor rather than return nothing: correctness first.
        for chunk in _iter_combos(per_axis, chunk_budget):
            region = _grow_region(shape, chunk, region_budget, halo=halo)
            score = (_prod(chunk),)
            if best is None or score > best[0]:
                best = (score, tuple(chunk), region)
    if best is None:
        chunk = tuple(min(int(c), s) for c, s in zip(src, shape))
        return chunk, _grow_region(shape, chunk, region_budget, halo=halo), notes

    _, chunk, region = best
    # Report a budget that could not be met. The path that HONOURS the request used
    # to print nothing while the path that missed it stayed silent, which is
    # backwards: a chunk that came out 4x under the requested size is exactly what
    # the caller needs told.
    for label, got, want, remedy in (
        ("chunk", _prod(chunk), chunk_budget, "chunks="),
        ("region", _prod(region), region_budget, "region_shape="),
    ):
        ratio = got / want if want else 1.0
        if ratio < 1.0 / (1.0 + _BUDGET_TOLERANCE) or ratio > 1.0 + _BUDGET_TOLERANCE:
            notes.append(
                f"the {label} came out {ratio:.2f}x its requested size (aligning to "
                f"the source grid {tuple(src)} quantises the available sizes). Pass "
                f"{remedy} to get an exact shape.")
    if any(s and r % s for r, s in zip(region, src)):
        notes.append(
            f"no region within the budget tiles the source grid {tuple(src)}; "
            f"reads will straddle stored chunks. Pass a larger region_size_mb, or "
            f"chunks=/region_shape= matching the source grid, to avoid this.")
    return chunk, region, notes


def _grow_region(shape, chunk, region_budget, halo=None):
    """Largest whole-chunk region within ``region_budget``.

    Uniform scaling first so the chunk's proportions survive (a singleton t/c says
    "this axis is thin", and the region must not undo that), then the leftover goes
    to the inner axes, last first: they are contiguous in C order, so a read there
    is one sequential run and their halo is the cheapest per voxel. An axis that is
    already whole drops out and hands its share to the others, which is how a short
    outer axis gets completed without a special case.

    Given ``halo``, the leftover goes instead to whichever axis carries the worst
    halo ratio ``(r + 2d)/r``, which is the thinnest one. See
    :func:`_region_for_chunk`, which uses the identical rule.
    """
    ndim = len(shape)
    region = [min(int(c), s) for c, s in zip(chunk, shape)]
    scale = 1
    while True:
        nxt = scale + 1
        trial = [min(int(c) * nxt, s) for c, s in zip(chunk, shape)]
        if trial == region or _prod(trial) > region_budget:
            break
        region, scale = trial, nxt
    grew = True
    while grew:
        grew = False
        order = range(ndim - 1, -1, -1)
        if halo is not None:
            order = sorted(
                (a for a in range(ndim) if region[a] < shape[a]),
                key=lambda a: ((region[a] + 2 * halo[a]) / region[a], -a),
                reverse=True,
            )
        for axis in order:
            if region[axis] >= shape[axis]:
                continue
            nxt = min(shape[axis], region[axis] + int(chunk[axis]))
            if shape[axis] - nxt < int(chunk[axis]):
                nxt = shape[axis]
            trial = list(region)
            trial[axis] = nxt
            if _prod(trial) <= region_budget:
                region, grew = trial, True
                break
    return tuple(int(v) for v in region)


def _snap_to_input_chunks(chunk, shape, input_chunks, max_elements):
    """Round each axis of ``chunk`` to a multiple (or exact divisor) of the input chunk.

    The output grid is what every read has to be expressed in, and
    ``_compute_region_shape`` aligns a region to the LCM of the two grids. If the two
    are coprime that LCM explodes - a (101,101,101) output against (8,256,256) input
    needs a 515 GB region before it lines up - so alignment is silently abandoned and
    every read decodes partial chunks on both sides.

    Snapping the output grid to the input one keeps the LCM at the input chunk itself,
    which makes each read a whole number of stored chunks: no partial decode, less
    transient memory, and less CPU. Axes are only ever made SMALLER than the budget
    allows, so the chunk never grows past ``max_elements``.
    """
    snapped = list(chunk)
    for axis, (c, ic, s) in enumerate(zip(chunk, input_chunks, shape)):
        ic = max(1, int(ic))
        if ic >= s:
            # One stored chunk spans the whole axis, so any output size reads that
            # same single chunk: alignment is already satisfied and snapping would
            # only shrink the chunk for nothing. Leave the budgeted value alone.
            continue
        if c >= ic:
            snapped[axis] = min(s, (c // ic) * ic)      # largest multiple that fits
        else:
            # Smaller than one input chunk: prefer a DIVISOR of it, so a stored
            # chunk still splits into a whole number of output chunks. Only accept
            # a divisor that is not a big step down from the budgeted size;
            # otherwise keep the budgeted value rather than waste the budget.
            best = max((d for d in range(c, 0, -1) if ic % d == 0), default=0)
            snapped[axis] = best if best * 2 >= c else c
    # The snap can only shrink or hold each axis, but guard the product anyway.
    while int(np.prod(snapped)) > max_elements:
        widest = int(np.argmax(snapped))
        if snapped[widest] <= 1:
            break
        ic = max(1, int(input_chunks[widest]))
        snapped[widest] = max(1, snapped[widest] - ic if snapped[widest] > ic
                              else snapped[widest] // 2)
    return tuple(int(v) for v in snapped)


def _default_chunks(shape, dtype, target_mb=_DEFAULT_CHUNK_MB, input_chunks=None):
    """Chunk shape of about ``target_mb`` for an array of ``shape`` and ``dtype``.

    Replaces a hardcoded ``(256,) * ndim``, which ignored both the dtype and the
    number of dimensions and so scaled catastrophically with rank: 67 MB for 3D
    float32, 8.6 GB for 4D uint16, 4.4 TB for 5D float32. Because a single chunk is
    the smallest region the writer can process, that default set the memory floor
    for every unchunked write regardless of ``region_size_mb``.

    The algorithm follows ome_zarr_pyramid's ``autocompute_chunk_shape``: size the last
    up-to-3 axes toward a cube and leave every leading axis at 1, with power-of-two
    sides (see the body for why, and how the non-cube budgets are split). ozp
    selects the spatial axes by NAME from an ``axes`` string; dyna is deliberately
    axis-agnostic (a plain array layer, with OME semantics living in ozp), so the
    trailing axes are taken as spatial, which is the same assumption its
    ``region_shape`` expansion already makes.

    Isotropy is the point. Filling trailing axes one at a time gives shapes like
    ``(1, 256, 1024)``: the same bytes, but a Z-oriented read or a 3D halo filter
    then pulls a full XY plane for every step along Z.
    """
    shape = tuple(int(s) for s in shape)
    if not shape:
        return ()
    itemsize = max(1, int(_as_np_dtype(dtype).itemsize))
    max_elements = max(1, int(target_mb * 1024 * 1024) // itemsize)

    chunk = [1] * len(shape)
    spatial = list(range(max(0, len(shape) - 3), len(shape)))   # last <=3 axes

    # POWER-OF-TWO sides. Stored data is almost always chunked in powers of two, so a
    # power-of-two chunk divides or is a multiple of the grids it meets (aligned reads,
    # regions), survives repeated 2x downsampling, and - every numeric itemsize being a
    # power of two - fills a power-of-two byte budget EXACTLY (float64 used to get
    # (50, 50, 50) = 0.95 MiB, uint8 an unalignable (101, 101, 101)).
    #
    # Grow by doubling the SMALLEST spatial axis, ties going to the INNERMOST: a cube
    # when the element budget is 2**(3k), otherwise the spare factors of 2 land on x,
    # then y - contiguous in memory, and measured to read faster than a round shape
    # (see _joint_search). So 1 MiB is (64,64,64) float32, (64,128,128) uint8,
    # (32,64,64) float64. An axis shorter than its share is taken WHOLE (spanning an
    # axis beats a power of two on it) and its slack goes to the others; the budget is
    # a ceiling, so a chunk may land under it but never over.
    done = set()
    while len(done) < len(spatial):
        i = min((a for a in spatial if a not in done), key=lambda a: (chunk[a], -a))
        grown = min(chunk[i] * 2, shape[i])
        if grown == chunk[i]:
            done.add(i)
            continue
        if int(np.prod([grown if a == i else chunk[a] for a in spatial])) > max_elements:
            done.add(i)
            continue
        chunk[i] = grown
        if grown == shape[i]:
            done.add(i)

    # Align to the source grid when we know it: an output chunk that is a multiple
    # (or divisor) of the input chunk keeps every read a whole number of stored
    # chunks. Without this the two grids are usually coprime and their LCM is
    # unreachable, so reads decode partial chunks on both sides.
    if input_chunks is not None and len(input_chunks) == len(shape):
        chunk = list(_snap_to_input_chunks(chunk, shape, input_chunks, max_elements))
    return tuple(chunk)


def _as_np_dtype(dt):
    """numpy dtype for ``dt``, which may be a TensorStore dtype.

    ``array.dtype`` on a TensorStore-backed DynamicArray is a ``tensorstore.dtype``,
    which ``np.dtype()`` refuses outright. Every helper here sizes things in bytes,
    so they all have to go through this rather than np.dtype directly.
    """
    if dt is None:
        return None
    try:
        return np.dtype(dt)
    except TypeError:
        from .utils import parse_dtype
        return parse_dtype(dt)[0]


def _widest_dtype(*dtypes):
    """The dtype with the largest itemsize among ``dtypes`` (unknowns ignored)."""
    widest, widest_size = None, -1
    for dt in dtypes:
        if dt is None:
            continue
        try:
            size = int(_as_np_dtype(dt).itemsize)
        except Exception:
            continue
        if size > widest_size:
            widest, widest_size = dt, size
    return widest


#: Attributes a Transform may use to hold its upstream DynamicArray(s). Every
#: Transform subclass uses exactly one of these (``array`` for the unary structural /
#: neighborhood / reduce / scan ops, ``arrays`` for concat and stack, ``operands`` for
#: pointwise map_blocks), so walking them reaches the whole chain without each op
#: having to declare anything.
_TRANSFORM_UPSTREAM_ATTRS = ("array", "arrays", "operands")


def _chain_widest_dtype(array, _seen=None, _depth=0):
    """Widest dtype appearing ANYWHERE in ``array``'s lazy chain.

    Region sizing has to budget for what the pipeline actually holds while a region
    is in flight, which is the widest intermediate, not the final result. Reading
    ``array.dtype`` alone under-counts badly whenever a chain narrows: a uint8 mask
    derived from a float32 volume looks like 1 byte per element while every region
    really carries 4, so the writer used ~4x the requested ``region_size_mb`` and
    peak memory tracked the array size instead of the budget.

    The walk is defensive on purpose - an unknown transform shape must degrade to
    "no extra information" rather than raise inside a write.
    """
    if array is None or _depth > 64:
        return None
    if _seen is None:
        _seen = set()
    if id(array) in _seen:
        return None
    _seen.add(id(array))

    widest = getattr(array, "dtype", None)
    transform = getattr(array, "_transform", None)
    if transform is None:
        return widest

    widest = _widest_dtype(widest, getattr(transform, "dtype", None))
    for attr in _TRANSFORM_UPSTREAM_ATTRS:
        upstream = getattr(transform, attr, None)
        if upstream is None:
            continue
        candidates = upstream if isinstance(upstream, (list, tuple)) else [upstream]
        for item in candidates:
            # operands may hold plain scalars alongside DynamicArrays
            if not hasattr(item, "dtype"):
                continue
            widest = _widest_dtype(widest, _chain_widest_dtype(item, _seen, _depth + 1))
    return widest


def _chain_halo_depth(array, ndim, _seen=None, _level=0):
    """Deepest per-axis halo any ``map_overlap`` in ``array``'s chain asks for.

    A halo op does not read its region, it reads the region PLUS ``depth`` on
    each side, and the cost it pays is that padded block, not the region. So a
    region shaped for I/O alone can be badly shaped for the op running over it:
    at depth 8, a region 8 deep on an axis triples along it while one 1024 wide
    grows 1.6%. Measured, a default `(8,256,1024)` region handed scipy 2.06x its
    own volume and the write took 16.8s, where `(32,256,256)` at the SAME 8 MiB
    budget took 7.6s.

    Returns a per-axis tuple, or None when the chain has no halo op (the common
    case, and the one where this must cost nothing).

    Walks the same upstream attributes as :func:`_chain_widest_dtype` and is
    defensive for the same reason: an unknown transform must degrade to "no
    information" rather than raise inside a write.
    """
    if array is None or _level > 64:
        return None
    if _seen is None:
        _seen = set()
    if id(array) in _seen:
        return None
    _seen.add(id(array))

    transform = getattr(array, "_transform", None)
    if transform is None:
        return None

    best = None
    depth = getattr(transform, "depth", None)
    if depth is not None:
        try:
            cand = tuple(int(d) for d in depth)
            if len(cand) == ndim and all(d >= 0 for d in cand):
                best = cand
        except (TypeError, ValueError):
            pass

    for attr in _TRANSFORM_UPSTREAM_ATTRS:
        upstream = getattr(transform, attr, None)
        if upstream is None:
            continue
        candidates = upstream if isinstance(upstream, (list, tuple)) else [upstream]
        for item in candidates:
            if not hasattr(item, "shape"):
                continue
            sub = _chain_halo_depth(item, ndim, _seen, _level + 1)
            if sub is None:
                continue
            # Several halo ops can stack; the widest per axis is what a region
            # must be shaped against.
            best = sub if best is None else tuple(max(a, b) for a, b in zip(best, sub))
    return best


class RepeatedReductionWarning(UserWarning):
    """A reduction too large to cache is broadcast back over the axes it reduced, so a
    region-wise write re-streams it once per region along those axes. The result is
    still correct; ``.persist()`` on the reduced array computes it once instead."""


def _span_repeated_axes(array, region, shape, chunk, region_mb, widest_dtype):
    """Region that spans, whole, every axis a broadcast reduction is repeated along.

    Returns ``region`` unchanged when it already spans them or nothing is repeated, a
    new region when spanning fits the ``region_mb`` budget (other axes are halved, in
    whole chunks, largest first, to make room), and None when it cannot fit - the
    caller then keeps its region and the repeat is reported instead.
    """
    from .operations.reductions import find_repeated_reductions
    axes = sorted({a for cshape, ax, _ in find_repeated_reductions(array)
                   if cshape == tuple(shape) for a in ax})
    if not axes or all(region[a] >= shape[a] for a in axes):
        return tuple(region)
    itemsize = 1
    if widest_dtype is not None:
        itemsize = max(1, int(_as_np_dtype(widest_dtype).itemsize))
    budget = max(1, int(region_mb * 1024 * 1024) // itemsize)
    r = list(region)
    for a in axes:
        r[a] = shape[a]
    others = [a for a in range(len(shape)) if a not in axes]
    while _prod(r) > budget:
        candidates = [a for a in others if r[a] > chunk[a]]
        if not candidates:
            return None
        a = max(candidates, key=lambda ax: r[ax] / chunk[ax])
        r[a] = max(chunk[a], (r[a] // 2 // chunk[a]) * chunk[a])
    return tuple(r)


def _warn_repeated_reductions(array, region, shape):
    """Warn, with the repeat count, for every broadcast reduction the regions re-read."""
    import warnings
    from .operations.reductions import find_repeated_reductions
    for cshape, axes, desc in find_repeated_reductions(array):
        if cshape == tuple(shape):
            n = 1
            for a in axes:
                n *= -(-shape[a] // region[a])
            if n <= 1:
                continue
            where = (f"along axes {tuple(axes)} and is re-read about {n}x (once per "
                     f"region along those axes)")
        else:
            where = "inside the chain and is re-read once per region that needs it"
        warnings.warn(
            f"{desc} is broadcast back {where}. The result is correct, only slower. "
            f"Call .persist() on the reduced array before combining it (e.g. "
            f"m = x.mean(axis=0, keepdims=True).persist(); x - m) to compute it once.",
            RepeatedReductionWarning, stacklevel=4)


def _halo_amplification(region, shape, depth):
    """Volume the halo op actually processes, per unit of region volume.

    An axis the region spans WHOLE gets no halo on either side (there is nothing
    outside to read), which is why completing a short axis is worth so much.
    """
    grown = 1
    plain = 1
    for r, s, d in zip(region, shape, depth):
        width = r if r >= s else min(s, r + 2 * d)
        grown *= width
        plain *= r
    return grown / plain if plain else 1.0


def _compute_region_shape(input_shape, final_chunks, region_size_mb, dtype=None, input_chunks=None):
    """
    Compute optimal region shape with simple deterministic algorithm.
    
    Algorithm:
    1. Start with a single output chunk
    2. Identify dimensions needing expansion (region < input_shape)
    3. Expand dimensions in reverse order (last â†’ first) until region_size_mb reached
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
    
    # STEP 3: Expand dimensions in reverse order (last â†’ first)
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


class _ZarristaRegion:
    """One ``output[region]`` handle. ``.write(data)`` returns a future-like object so
    the writer thread's ``future.done()`` / ``.result()`` protocol is unchanged.

    zarrista's ``store_array_subset`` is synchronous, so the write has already happened
    by the time this is constructed; ``done()`` is therefore always True. The pipeline's
    backpressure is driven by the queue and the reader threads, which still applies.
    """

    __slots__ = ("_exc",)

    def __init__(self, array, region, data):
        from .backends.zarrista_backend import _call

        self._exc = None
        try:
            # via _call: on a remote (object-store) array the write is async and
            # needs an event loop, which _call supplies; local writes are direct.
            _call(array.store_array_subset, region, data)
        except BaseException as exc:      # surfaced by .result(), like a TS future
            self._exc = exc

    def done(self):
        return True

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc
        return None


class _ZarristaOutput:
    """Minimal ``output[region].write(data)`` facade over a zarrista array."""

    __slots__ = ("_array", "_write_unit")

    def __init__(self, array):
        self._array = array
        self._write_unit = array.write_unit

    @property
    def write_unit(self):
        """Shard when sharded, else chunk - the unit concurrent writers must not share."""
        return self._write_unit

    def __getitem__(self, region):
        outer = self

        class _Slot:
            __slots__ = ()

            def write(self, data):
                import numpy as _np
                # zarrista requires C-contiguous input; a region sliced out of a
                # larger buffer usually is not.
                return _ZarristaRegion(
                    outer._array._array, region, _np.ascontiguousarray(data)
                )

        return _Slot()


def _open_zarrista_output(output_path, shape, chunks, dtype, codecs,
                          zarr_format, shards, storage_options=None):
    """Create the output array through the optional zarrista backend.

    ``shards`` is the resolved shard SHAPE (already coefficients x chunks), or None.
    """
    from .backends.zarrista_backend import create_v2, create_v3

    if zarr_format == 2:
        # write_array rejects shard_coefficients on v2 before reaching here, so
        # shards is always None on this branch.
        array = create_v2(output_path, shape, chunks, dtype, codecs=codecs,
                          storage_options=storage_options)
    else:
        array = create_v3(output_path, shape, chunks, dtype,
                          shard=tuple(shards) if shards else None, codecs=codecs,
                          storage_options=storage_options)
    return _ZarristaOutput(array)


@dataclass
class _OutputSpec:
    """Everything io.write decides about its output, BEFORE anything is written.

    Built by :func:`_resolve_output`, where ALL argument validation happens, and consumed
    by :func:`_open_output` and by every write strategy (the region pipeline and the
    staged flatten/reshape/scan writers). Keeping it in one place is the point: the
    staged writers used to branch off above the resolution and so silently lost every
    setting resolved after the branch (format, codecs, shards, backend, overwrite and
    all validation). Internal; nothing here is user-facing.
    """
    path: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    dtype_v2: str
    dtype_v3: str
    chunks: Tuple[int, ...]
    shards: Optional[Tuple[int, ...]]
    codecs: Codecs
    zarr_format: int
    backend: str
    storage_options: Optional[dict]
    overwrite: bool
    remote: bool
    region_mb: float
    solved_region: Tuple[int, ...]
    notes: List[str] = field(default_factory=list)


def _resolve_output(array, output_path, *, max_workers, num_readers, chunks, chunk_size_mb,
                    shard_coefficients, dtype, compressor, zarr_format, region_size_mb,
                    gc_interval, device, region_shape, memory_budget_mb, overwrite,
                    backend, storage_options) -> _OutputSpec:
    """Validate every io.write argument and resolve the output's settings.

    Raises before anything touches storage, so a rejected write leaves nothing behind.
    Defaults, each applied only when the caller did not say:
      - format: the INPUT's zarr format, else v3 (sources without one: numpy, TIFF,
        creation ops, ...);
      - codecs / shards: the input's, else blosc-lz4 / unsharded;
      - chunks / region: see _solve_grids (input chunks kept by default).
    """
    # --- contradictory pairs --------------------------------------------------
    # `chunks` (exact) and `chunk_size_mb` (budget) are alternatives, as are
    # `region_shape` and `region_size_mb`: passing both is a contradiction, not a
    # preference, so say so rather than silently honouring one.
    if chunks is not None and chunk_size_mb is not None:
        raise ValueError(
            "pass either chunks= (an exact shape) or chunk_size_mb= (a size budget), "
            "not both"
        )
    if region_shape is not None and region_size_mb is not None:
        raise ValueError(
            "pass either region_shape= (an exact shape) or region_size_mb= (a size "
            "budget), not both"
        )
    if chunk_size_mb is not None and float(chunk_size_mb) <= 0:
        raise ValueError(f"chunk_size_mb must be > 0, got {chunk_size_mb}")
    if region_size_mb is not None and float(region_size_mb) <= 0:
        raise ValueError(f"region_size_mb must be > 0, got {region_size_mb}")
    region_mb = _DEFAULT_REGION_MB if region_size_mb is None else float(region_size_mb)

    if backend not in ("tensorstore", "zarrista"):
        raise ValueError(
            f"unknown backend {backend!r}; expected 'tensorstore' (default) or 'zarrista'"
        )
    output_path = _local_path_from_file_url(output_path)
    remote = '://' in str(output_path)
    if backend == "tensorstore":
        if remote:
            # Remote output through TensorStore's own stores (s3/gcs), with the same
            # storage_options names as zarrista. Validated NOW, so an unsupported scheme
            # (plain HTTP is read-only there, az:// has no store) or an unsupported option
            # fails before anything is created. (It used to open the local `file` store
            # with the URL as a PATH.)
            _ts_kvstore(output_path, storage_options, write=True)
        elif storage_options:
            # Same rule as the zarrista guard: remote-store options on a local path.
            from .backends.zarrista_backend import UnsupportedByBackend
            raise UnsupportedByBackend(
                f"storage_options are for remote stores, but the output {str(output_path)!r} "
                f"is a local path. Drop storage_options, or pass a remote URL.")

    # --- argument validation --------------------------------------------------
    # These are BACKEND-INDEPENDENT: a nonsensical value is wrong on every backend,
    # so it is caught here rather than in a backend guard (which would leave the
    # default path unchecked). Found by fuzzing each parameter with degenerate
    # values: every case below previously either wrote silently with the bad value
    # ignored, or reached a library and failed with an unrelated error - in the
    # worst case a Rust panic that left a CORRUPT array on disk.

    # dyna writes zarr v2 or v3. Anything else used to fall through to the v3
    # branch and write SILENTLY (zarr_format=4 produced a v3 store).
    if zarr_format is not None and int(zarr_format) not in (2, 3):
        raise ValueError(f"zarr_format must be 2 or 3, got {zarr_format}.")

    ndim = len(array.shape)

    def _check_axes(name, value):
        """Positive ints, one per axis. Rejects 0, negatives and wrong rank."""
        if value is None:
            return None
        try:
            vals = tuple(int(v) for v in value)
        except TypeError:
            raise ValueError(f"{name} must be a tuple of ints, got {value!r}") from None
        if len(vals) != ndim:
            # Same wording as _solve_grids' rank errors, which this check now pre-empts.
            raise ValueError(f"{name} {vals} has {len(vals)} dims but array is {ndim}-D")
        if any(v < 1 for v in vals):
            raise ValueError(f"{name}={vals} must be positive on every axis")
        return vals

    # chunks/shard bigger than the array are fine (zarr clamps them), but a huge
    # value overflows zarrista's Rust allocator and PANICS mid-write, leaving a
    # corrupt store; cap them at something that cannot.
    # A chunk is allocated as one buffer, so bound it by ELEMENT COUNT, not by a
    # per-axis cap: (10**7)**3 elements overflowed zarrista's Rust allocator, which
    # PANICKED mid-write and left a corrupt store on disk. 2**34 elements is far
    # beyond any sane chunk while still being an obvious "you meant something else".
    _MAX_CHUNK_ELEMENTS = 2 ** 34
    for _name, _val in (("chunks", chunks), ("region_shape", region_shape)):
        vals = _check_axes(_name, _val)
        if vals:
            _n = 1
            for _v in vals:
                _n *= _v
            if _n > _MAX_CHUNK_ELEMENTS:
                raise ValueError(
                    f"{_name}={vals} is {_n} elements, which cannot be allocated as "
                    f"a single buffer (limit {_MAX_CHUNK_ELEMENTS}). Chunks are "
                    f"clamped to the array shape, so pick a realistic size."
                )
    _check_axes("shard_coefficients", shard_coefficients)

    for _name, _val, _min in (
        ("max_workers", max_workers, 1),
        ("num_readers", num_readers, 1),
        ("region_size_mb", region_mb, None),
        ("memory_budget_mb", memory_budget_mb, None),
        ("gc_interval", gc_interval, None),
    ):
        if _val is None:
            continue
        if _min is not None and int(_val) < _min:
            raise ValueError(f"{_name} must be >= {_min}, got {_val}")
        if _min is None and float(_val) <= 0:
            raise ValueError(f"{_name} must be > 0, got {_val}")

    if device is not None:
        dev = str(device)
        if dev != "cpu" and not dev.startswith("cuda"):
            raise ValueError(
                f"device must be 'cpu', 'cuda' or 'cuda:N', got {device!r}"
            )

    # io.write only ever writes Zarr, on every backend. A `.tif` output path would
    # silently produce a ZARR STORE with a misleading name, so this is refused here
    # rather than in a backend guard: it is not a backend limitation.
    if str(output_path).lower().endswith(('.tif', '.tiff')):
        raise ValueError(
            f"io.write writes Zarr, not TIFF, so {str(output_path)!r} would be a zarr "
            f"store with a misleading name. Use a .zarr path, or write the TIFF "
            f"yourself (e.g. tifffile.imwrite(path, array.compute()))."
        )

    # --- dtype and grids ------------------------------------------------------
    # dtype first: the chunk is sized in BYTES, so it needs the dtype that will
    # actually be written (which `dtype=` may have changed). The CHUNK is sized in
    # that stored dtype; the REGION has to be sized in the widest dtype the chain
    # carries, or a narrowing write (a uint8 mask from a float32 volume) budgets at
    # 1 byte per element while each region really holds 4.
    final_dtype = dtype if dtype is not None else array.dtype
    # A neighbourhood op reads its region PLUS a halo on every side, so the region
    # has to be shaped against that work and not only against the I/O grid.
    final_chunks, solved_region, notes = _solve_grids(
        tuple(array.shape), getattr(array, "chunks", None), final_dtype,
        _widest_dtype(_chain_widest_dtype(array), final_dtype),
        chunks=chunks, chunk_mb=chunk_size_mb,
        region_shape=region_shape, region_mb=region_size_mb,
        halo=_chain_halo_depth(array, ndim),
    )
    dtype_obj, dtype_v2, dtype_v3 = parse_dtype(final_dtype)

    # --- format: explicit, else the input's own, else v3 ------------------------
    final_format = (int(zarr_format) if zarr_format is not None
                    else (getattr(array, "zarr_format", None) or 3))

    # --- codecs: explicit, else the input's, else blosc-lz4 --------------------
    if compressor is None:
        if getattr(array, "codecs", None) is not None:
            final_codecs = array.codecs
        else:
            final_codecs = Codecs('blosc', clevel=5, cname='lz4')
    elif isinstance(compressor, Codecs):
        final_codecs = compressor
    else:
        # Assume it's a numcodecs compressor - convert it
        final_codecs = Codecs.from_numcodecs(compressor)

    # blosc cnames are limited by what THIS numcodecs build has compiled in - the
    # same list for v2 and v3, since zarr's v3 BloscCodec also goes through
    # numcodecs. zarr-python rejects an unavailable cname at array creation on both
    # formats, but dyna's TensorStore path does not go through zarr, so
    # `cname='snappy'` wrote a store that then raised "blosc decompression: -5" on
    # read. Check it here so every backend and format behaves like zarr-python.
    _requested = compressor if isinstance(compressor, Codecs) else None
    if _requested is not None:
        _name = getattr(_requested, "compressor", None)
        if _name == "blosc":
            _cname = getattr(_requested, "cname", "lz4")
            try:
                import numcodecs.blosc as _blosc
                _available = set(_blosc.list_compressors())
            except Exception:
                _available = None
            if _available is not None and _cname not in _available:
                raise ValueError(
                    f"blosc cname={_cname!r} is not available in this blosc build "
                    f"(have: {', '.join(sorted(_available))}). Writing it produces a "
                    f"store that cannot be read back."
                )
        elif _name in ("lz4", "bz2") and final_format == 3:
            raise ValueError(
                f"compressor {_name!r} has no zarr v3 codec. Use zarr_format=2, or "
                f"blosc with cname={_name!r} (lz4), or zstd/gzip."
            )

    if backend == "zarrista":
        # ONE guard, before the store is touched: it sees the RESOLVED format and
        # codecs, so an unsupported combination fails here rather than part-way
        # through creating the output.
        from .backends.zarrista_backend import check_supported as _zst_check

        _zst_check(output=output_path, zarr_format=final_format,
                   codecs=final_codecs, storage_options=storage_options)

    # --- shards: explicit (v3 only), else the input's when they still fit ------
    # Sharding exists only in zarr v3. Asking for it on v2 used to be dropped in
    # silence (on BOTH backends), so the caller got an unsharded array and no hint
    # that the argument had been discarded.
    if shard_coefficients is not None and final_format != 3:
        raise ValueError(
            f"shard_coefficients={tuple(shard_coefficients)} requires zarr_format=3; "
            f"zarr v{final_format} has no sharding. Pass zarr_format=3, or drop "
            f"shard_coefficients."
        )
    final_shards = None
    if shard_coefficients is not None:
        final_shards = tuple(int(c) * int(k) for c, k in zip(shard_coefficients, final_chunks))
    elif final_format == 3 and getattr(array, "shards", None) is not None:
        inherited = tuple(int(s) for s in array.shards)
        # A shard must hold whole chunks. The input's shards always did for the input's
        # chunks; with other output chunks they may not, and then they are dropped
        # (with a note) rather than producing a store TensorStore/zarr would reject.
        if len(inherited) == ndim and all(s % c == 0 for s, c in zip(inherited, final_chunks)):
            final_shards = inherited
        else:
            notes.append(
                f"the input's shards {inherited} do not hold whole output chunks "
                f"{tuple(final_chunks)}, so the output is written unsharded. Pass "
                f"shard_coefficients= to shard it.")

    # --- region_shape: whole output chunks per axis -----------------------------
    if region_shape is not None:
        misaligned = [
            (ax, int(r), c) for ax, (r, c) in enumerate(zip(region_shape, final_chunks))
            if int(r) % c != 0
        ]
        if misaligned:
            # Deliberately NOT rounded for the caller. region_shape is the knob that
            # bounds peak RAM, and rounding up can multiply it (e.g. (4,17,17) ->
            # (4,32,32) is 3.5x), while rounding down silently destroys the alignment
            # the caller was usually trying to achieve. Both nearest valid shapes are
            # offered instead, so the fix is a copy-paste rather than arithmetic.
            up = tuple(-(-int(r) // c) * c for r, c in zip(region_shape, final_chunks))
            down = tuple(max(c, (int(r) // c) * c)
                         for r, c in zip(region_shape, final_chunks))
            axes = ", ".join(
                f"axis {ax}: {r} is not a multiple of {c}" for ax, r, c in misaligned
            )
            raise ValueError(
                f"region_shape {tuple(int(r) for r in region_shape)} must be a per-axis "
                f"multiple of the output chunks {tuple(final_chunks)}, so that each region "
                f"writes whole chunks and concurrent region writes cannot race on a shared "
                f"chunk ({axes}). Nearest valid shapes: {up} (rounding up) or {down} "
                f"(rounding down); note the larger one raises peak memory."
            )

    return _OutputSpec(
        path=str(output_path), shape=tuple(int(s) for s in array.shape),
        dtype=dtype_obj, dtype_v2=dtype_v2, dtype_v3=dtype_v3,
        chunks=tuple(int(c) for c in final_chunks), shards=final_shards,
        codecs=final_codecs, zarr_format=final_format, backend=backend,
        storage_options=storage_options, overwrite=bool(overwrite), remote=remote,
        region_mb=region_mb, solved_region=tuple(int(r) for r in solved_region),
        notes=list(notes),
    )


def _ts_output_spec(spec: _OutputSpec) -> dict:
    """TensorStore create-spec for a resolved output: the local ``file`` store, or a
    remote s3/gcs store built from the URL and storage_options (see _ts_kvstore)."""
    kvstore = (_ts_kvstore(spec.path, spec.storage_options, write=True) if spec.remote
               else {'driver': 'file', 'path': str(Path(spec.path).absolute())})
    ts_spec = {
        'driver': 'zarr3' if spec.zarr_format == 3 else 'zarr',
        'kvstore': kvstore,
    }
    if spec.zarr_format == 3:
        codecs = spec.codecs.to_v3_config(spec.dtype)
        if spec.shards is not None:
            # Shards are the outer chunks (files), chunks are the inner chunks.
            grid_shape = list(spec.shards)
            codecs = [{
                'name': 'sharding_indexed',
                'configuration': {
                    'chunk_shape': list(spec.chunks),
                    'codecs': codecs,
                    'index_codecs': [
                        {'name': 'bytes', 'configuration': {'endian': 'little'}},
                        {'name': 'crc32c'},
                    ],
                },
            }]
        else:
            grid_shape = list(spec.chunks)
        ts_spec['metadata'] = {
            'shape': list(spec.shape),
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': grid_shape}},
            'chunk_key_encoding': {'name': 'default', 'configuration': {'separator': '/'}},
            'codecs': codecs,
            'data_type': spec.dtype_v3,
        }
    else:
        ts_spec['metadata'] = {
            'shape': list(spec.shape),
            'chunks': list(spec.chunks),
            'dtype': spec.dtype_v2,
            'dimension_separator': '/',
            'compressor': spec.codecs.to_v2_config(),
        }
    return ts_spec


def _open_output(spec: _OutputSpec):
    """Create the output array for a resolved spec and return its write handle.

    The handle supports ``out[region].write(data)`` (a future, for the concurrent
    region pipeline) on every backend. ``overwrite`` is applied here, once, for every
    write strategy.
    """
    if spec.overwrite and not spec.remote:
        # Only remove something that already IS a zarr store: a mistyped output_path
        # must never delete an unrelated tree. _detect_zarr_format_local checks for
        # .zarray/.zgroup/zarr.json.
        _out = Path(spec.path)
        if _out.exists():
            if not _out.is_dir():
                raise ValueError(
                    f"overwrite=True: {spec.path} exists and is not a directory")
            if _detect_zarr_format_local(_out) is None and any(_out.iterdir()):
                raise ValueError(
                    f"overwrite=True refused: {spec.path} is a non-empty directory that "
                    f"does not look like a zarr store (no .zarray/.zgroup/zarr.json). "
                    f"Delete it yourself if you really mean to replace it.")
            shutil.rmtree(_out)

    if not spec.remote:
        # A URL is not a directory to create; the object store makes keys on write.
        Path(spec.path).mkdir(parents=True, exist_ok=True)

    if spec.backend == "zarrista":
        # The writers only ever do `output[region].write(data)` and wait on the returned
        # future, so the optional backend is swapped in behind that one interface
        # (_ZarristaOutput) and the write machinery stays exactly as it is.
        return _open_zarrista_output(
            spec.path, spec.shape, spec.chunks, spec.dtype, spec.codecs,
            spec.zarr_format, spec.shards, spec.storage_options,
        )
    import tensorstore as ts
    return ts.open(_ts_output_spec(spec), create=True).result()


class _SyncSink:
    """Blocking ``sink[key] = block`` over an :func:`_open_output` handle.

    The staged writers (flatten/reshape/scan) write one block at a time, strictly in
    order, through this - so they get every resolved output setting and every backend,
    local or remote, from the same opener the region pipeline uses. Serial writes also
    keep them clear of zarrista's one data-loss bug, which needs CONCURRENT writes to a
    shared chunk/shard (reports/zarrista_bugs).
    """

    def __init__(self, handle, spec: _OutputSpec):
        self._handle = handle
        self.shape = spec.shape
        self.chunks = spec.chunks
        self.dtype = spec.dtype

    def __setitem__(self, key, value):
        data = np.ascontiguousarray(value, dtype=self.dtype)
        if isinstance(self._handle, _ZarristaOutput):
            self._handle._array[key] = data         # normalizes the key, handles remote
        else:
            self._handle[key].write(data).result()


#: The outermost transforms io.write hands to a STAGED writer instead of the region
#: pipeline, which cannot produce them memory-bounded.
_STAGED_WRITES = {"FlattenTransform": "flatten", "ReshapeTransform": "reshape",
                  "ScanTransform": "scan"}


def _write_staged(kind, array, spec: _OutputSpec, *, max_workers, device,
                  region_shape, memory_budget_mb, num_readers):
    """Write ``array`` (an outermost reshape/flatten/scan) with its staged writer.

    - flatten/reshape: a C-order flat re-index conflicts with nd chunk layout, so they go
      through a disk-staged rechunk to a flat-contiguous layout (read-once source) in a
      LOCAL temp directory, then a relabel/reindex into the output.
    - scan: a bounded-carry stream along the scan axis (no staging).

    They write into the same output the region pipeline would open (format, codecs,
    shards, backend, overwrite, remote stores), under the same execution ``device``,
    with ``region_size_mb`` as their per-worker memory budget.
    """
    # These knobs shape the CONCURRENT REGION pipeline; a staged write has no such
    # regions, so honouring them is impossible - refuse rather than silently ignore.
    ignored = [name for name, value in (("region_shape", region_shape),
                                        ("memory_budget_mb", memory_budget_mb),
                                        ("num_readers", num_readers)) if value is not None]
    if ignored:
        raise ValueError(
            f"{', '.join(ignored)} {'does' if len(ignored) == 1 else 'do'} not apply to a "
            f"{kind} write: an outermost {kind} is written by a staged writer, not by the "
            f"region pipeline. Its memory budget is region_size_mb (per worker, x "
            f"max_workers). Drop {', '.join(ignored)}.")

    from .rechunk import check_staging_space, staging_bytes, flatten_write, reshape_write
    from .operations.scan import scan_write
    from .operations._backend import device_context

    transform = array._transform
    source = transform.array
    max_mem = int(spec.region_mb * 1024 * 1024)
    # Checked BEFORE the output exists, so a refusal leaves nothing behind.
    check_staging_space(staging_bytes(kind, source, spec.dtype, max_mem), kind)

    sink = _SyncSink(_open_output(spec), spec)
    common = dict(output_chunks=spec.chunks, max_mem=max_mem, dtype=spec.dtype,
                  zarr_format=spec.zarr_format)
    with device_context(device):
        if kind == "flatten":
            flatten_write(source, sink, max_workers=max_workers, **common)
        elif kind == "reshape":
            reshape_write(source, spec.shape, sink, max_workers=max_workers, **common)
        else:
            scan_write(source, transform.op, transform.axis, sink, **common)
    return spec.path


def write_array(
    array: 'DynamicArray',
    output_path: str,
    max_workers: int = 4,
    num_readers: Optional[int] = None,
    chunks: Optional[Tuple[int, ...]] = None,
    chunk_size_mb: Optional[float] = None,
    shard_coefficients: Optional[Tuple[int, ...]] = None,
    dtype: Optional[Any] = None,
    compressor: Optional[Union[Codecs, Any]] = None,
    zarr_format: Optional[int] = None,
    region_size_mb: Optional[float] = None,
    gc_interval: float = 15.0,
    early_quarter_timeout: Optional[float] = None,
    early_tenth_timeout: Optional[float] = None,
    device: Optional[str] = None,
    region_shape: Optional[Tuple[int, ...]] = None,
    memory_budget_mb: Optional[float] = None,
    overwrite: bool = False,
    backend: str = "tensorstore",
    storage_options: Optional[dict] = None,
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

        peak RAM ~= FIXED + k * max_workers * region_bytes, with k ~= 2.5 (a region
        in flight is held as source AND as result, plus copies) and FIXED ~= 200 MB
        per write. Independent of the ARRAY size, which is the guarantee that
        matters; see "What peak RAM actually is" in the README.

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
        writes. Peak RAM ~= FIXED + k * max_workers * region_bytes (k ~= 2.5).
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
        EXACT output chunk shape, honoured verbatim. For v3 with sharding, this is
        the inner chunk shape. **Defaults to the INPUT chunks**, so a read-modify-
        write round trip preserves the grid and the output shape is predictable
        without consulting these docs. Passing both `chunks` and `chunk_size_mb` is
        an error: one is exact, the other a budget.
    chunk_size_mb : float, optional
        Target size of an auto-chosen chunk, in MiB. The budget form of `chunks`,
        exactly as `region_size_mb` is the budget form of `region_shape`. Giving it
        opts OUT of preserving the input grid: the chunk is then solved for, jointly
        with the region, from sizes that tile the source grid. Approximate by
        definition - those sizes are quantised, so the result can miss the request;
        it is reported when it does. Pass `chunks=` for an exact shape.
    shard_coefficients : tuple of int, optional
        Multipliers for chunk sizes to compute shard shape (Zarr v3 only).
        Example: chunks=(64, 64), shard_coefficients=(4, 4) â†’ shards=(256, 256)
        If None, infers from input array. If input has no shards, writes without sharding.
    compressor : Codecs, numcodecs compressor, or None
        Compression configuration. Can be:
        - Codecs object: Unified compression config for v2/v3
        - numcodecs compressor: For backward compatibility (converted to Codecs)
        - None: Infers from input array, or uses Blosc with LZ4 as default
    zarr_format : int
        Zarr format version (2 or 3, default: 3)
    region_size_mb : float, optional
        Target size of read regions in MiB (default 8.0). The budget form of
        `region_shape`; passing both is an error. Approximate, and adjusted to hold
        a whole number of output chunks - that part is correctness, not tuning,
        since concurrent regions sharing a chunk would race on it.
    region_shape : tuple of int, optional
        EXACT read-region shape, honoured verbatim. Must be a per-axis multiple
        of `chunks` (clamped to the array shape). Use it to align the read regions to a
        producer's processing grid - e.g. pass a tilewise-ccl `tile_shape` so its position-aware
        Phase-B `map_overlap` reads each whole tile exactly once (no re-read / re-label
        amplification) while still storing small `chunks`.
    overwrite : bool
        Replace an existing array at `output_path`. Default False: on the default
        (TensorStore) region writer, writing to a path that already holds an array
        raises ALREADY_EXISTS, even when the existing array has the same metadata.
        NOT yet uniform: `backend='zarrista'` and the staged flatten/reshape/scan
        writers still write into the existing array without raising, and the staged
        writers ignore `overwrite` entirely. Only a local path that already looks like
        a zarr store is removed, so a mistyped `output_path` cannot delete an
        unrelated tree.
    gc_interval : float
        Seconds between GC runs (default: 15.0)
    """
    # Resolve and validate EVERYTHING first, for every write strategy, before any
    # storage is touched: format, chunks, codecs, shards, backend, overwrite and all
    # argument checks. A rejected write therefore leaves nothing behind.
    spec = _resolve_output(
        array, output_path, max_workers=max_workers, num_readers=num_readers,
        chunks=chunks, chunk_size_mb=chunk_size_mb, shard_coefficients=shard_coefficients,
        dtype=dtype, compressor=compressor, zarr_format=zarr_format,
        region_size_mb=region_size_mb, gc_interval=gc_interval, device=device,
        region_shape=region_shape, memory_budget_mb=memory_budget_mb,
        overwrite=overwrite, backend=backend, storage_options=storage_options,
    )
    _region_mb = spec.region_mb
    for _note in spec.notes:
        print(f"[Alignment] {_note}")

    # Small reductions the chain re-reads (the 0-d `level` in `x > level`, per-channel
    # stats broadcast back) are computed NOW, once, fused per input, before any region
    # is read. Otherwise every region would re-stream them. See operations.reductions.
    from .operations.reductions import evaluate_small_reductions
    from .operations._backend import device_context as _device_context
    with _device_context(device):
        evaluate_small_reductions(array)

    # An OUTERMOST reshape/flatten/scan cannot be produced memory-bounded by the region
    # pipeline; it goes to its staged writer - after the same resolution, into the same
    # kind of output, on every backend, local or remote. See _write_staged.
    _staged = _STAGED_WRITES.get(type(getattr(array, "_transform", None)).__name__)
    if _staged is not None:
        return _write_staged(_staged, array, spec, max_workers=max_workers, device=device,
                             region_shape=region_shape, memory_budget_mb=memory_budget_mb,
                             num_readers=num_readers)

    # Opened BEFORE gc is disabled, so a failure to create the store cannot leave the
    # interpreter's collector switched off.
    output = _open_output(spec)

    # Aggressive memory cleanup before starting
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()

    input_shape_temp = spec.shape
    final_dtype = spec.dtype
    final_dtype_obj = spec.dtype
    final_chunks = spec.chunks
    _solved_region = spec.solved_region

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
            # Solved jointly with the chunks, so it is already a whole multiple of
            # them and, where the source grid allowed it, of the source chunks too.
            # Recomputing it here would undo that.
            region_shape = tuple(min(int(r), s) for r, s in
                                 zip(_solved_region, input_shape))
            # A reduction too large to cache, broadcast back over the axes it reduced
            # (x - x.mean(axis=0)), is re-streamed for every region along those axes.
            # Regions spanning those axes whole read it exactly once, at no extra
            # storage, whenever that still fits the region budget.
            _spanned = _span_repeated_axes(
                array, region_shape, input_shape, final_chunks, _region_mb,
                _widest_dtype(_chain_widest_dtype(array), final_dtype))
            if _spanned is not None and _spanned != region_shape:
                print(f"[Optimized] Region {region_shape} -> {_spanned}: spans the axes "
                      f"a broadcast reduction is repeated along, so it is read once")
                region_shape = _spanned
            print(f"[Optimized] Output chunks: {final_chunks}, Region: {region_shape}, Size: ~{_region_mb}MB")

    # Report any grid that does not tile cleanly. Misalignment is legal and often
    # unavoidable (a coprime source grid, or an exact region_shape the caller chose),
    # but it silently costs read amplification - a read then straddles source chunks
    # on both sides - so it must never be invisible.
    _src_grid = input_chunks if input_chunks else None
    if _src_grid and len(_src_grid) == len(final_chunks):
        _bad = [f"axis {a}: chunk {c} vs source chunk {s}"
                for a, (c, s) in enumerate(zip(final_chunks, _src_grid))
                if s and c % s and s % c]
        if _bad:
            print(f"[Alignment] Output chunks {tuple(final_chunks)} do not tile the "
                  f"source grid {tuple(_src_grid)} ({'; '.join(_bad)}). Reads will "
                  f"straddle stored chunks; omit chunks=/chunk_size_mb= to keep the "
                  f"source grid, or pass chunks= matching it.")
    _bad_region = [f"axis {a}: region {r} vs chunk {c}"
                   for a, (r, c) in enumerate(zip(region_shape, final_chunks))
                   if c and r % c and r != input_shape[a]]
    if _bad_region:
        print(f"[Alignment] Region {tuple(region_shape)} does not tile the output "
              f"chunks {tuple(final_chunks)} ({'; '.join(_bad_region)}). Concurrent "
              f"region writes can then share a chunk.")

    if backend == "zarrista":
        # Regions are written CONCURRENTLY, and every region above tiles the output
        # CHUNK - but on a sharded array the unit two writers must not share is the
        # SHARD. A region smaller than a shard would have concurrent writers meeting
        # inside one shard file, which zarrista 0.1.0 does not survive: it drops the
        # overlapping data silently, with no error (see reports/zarrista_bugs/). Grow
        # the region to whole shards rather than fail, since the region is an internal
        # tuning choice here, not something the caller asked for.
        unit = output.write_unit
        if any(r % u for r, u in zip(region_shape, unit)):
            grown = tuple(
                min(-(-r // u) * u, s)
                for r, u, s in zip(region_shape, unit, input_shape)
            )
            print(f"[Optimized] Region {region_shape} would split a zarrista write unit "
                  f"{unit}; grown to {grown} to keep concurrent writes safe")
            region_shape = grown

    _warn_repeated_reductions(array, region_shape, input_shape)

    # Generate chunk indices
    align_offset = getattr(getattr(array, "_transform", None), "align_offset", None)
    if align_offset is None:
        align_offset = (0,) * len(input_shape)

    def _bands(size, rs, off, chunk):
        """Per-axis (start, stop) read bands.

        Bands are cut on the PRODUCER's cell grid, so each cell is read and computed
        exactly once. With a cell offset those cuts generally do not land on the output
        chunk grid, so two neighbouring bands write into the same boundary chunk - that is
        safe here because the writes go through TensorStore, which serialises the
        read-modify-write of a chunk internally (verified: 8 threads, compressed chunks,
        bands cut mid-chunk, byte-exact results). Cutting on the chunk grid instead would
        make every band straddle a cell border and the producer would recompute those
        cells once per band (measured 2.4x extra labeling).
        """
        if not off % rs:
            return [(x, min(x + rs, size)) for x in range(0, size, rs)]
        if size <= rs:
            # The whole axis fits inside one cell-span. Splitting it can only ADD cells:
            # a crop of 71 voxels straddling one cell border becomes 2 bands that together
            # touch the same 2 cells, but each band re-reads whichever cells it overlaps.
            # One band touches each cell exactly once, so never split here.
            return [(0, size)]
        cuts = [0]
        c = (rs - off % rs) % rs
        while c < size:
            if c > cuts[-1]:
                cuts.append(c)
            c += rs
        cuts.append(size)
        return [(a, b) for a, b in zip(cuts, cuts[1:]) if a < b]

    chunk_indices = list(itertools.product(
        *[_bands(s, rs, o, c) for s, rs, o, c in
          zip(input_shape, region_shape, align_offset, final_chunks)]
    ))
    
    total_chunks = len(chunk_indices)
    quarter_target = max(1, total_chunks // 4)
    tenth_target = max(1, total_chunks // 10)
    
    print(f"[Optimized] Total regions to process: {total_chunks}, Shape: {input_shape}, Region: {region_shape}", flush=True)
    
    # --- Pipeline sizing: ONE budget (max_workers = total live regions) split across the
    # three stages that actually pin buffers (readers / queue / in-flight writes).
    # Peak RAM ~= FIXED + k * max_workers * region_bytes (k ~= 2.5, FIXED ~= 200 MB);
    # this budget covers the live-region term only, so the caller can reason about the two
    # knobs they set. Previously each stage was an independent multiple of max_workers
    # (8 + 8 + 16 = 32 live regions) with region SIZE never entering the formula: fine at
    # the 8MB default (~256MB), ~36GB at a 672^3 int32 region.
    # Bands are cell-cut and chunk-rounded, so the widest one can exceed `region_shape`.
    # Size the pipeline from what will ACTUALLY be read, or the budget under-counts.
    _max_extent = [0] * len(input_shape)
    for _band in chunk_indices:
        for _a, (_lo, _hi) in enumerate(_band):
            _max_extent[_a] = max(_max_extent[_a], min(_hi, input_shape[_a]) - _lo)
    region_bytes = int(np.prod(_max_extent)) * int(np.dtype(final_dtype_obj).itemsize)

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
                        slice(lo, min(hi, dim_size))
                        for (lo, hi), dim_size in zip(chunk_start, input_shape)
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
    def read(source: Union[str, Path], backend: Optional[str] = None,
             storage_options: Optional[dict] = None) -> 'DynamicArray':
        """Read array from file path. See read_array() for details.

        ``backend`` selects the Zarr reader: ``'tensorstore'`` (default) or the
        optional ``'zarrista'``. Mirrors ``read_array``'s signature and forwards by
        keyword, for the same reason ``write`` does.
        """
        return read_array(source, backend=backend, storage_options=storage_options)

    @staticmethod
    def clear_cache():
        """Forget every cached reduction result.

        Reductions with a small output (at most 1 MiB, e.g. ``x.mean()``) are computed
        once and cached on their node, so a chain like ``x > x.mean()`` does not
        recompute the mean for every region. The cache assumes the data does not
        change underneath; call this after rewriting a source in place, so the next
        read recomputes. ``arr.clear_cache()`` does the same for one array's chain.
        Nothing on disk is touched (``persist()`` stores are separate).
        """
        from .operations.reductions import clear_all_caches
        clear_all_caches()

    @staticmethod
    def write(
        array: 'DynamicArray',
        output_path: str,
        max_workers: int = 4,
        num_readers: Optional[int] = None,
        chunks: Optional[Tuple[int, ...]] = None,
        chunk_size_mb: Optional[float] = None,
        shard_coefficients: Optional[Tuple[int, ...]] = None,
        dtype: Optional[Any] = None,
        compressor: Optional[Union[Codecs, Any]] = None,
        zarr_format: Optional[int] = None,
        region_size_mb: Optional[float] = None,
        gc_interval: float = 15.0,
        early_quarter_timeout: Optional[float] = None,
        early_tenth_timeout: Optional[float] = None,
        device: Optional[str] = None,
        region_shape: Optional[Tuple[int, ...]] = None,
        memory_budget_mb: Optional[float] = None,
        overwrite: bool = False,
        backend: str = "tensorstore",
        storage_options: Optional[dict] = None,
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
            chunk_size_mb=chunk_size_mb,
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
            backend=backend,
            storage_options=storage_options,
        )
