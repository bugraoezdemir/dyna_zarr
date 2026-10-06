"""Optional zarrista storage backend.

zarrista is a Rust-backed (zarrs) Zarr implementation. Benchmarked against the
TensorStore path on this repo's ``benchmarks/`` harness -- matched codecs, matched
concurrency, byte-identical output verified in both directions -- it was faster in
every configuration measured (v3 unsharded 2.40x write / 2.00x read; v3 sharded
1.62x / 2.15x; v2 1.62x / 1.47x).

It is **opt-in**, and deliberately so: zarrista 0.1.0 is beta-quality by its own
README, and has two silent failure modes this module exists to contain. Both are
documented with runnable repros in ``reports/zarrista_bugs/``.

1. **Misaligned concurrent writes silently lose data.** Threads writing regions that
   do not tile the array's write unit (the chunk when unsharded, the shard when
   sharded) race on the shared unit and drop roughly half the array, with no
   exception. :func:`check_write_alignment` refuses that combination up front. This
   is the same invariant dyna already enforces for ``region_shape`` -- extended here
   to cover shards, which are the write unit on sharded v3.

2. **Integer indexing does not drop the indexed axis**, and steps/newaxis/fancy
   indexing are unimplemented. dyna's ``_read_direct`` does send bare ints and
   ``step != 1`` slices to the backend, so :class:`ZarristaArray` normalizes numpy
   basic-indexing semantics itself: it asks zarrista only for contiguous unit-step
   spans and re-applies the int-squeeze and the step in memory. That mirrors what
   ``operations/structural.py`` already does one layer up, and costs nothing --
   the wrapper measured within noise of raw zarrista.

Zarr v2 is supported, but only through ``Array.from_metadata``: zarrista's
``ArrayBuilder`` exposes no format knob and always emits ``zarr_format: 3``. That
path is undocumented upstream, so :func:`create_v2` is the single place that depends
on it, and ``tests/test_zarrista_backend.py`` round-trips v2 against zarr-python to
catch a change in a future release.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    import zarrista as _zarrista
    from zarrista import codec as _codec
    from zarrista.store import FilesystemStore as _FilesystemStore

    ZARRISTA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by the import-guard test
    _zarrista = None
    _codec = None
    _FilesystemStore = None
    ZARRISTA_AVAILABLE = False


#: The one version this backend's guards were written and verified against. The
#: ``[zarrista]`` extra pins it, but the pin only binds when the extra is what
#: installed zarrista -- a direct ``pip install zarrista`` or an editable checkout
#: bypasses it entirely -- so the version is also checked at runtime.
TESTED_VERSION = "0.1.0"

__all__ = [
    "SUPPORTED_COMPRESSORS",
    "TESTED_VERSION",
    "ZARRISTA_AVAILABLE",
    "AlignmentError",
    "UnsupportedByBackend",
    "check_supported",
    "ZarristaArray",
    "check_write_alignment",
    "create_v2",
    "create_v3",
    "installed_version",
    "open_array",
    "require_zarrista",
]


class AlignmentError(ValueError):
    """Raised instead of letting a misaligned concurrent write corrupt data."""


def installed_version():
    """Version of the installed zarrista, or None when it is absent/unknown."""
    if not ZARRISTA_AVAILABLE:
        return None
    return getattr(_zarrista, "__version__", None)


_version_warned = False


def _check_version():
    """Warn once when running against a zarrista this backend was not verified on.

    A warning, not an error: the guards may well still hold, and blocking someone
    who deliberately upgraded would be worse than telling them what is unverified.
    The failure modes are silent, though, so the warning names them -- in particular
    an upstream FIX to the integer-index bug would make ``_normalize_key`` squeeze a
    second time, turning a fix into a wrong-shape regression.
    """
    global _version_warned
    found = installed_version()
    if _version_warned or found is None or found == TESTED_VERSION:
        return
    _version_warned = True
    import warnings

    warnings.warn(
        f"zarrista {found} is installed, but the dyna_zarr zarrista backend was "
        f"verified against {TESTED_VERSION}. Its guards assume upstream behaviour "
        f"that may have changed: the zarr-v2 path rides on the undocumented "
        f"Array.from_metadata API, and _normalize_key compensates for integer "
        f"indexing not dropping an axis (were that fixed upstream, the result "
        f"would be squeezed twice). Run tests/test_zarrista_backend.py before "
        f"relying on this backend.",
        RuntimeWarning,
        stacklevel=3,
    )


def require_zarrista():
    """Raise a useful message rather than an AttributeError on a missing optional dep."""
    if not ZARRISTA_AVAILABLE:
        raise ImportError(
            "the zarrista backend requires the 'zarrista' package "
            "(pip install 'dyna-zarr[zarrista]' to get the verified version); "
            "dyna_zarr works without it via tensorstore"
        )
    _check_version()


# --------------------------------------------------------------------------- #
# Index normalization (bug 2 containment)
# --------------------------------------------------------------------------- #

def _normalize_key(key, shape):
    """numpy basic-index key -> (contiguous unit-step key, squeeze axes, per-axis steps).

    zarrista only accepts contiguous ``step == 1`` ranges and never drops an
    integer-indexed axis, so the span it can serve is separated here from the
    reshaping that has to happen in memory afterwards.
    """
    if not isinstance(key, tuple):
        key = (key,)

    if any(k is Ellipsis for k in key):
        i = next(n for n, k in enumerate(key) if k is Ellipsis)
        fill = len(shape) - (len(key) - 1)
        if fill < 0:
            raise IndexError("too many indices for array")
        key = key[:i] + (slice(None),) * fill + key[i + 1:]

    if any(k is None for k in key):
        raise NotImplementedError(
            "np.newaxis is not supported at the storage layer; add the axis after reading"
        )
    if len(key) > len(shape):
        raise IndexError(f"too many indices: array is {len(shape)}-D, got {len(key)}")

    key = key + (slice(None),) * (len(shape) - len(key))

    base, squeeze, steps = [], [], []
    for axis, (k, size) in enumerate(zip(key, shape)):
        if isinstance(k, (int, np.integer)):
            idx = int(k) + size if k < 0 else int(k)
            if not 0 <= idx < size:
                raise IndexError(f"index {k} is out of bounds for axis {axis} with size {size}")
            base.append(slice(idx, idx + 1))
            squeeze.append(axis)
            steps.append(1)
        elif isinstance(k, slice):
            start, stop, step = k.indices(size)
            if step < 0:
                raise NotImplementedError(
                    "negative-step slicing is not supported at the storage layer"
                )
            # An empty span (start >= stop) still has to produce an empty result, so
            # keep it well-formed rather than handing zarrista a reversed range.
            base.append(slice(start, max(start, stop)))
            steps.append(step)
        else:
            raise NotImplementedError(
                f"unsupported index type {type(k).__name__}; "
                "only integers and unit/positive-step slices reach the storage layer"
            )

    return tuple(base), tuple(squeeze), tuple(steps)


# --------------------------------------------------------------------------- #
# Alignment guard (bug 1 containment)
# --------------------------------------------------------------------------- #

def check_write_alignment(region, write_unit, threads):
    """Refuse a concurrent write whose regions would share a write unit.

    Serial writes are safe at any region shape, so only ``threads > 1`` is guarded.
    """
    if threads <= 1:
        return
    if len(region) != len(write_unit):
        raise ValueError(
            f"region {tuple(region)} has {len(region)} dims but the array's write unit "
            f"{tuple(write_unit)} has {len(write_unit)}"
        )
    bad = [(a, r, u) for a, (r, u) in enumerate(zip(region, write_unit)) if r % u]
    if bad:
        detail = ", ".join(f"axis {a}: {r} % {u} != 0" for a, r, u in bad)
        raise AlignmentError(
            f"region {tuple(region)} is not a per-axis multiple of the write unit "
            f"{tuple(write_unit)} ({detail}). Concurrent writes to a shared unit "
            f"SILENTLY lose data in zarrista 0.1.0 (see reports/zarrista_bugs/). "
            f"Use a region that tiles the write unit, or pass threads=1."
        )


# --------------------------------------------------------------------------- #
# Array wrapper
# --------------------------------------------------------------------------- #

class ZarristaArray:
    """numpy-semantics view over a ``zarrista.Array``.

    Exposes the ``shape``/``chunks``/``dtype`` surface and the ``__getitem__`` /
    ``__setitem__`` contract that ``DynamicArray`` expects of a storage source.
    """

    def __init__(self, inner):
        require_zarrista()
        self._array = inner

    # -- metadata -------------------------------------------------------- #

    @property
    def shape(self):
        return tuple(self._array.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return np.dtype(self._array.dtype.name)

    @property
    def is_sharded(self):
        return bool(self._array.is_sharded)

    @property
    def chunks(self):
        """Inner chunk shape -- the subchunk when sharded, matching zarr's ``chunks``."""
        if self.is_sharded:
            return tuple(self._array.effective_subchunk_shape)
        return self.write_unit

    @property
    def write_unit(self):
        """Contention unit for concurrent writes.

        ``chunk_shape(idx)`` returns the OUTER grid cell, which is the shard on a
        sharded array and the chunk otherwise -- exactly the unit that must not be
        shared between concurrent writers.
        """
        return tuple(self._array.chunk_shape([0] * self.ndim))

    # -- data ------------------------------------------------------------ #

    def __getitem__(self, key):
        base, squeeze, steps = _normalize_key(key, self.shape)
        out = np.asarray(_call(self._array.retrieve_array_subset, base))
        if any(s != 1 for s in steps):
            out = out[tuple(slice(None, None, s) for s in steps)]
        if squeeze:
            out = out.squeeze(axis=squeeze)
        return out

    def __setitem__(self, key, value):
        base, squeeze, steps = _normalize_key(key, self.shape)
        if any(s != 1 for s in steps):
            raise NotImplementedError(
                "strided writes are not supported; write a contiguous region"
            )
        target = tuple(b.stop - b.start for b in base)
        data = np.ascontiguousarray(value, dtype=self.dtype)
        if data.shape != target:
            # Broadcasting covers scalars and the squeezed-axis case (a[3] = plane).
            data = np.ascontiguousarray(np.broadcast_to(data, target), dtype=self.dtype)
        _call(self._array.store_array_subset, base, data)

    def __array__(self, dtype=None, copy=None):
        out = self[...]
        return out.astype(dtype) if dtype is not None else out

    # -- bulk region write ----------------------------------------------- #

    def write_regions(self, region, data, threads=8):
        """Write ``data`` region-by-region, guarding the misaligned-concurrent trap."""
        region = tuple(int(r) for r in region)
        check_write_alignment(region, self.write_unit, threads)

        shape = self.shape
        slices = [
            tuple(slice(a, min(a + r, s)) for a, r, s in zip(start, region, shape))
            for start in itertools.product(
                *[range(0, s, r) for s, r in zip(shape, region)]
            )
        ]
        if threads <= 1:
            for sl in slices:
                self[sl] = data[sl]
            return
        with ThreadPoolExecutor(max_workers=min(threads, len(slices))) as pool:
            list(pool.map(lambda sl: self.__setitem__(sl, data[sl]), slices))

    def __repr__(self):
        kind = "sharded" if self.is_sharded else "chunked"
        return (f"ZarristaArray(shape={self.shape}, dtype={self.dtype.name}, "
                f"{kind}, write_unit={self.write_unit})")


# --------------------------------------------------------------------------- #
# Constructors
# --------------------------------------------------------------------------- #

def _store(path):
    # A file:// URL passed straight to Path() resolves to a DRIVE-RELATIVE path on
    # Windows ('file:///C:/x' -> 'C:x'), i.e. somewhere under the working directory.
    from ..io import _local_path_from_file_url
    return _FilesystemStore(str(Path(_local_path_from_file_url(path)).resolve()))


def create_v3(path, shape, chunks, dtype, shard=None, codecs=None,
              storage_options=None, dimension_names=None):
    """Create a zarr v3 array. ``shard`` makes ``chunks`` the inner subchunk shape."""
    require_zarrista()
    from ..codecs import Codecs

    dtype = np.dtype(dtype)
    codecs = codecs if codecs is not None else Codecs()
    grid = _zarrista.ChunkGrid.regular(tuple(shape), chunk_shape=tuple(shard or chunks))
    builder = _zarrista.ArrayBuilder(
        grid,
        _zarrista.DataType.from_string(dtype.name),
        _zarrista.FillValue(dtype.type(0).tobytes()),
    ).compressors(_compressor_chain(codecs, dtype))
    if shard:
        builder = builder.subchunk_shape(tuple(chunks))
    if dimension_names is not None:
        builder = builder.dimension_names(list(dimension_names))
    if _is_remote(path):
        store = _obstore(path, storage_options)
        return ZarristaArray(
            _run_coroutine_fn(lambda: builder.create_async(store=store, path="/"))
        )
    return ZarristaArray(builder.create(store=_store(path), path="/"))


def create_v2(path, shape, chunks, dtype, codecs=None, separator="/",
              storage_options=None):
    """Create a zarr v2 array.

    zarrista's ``ArrayBuilder`` cannot emit v2 (it hardcodes ``zarr_format: 3``), so
    this goes through ``Array.from_metadata``, which is undocumented upstream and
    accepts a dict but NOT a JSON string. Keep this the only caller of that path.
    """
    require_zarrista()
    from ..codecs import Codecs

    dtype = np.dtype(dtype)
    codecs = codecs if codecs is not None else Codecs()
    if not _is_remote(path):
        Path(path).mkdir(parents=True, exist_ok=True)
    metadata = {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": dtype.str,
        "compressor": codecs.to_v2_config(),
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": separator,
    }
    if _is_remote(path):
        store = _obstore(path, storage_options)
        # Unlike create_async/open, from_metadata returns the AsyncArray directly
        # rather than a coroutine (it only builds the object; store_metadata does
        # the I/O), so this must NOT be awaited.
        array = _zarrista.AsyncArray.from_metadata(metadata, store, "/")
    else:
        array = _zarrista.Array.from_metadata(metadata, _store(path), "/")
    _call(array.store_metadata)
    return ZarristaArray(array)


def _compressor_chain(codecs, dtype):
    """Map a dyna ``Codecs`` onto zarrista's bytes-to-bytes codec list."""
    name = getattr(codecs, "compressor", None)
    if name is None:
        return []
    clevel = getattr(codecs, "clevel", 5)
    if name == "blosc":
        shuffle = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}.get(
            getattr(codecs, "shuffle", 1), "shuffle"
        )
        kwargs = {} if shuffle == "noshuffle" else {"typesize": dtype.itemsize}
        return [_codec.blosc(getattr(codecs, "cname", "lz4"), clevel, shuffle, **kwargs)]
    if name == "zstd":
        # checksum=False matches the default path: Codecs.to_v3_config builds
        # ZstdCodec(level=...) and zarr's default for that flag is False, so both
        # backends must write the same `checksum` in the metadata to stay identical.
        return [_codec.zstd(clevel, False)]
    if name == "gzip":
        return [_codec.gzip(clevel)]
    # check_supported() rejects these at the entry point; this is the backstop for
    # anyone calling create_v2/create_v3 directly.
    raise UnsupportedByBackend(
        f"compressor {name!r} has no zarrista equivalent; "
        f"use blosc, zstd, gzip, or None"
    )


def _is_remote(path):
    return "://" in str(path) and not str(path).startswith("file://")


# --------------------------------------------------------------------------- #
# Capability guard
# --------------------------------------------------------------------------- #

class UnsupportedByBackend(ValueError):
    """``backend='zarrista'`` was asked for something this backend cannot do.

    Its own class so callers can distinguish "this backend can't" from an ordinary
    bad argument, and fall back to the default backend deliberately if they want to.
    """


#: Compressors with a zarrista equivalent. Anything else must be refused up front
#: rather than at codec-build time, half way through opening the output.
SUPPORTED_COMPRESSORS = frozenset({None, "blosc", "zstd", "gzip"})


def check_supported(*, source=None, output=None, zarr_format=None, codecs=None,
                    storage_options=None, path_in_store="/"):
    """Refuse, before any work happens, anything this backend cannot honour.

    ``backend='zarrista'`` is an explicit request, so every unsupported combination
    has to fail loudly and say why: silently doing something else would make a
    benchmark, a bug report, or a correctness claim meaningless. This is the ONE
    place that knows the backend's limits - add new ones here, not at the call site.

    **Scope: only what ZARRISTA specifically cannot do.** A request that is wrong on
    every backend belongs to whichever layer owns it, so that the default path gets
    the same error - putting it here would leave the common case unguarded. Hence
    ``zarr_format``, a TIFF output path and a TIFF source with an explicit backend
    are all checked in ``io.py``, not here; what stays is the group-vs-array
    distinction, the codec set, remote-store construction, and misplaced
    ``storage_options``.

    Raises:
        UnsupportedByBackend: with a message naming the limitation and the way out.
        ImportError: when a remote path is used without obstore installed.
    """
    require_zarrista()

    for role, path in (("source", source), ("output", output)):
        if path is None:
            continue
        text = str(path)

        # NB the TIFF cases are NOT handled here. A TIFF *source* is rejected in
        # read_array for ANY explicit backend (tifffile reads it under all of them,
        # so the argument simply does not apply), and a TIFF *output* is not a
        # backend question at all -- dyna only ever writes Zarr, so a `.tif` output
        # path is a misleading filename on every backend and is refused in
        # write_array. Both live where they apply to both backends, not here.

        if _is_remote(text):
            # Fail here, not at the first byte of I/O, and say which package.
            _obstore(text, storage_options)
        elif storage_options:
            raise UnsupportedByBackend(
                f"storage_options are for remote stores, but the {role} {text!r} is "
                f"a local path. Drop storage_options, or pass a remote URL."
            )

    # NB zarr_format is NOT validated here: dyna writes v2/v3 on every backend, so
    # write_array owns that check. This guard answers only "can ZARRISTA do it".

    name = getattr(codecs, "compressor", None) if codecs is not None else None
    if codecs is not None and name not in SUPPORTED_COMPRESSORS:
        # v2 accepts numcodecs ids that zarrista's codec set has no mapping for
        # (bz2, for instance, writes fine on the default v2 path), so this refusal
        # is real and specific to zarrista - name the alternative rather than
        # implying the codec itself is invalid.
        raise UnsupportedByBackend(
            f"backend='zarrista' has no equivalent for compressor {name!r} "
            f"(it supports blosc, zstd, gzip, or None), though the default backend "
            f"may accept it for this format. Use backend='tensorstore', or pick one "
            f"of those codecs."
        )

    # Local source: say what is actually wrong with the path before zarrista does it
    # with a bare "array metadata is missing", which names neither the path nor the
    # reason. Remote paths are left to the store, which cannot be probed cheaply.
    if source is not None and path_in_store == "/" and not _is_remote(str(source)):
        local = Path(source)
        if not local.exists():
            # Not backend-specific (every backend needs the path to exist), but
            # zarrista would otherwise report it as "array metadata is missing",
            # which names neither the path nor the real problem.
            raise FileNotFoundError(f"{source} does not exist")
        if local.is_dir():
            is_array = (local / ".zarray").exists() or (local / "zarr.json").exists()
            if not is_array:
                members = sorted(
                    c.name for c in local.iterdir()
                    if c.is_dir()
                    and ((c / ".zarray").exists() or (c / "zarr.json").exists())
                )
                if (local / ".zgroup").exists() or members:
                    hint = f" (try one of: {', '.join(members)})" if members else ""
                    raise UnsupportedByBackend(
                        f"backend='zarrista' opens a zarr ARRAY, but {source} is a "
                        f"group{hint}. Point at the array itself, or use the default "
                        f"backend, which picks a group's first array for you."
                    )
                raise UnsupportedByBackend(
                    f"{source} is not a zarr array (no .zarray or zarr.json)."
                )


def _obstore(url, storage_options=None):
    """obstore ObjectStore for ``url`` (s3://, gs://, az://, http://).

    Object stores are reachable only through zarrista's ASYNC API, which takes an
    obstore store rather than one of zarrista's own store types.

    ``storage_options`` is forwarded to ``obstore.store.from_url``, which is how
    every non-default S3 is reached: a MinIO/Ceph/EMBL-style deployment needs
    ``endpoint=``, often ``region=``, and ``virtual_hosted_style_request=False``;
    a public bucket needs ``skip_signature=True``. Credentials otherwise come from
    obstore's usual environment variables.

    Note ``from_url`` dispatches on the SCHEME: ``https://host/bucket/key`` builds an
    HTTPStore (plain HTTP GETs), not an S3Store, so an S3 endpoint behind https must
    be given as ``s3://bucket/key`` with ``endpoint="https://host"``.
    """
    try:
        from obstore.store import from_url as _from_url
    except ImportError as exc:
        raise ImportError(
            "remote stores with backend='zarrista' need obstore "
            "(pip install 'dyna-zarr[zarrista]'). Local paths work without it, and "
            "the default backend='tensorstore' reaches remote stores directly."
        ) from exc
    if not storage_options:
        return _from_url(str(url))
    try:
        return _from_url(str(url), **storage_options)
    except Exception as exc:
        # obstore signals a bad key with its own UnknownConfigurationKeyError, and a
        # bad shape with TypeError; neither says the options came from dyna, so
        # re-raise with that context. The original is kept as __cause__.
        raise ValueError(
            f"obstore rejected these storage_options for {url!r}: "
            f"{type(exc).__name__}: {exc}. They are passed straight to "
            f"obstore.store.from_url - see its docs for the accepted keys "
            f"(endpoint, region, skip_signature, virtual_hosted_style_request, ...)."
        ) from exc


def _run_coroutine_fn(make_coro):
    """Call ``make_coro()`` inside an event loop and return its result.

    Takes a FACTORY rather than a coroutine: zarrista's async entry points bind to the
    running loop when they are called, so the coroutine has to be created inside the
    loop, not handed in from outside.
    """
    import asyncio

    async def _runner():
        return await make_coro()

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_runner())
    # Already inside a loop (notebook, async caller): run it on its own thread so
    # this stays a plain synchronous call for every caller.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_runner())).result()


def _call(method, *args, **kwargs):
    """Call ``method`` and return its result, driving it in a loop when it is async.

    The sync ``Array`` returns data directly; ``AsyncArray`` (the only way to reach an
    object store) returns awaitables from the same method names. Absorbing that here
    keeps ZarristaArray a single synchronous class for local and remote alike.
    """
    import inspect

    try:
        result = method(*args, **kwargs)
    except RuntimeError as exc:
        # zarrista's async methods raise "no running event loop" at CALL time rather
        # than returning a coroutine, so the retry has to happen inside a loop.
        if "event loop" not in str(exc):
            raise

        async def _inside():
            out = method(*args, **kwargs)
            return await out if inspect.isawaitable(out) else out

        return _run_coroutine_fn(_inside)

    if inspect.isawaitable(result):
        async def _await():
            return await result

        return _run_coroutine_fn(_await)
    return result


def open_array(path, path_in_store="/", storage_options=None):
    """Open an existing v2 or v3 array (zarrista reads both).

    Local paths use zarrista's sync ``Array``. Remote URLs (``s3://``, ``gs://``,
    ``az://``, ``http(s)://``) go through its ASYNC ``AsyncArray`` over an obstore
    ``ObjectStore``, which is the only API zarrista exposes for object stores; that
    needs the extra dependency, hence the targeted error below rather than a blanket
    "local paths only".
    """
    require_zarrista()
    if _is_remote(path):
        return ZarristaArray(_open_remote(path, path_in_store, storage_options))

    # Path validation (group vs array, missing, TIFF, ...) lives in
    # check_supported(), which io.read calls before any work; calling it here too
    # keeps direct open_array() users on the same guard.
    check_supported(source=path, storage_options=storage_options,
                    path_in_store=path_in_store)
    return ZarristaArray(_zarrista.Array.open(_store(path), path_in_store))


def _open_remote(url, path_in_store="/", storage_options=None):
    """Open a remote array through obstore + zarrista's async API.

    Both the open and the subsequent reads are async; ``_resolve`` awaits the read
    results, so ZarristaArray stays one synchronous class for local and remote alike.
    """
    store = _obstore(url, storage_options)

    # Built inside the coroutine: zarrista's async entry points bind to the running
    # loop when they are CALLED, not when they are awaited, so constructing the
    # coroutine outside a loop raises "no running event loop".
    async def _open():
        return await _zarrista.AsyncArray.open(store, path_in_store)

    return _run_coroutine_fn(_open)
