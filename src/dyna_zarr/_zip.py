"""Zarr arrays written straight into a zip file (zarr-zip's ZipWriter).

A zip is opened for writing with :func:`open_zip`; paths inside it - ``<zip>/<key>``, e.g.
``"img.ome.zarr.zip/0"`` - are then ordinary output and input paths for io.write,
io.write_levels, io.create_sink and io.read::

    with io.open_zip("img.ome.zarr.zip") as z:
        io.write_levels(arr, [dict(path=z / "0", factors=(1, 1, 1)),
                              dict(path=z / "1", factors=(1, 2, 2))])
        level0 = io.read(z / "0")              # read back before the zip is closed
        z.write("zarr.json", group_metadata)   # the caller's own files

How the data goes in. A zip entry's bytes are contiguous, so only finished pieces go in:

- An array is created by TensorStore in a store held in memory (one per array). Every
  write unit (shard, else chunk) is moved into the zip, and freed, the moment the regions
  written into it cover it; the writers write whole units, so nothing waits.
- Shards larger than the memory budget (io.write_levels' chunk-by-chunk mode) are
  assembled in temporary files next to the zip, and appended when their last chunk is in.

A finished zip is read like any store: ``io.read("img.ome.zarr.zip/0")``.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import zipfile
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ZIP_SUFFIXES = (".zip", ".ozx")
_METADATA = ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata")

_OPEN: Dict[str, "ZipTarget"] = {}          # normcase(absolute zip path) -> its target
_OPEN_LOCK = threading.Lock()


def _zarr_zip():
    try:
        import zarr_zip
    except ImportError:                                     # pragma: no cover
        raise ImportError("writing or reading zipped Zarr needs zarr-zip "
                          "(pip install zarr-zip)") from None
    return zarr_zip


class ZipTarget:
    """A zip being written. ``target / "0"`` is the path of the array (or group) ``0``
    inside it, for io.write / io.write_levels / io.create_sink / io.read.

    Close it (or leave its ``with`` block) to finish every array, write the zip's table of
    contents and rename ``<zip>.part`` to the zip; an error inside the block deletes it.
    """

    def __init__(self, path, *, layout: Optional[str] = None, root: Optional[str] = None,
                 overwrite: bool = False):
        zz = _zarr_zip()
        path = Path(os.path.abspath(str(path)))
        key = os.path.normcase(str(path))
        with _OPEN_LOCK:
            if key in _OPEN:
                raise ValueError(f"{path} is already open for writing")
            self.writer = zz.open_writer(path, layout=layout, root=root, overwrite=overwrite)
            _OPEN[key] = self
        self._key = key
        self.path = path
        self.layout = layout or zz.layout_of(path)
        self._outputs: List["ZipOutput"] = []
        self._lock = threading.Lock()
        self._temp = Path(str(self.writer.part) + ".shards")
        self._closed = False

    # ── addresses and the caller's own files ──────────────────────────────────

    def __truediv__(self, key) -> str:
        return self.join(key)

    def join(self, *keys) -> str:
        """The path of ``keys`` inside the zip, as io.write / io.read take it."""
        key = "/".join(str(k).strip("/\\") for k in keys if str(k).strip("/\\"))
        return str(self.path) + ("/" + key if key else "")

    def write(self, key: str, data) -> None:
        """Write a whole file into the zip (metadata: ``zarr.json``, ``.zattrs`` ...).
        Written again, the last copy is the one the zip lists."""
        self.writer.write(key, data)

    def read(self, key: str) -> bytes:
        return self.writer.read(key)

    def __contains__(self, key) -> bool:
        return key in self.writer

    def zarr_store(self, read_only: bool = False, path: str = ""):
        """A zarr-python store over the zip (reads what is written so far; writes append),
        rooted at ``path`` inside it as a folder path roots a local store."""
        from zarr_zip.zarr_store import WriterStore
        return WriterStore(self.writer, read_only=read_only, prefix=path)

    # ── outputs ───────────────────────────────────────────────────────────────

    def _add(self, output: "ZipOutput") -> None:
        with self._lock:
            if self._closed:
                raise ValueError(f"{self.path}: the zip is closed")
            self._outputs.append(output)

    def _done(self, output: "ZipOutput") -> None:
        with self._lock:
            if output in self._outputs:
                self._outputs.remove(output)

    def temp_dir(self) -> Path:
        self._temp.mkdir(parents=True, exist_ok=True)
        return self._temp

    # ── closing ───────────────────────────────────────────────────────────────

    def close(self, verify: str = "list") -> Path:
        """Finish every array still open, write the table of contents, check it
        (``verify``: "list", "full" = also every CRC, "none") and rename the zip."""
        if self._closed:
            return self.path
        try:
            with self._lock:
                outputs = list(self._outputs)
            for output in outputs:
                output.finish()
            with self._lock:
                self._closed = True
            return self.writer.close(verify=verify)
        except BaseException:
            self._discard()
            raise
        finally:
            self._release()

    def abort(self) -> None:
        """Stop and delete the unfinished zip and its temporary files."""
        self._discard()
        self._release()

    def _discard(self) -> None:
        with self._lock:
            self._closed = True
        self.writer.abort()

    def _release(self) -> None:
        shutil.rmtree(self._temp, ignore_errors=True)
        with _OPEN_LOCK:
            if _OPEN.get(self._key) is self:
                del _OPEN[self._key]

    def __enter__(self) -> "ZipTarget":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()

    def __repr__(self) -> str:
        return f"ZipTarget({str(self.path)!r}, {'closed' if self._closed else 'open'})"


def open_zip(path, *, layout: Optional[str] = None, root: Optional[str] = None,
             overwrite: bool = False) -> ZipTarget:
    """Open the zip ``path`` for writing Zarr straight into it (see :class:`ZipTarget`).

    ``layout`` (default from the name): ``"bia"`` for ``X.ome.zarr.zip`` - the store's
    folder ``X.ome.zarr`` inside the zip (or ``root``), as the BioImage Archive asks -
    and ``"rfc9"`` for ``X.ozx`` - the store's root at the zip's root, ZIP64, and the
    comment naming the OME-Zarr version, read from the root ``zarr.json`` at close (NGFF
    RFC-9, Zarr v3 only). Written as ``<path>.part`` and renamed when closed and checked;
    an existing zip is refused unless ``overwrite``.
    """
    return ZipTarget(path, layout=layout, root=root, overwrite=overwrite)


# ── addresses ─────────────────────────────────────────────────────────────────────


def _split(path) -> Optional[Tuple[str, str]]:
    """``(absolute zip path, key)`` when a component of a local ``path`` is named like a
    zip (``.zip`` / ``.ozx``) and is not a directory, else None."""
    s = str(path)
    if "://" in s:
        return None
    parts = Path(os.path.abspath(s)).parts
    for i, part in enumerate(parts):
        if part.lower().endswith(ZIP_SUFFIXES):
            zp = Path(*parts[: i + 1])
            if zp.is_dir():
                continue
            return str(zp), "/".join(parts[i + 1:])
    return None


def open_target(path) -> Optional[Tuple[ZipTarget, str]]:
    """``(target, key)`` when ``path`` is inside a zip open for writing."""
    if not _OPEN:
        return None
    hit = _split(path)
    if hit is None:
        return None
    with _OPEN_LOCK:
        target = _OPEN.get(os.path.normcase(hit[0]))
    return (target, hit[1]) if target is not None else None


def finished_zip(path) -> Optional[Tuple[str, str]]:
    """``(zip file, key)`` when ``path`` is inside an existing zip file not open for
    writing."""
    hit = _split(path)
    if hit is None or not Path(hit[0]).is_file():
        return None
    return hit


def zip_file_of(path) -> Optional[str]:
    """The zip ``path`` is, or is inside (a component named ``.zip`` / ``.ozx`` that is not
    a directory: open for writing, an existing file, or not there yet), else None."""
    hit = _split(path)
    return hit[0] if hit is not None else None


def store_at(path, *, write: bool = False):
    """``(zarr-python store, path inside it)`` for ``path`` inside a zip: one open for
    writing (reads see what is written so far; ``write`` lets zarr append metadata), or,
    read-only, a finished zip file. None when ``path`` is not inside either."""
    import zarr
    hit = open_target(path)
    if hit is not None:
        target, key = hit
        return target.zarr_store(read_only=not write, path=key), ""
    hit = finished_zip(path)
    if hit is None:
        return None
    if write:
        raise ValueError(f"{hit[0]} is a finished zip, which cannot be written into; "
                         f"write a new one with dyna_zarr.io.open_zip")
    return zarr.storage.ZipStore(hit[0], mode="r"), _inner(*hit)


def refuse_unopened(path) -> None:
    """A write to ``<x.zip>/<key>`` with no :func:`open_zip` for it: refused, rather than
    creating a directory named like a zip."""
    hit = _split(path)
    if hit is not None:
        raise ValueError(
            f"{str(path)!r} is inside the zip {hit[0]!r}, which is not open for writing. "
            f"Open it first: with dyna_zarr.io.open_zip({hit[0]!r}) as z: "
            f"io.write(arr, z / {hit[1] or 'key'!r}).")


def existing(target: ZipTarget, key: str) -> Optional[str]:
    """What the zip already holds at ``key`` (as io.write's _classify_store)."""
    k = (key + "/") if key else ""
    if k + ".zarray" in target or k + ".zgroup" in target:
        return "array" if k + ".zarray" in target else "group"
    if k + "zarr.json" in target:
        try:
            node = json.loads(target.read(k + "zarr.json")).get("node_type")
        except ValueError:
            node = None
        return node if node in ("array", "group") else "other"
    if any(name.startswith(k) for name in target.writer.keys()) and key:
        return "other"
    return None


# ── reading ───────────────────────────────────────────────────────────────────────


def zarr_array(path):
    """The zarr-python array at ``path`` inside a zip (open for writing, or finished), or
    None when ``path`` is not inside a zip."""
    import zarr
    hit = open_target(path)
    if hit is not None:
        target, key = hit
        return zarr.open_array(target.zarr_store(read_only=True, path=key), mode="r")
    hit = finished_zip(path)
    if hit is None:
        return None
    zip_path, key = hit
    store = zarr.storage.ZipStore(zip_path, mode="r")
    return zarr.open_array(store, path=_inner(zip_path, key), mode="r")


def _inner(zip_path: str, key: str) -> str:
    """``key`` inside the zip: under the store's folder when the zip holds one (the
    BioImage Archive's layout), else at the zip's root (RFC-9)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    top = {n.split("/", 1)[0] for n in names}
    meta_at_root = any(n in _METADATA for n in names)
    if not meta_at_root and len(top) == 1:
        folder = next(iter(top))
        return f"{folder}/{key}" if key else folder
    return key


# ── writing ───────────────────────────────────────────────────────────────────────


class _Done:
    """A finished write, with the future protocol io.write's pipeline expects."""
    __slots__ = ("_exc",)

    def __init__(self, exc=None):
        self._exc = exc

    def done(self):
        return True

    def result(self, timeout=None):
        if self._exc is not None:
            raise self._exc


class ZipOutput:
    """An output array inside an open zip: ``out[region].write(data)`` writes into a
    TensorStore array held in memory, and each write unit moves into the zip (and out of
    memory) as soon as the regions written cover it. Each element must be written once:
    a unit already in the zip is never written again (refused)."""

    def __init__(self, target: ZipTarget, key: str, ts_handle, *, shape, chunks, shards,
                 zarr_format):
        self.target, self.key = target, key
        self._ts = ts_handle
        self._kv = ts_handle.kvstore
        self.shape = tuple(int(s) for s in shape)
        self.chunks = tuple(int(c) for c in chunks)
        self.unit = tuple(int(u) for u in (shards if shards is not None else chunks))
        self.zarr_format = int(zarr_format)
        self._left: Dict[tuple, int] = {}       # unit -> elements not yet written
        self._moved = set()
        self._lock = threading.Lock()
        self._assemblers = []
        self._finished = False
        for name in self._kv.list().result():     # the array's metadata, at once
            name = name.decode()
            self.target.write(self._entry(name), self._kv.read(name).result().value)
        target._add(self)

    @property
    def write_unit(self):
        return self.unit

    def _entry(self, name: str) -> str:
        return f"{self.key}/{name}" if self.key else name

    def _unit_key(self, idx) -> str:
        if self.zarr_format == 3:
            return "/".join(["c"] + [str(i) for i in idx])
        return "/".join(str(i) for i in idx) if idx else "0"

    def _units(self, region):
        return product(*[range(r.start // u, -(-r.stop // u)) if r.stop > r.start else ()
                         for r, u in zip(region, self.unit)])

    def _unit_size(self, idx) -> int:
        n = 1
        for i, u, s in zip(idx, self.unit, self.shape):
            n *= min((i + 1) * u, s) - i * u
        return n

    def _overlap(self, idx, region) -> int:
        n = 1
        for i, u, r in zip(idx, self.unit, region):
            n *= max(0, min((i + 1) * u, r.stop) - max(i * u, r.start))
        return n

    def __getitem__(self, region):
        output = self

        class _Slot:
            __slots__ = ()

            def write(self, data):
                try:
                    output._write(region, data)
                except BaseException as exc:          # surfaced by .result()
                    return _Done(exc)
                return _Done()
        return _Slot()

    def _region(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (len(self.shape) - len(key))
        return tuple(slice(*k.indices(n)[:2]) if isinstance(k, slice) else slice(int(k), int(k) + 1)
                     for k, n in zip(key, self.shape))

    def _write(self, key, data) -> None:
        region = self._region(key)
        units = list(self._units(region))
        with self._lock:
            again = [u for u in units if u in self._moved]
        if again:
            raise ValueError(f"{self.target.path}/{self.key}: the {'shard' if self.unit != self.chunks else 'chunk'} "
                             f"{again[0]} is already in the zip; a zip entry is written once")
        self._ts[key].write(np.ascontiguousarray(data)).result()
        complete = []
        with self._lock:
            for u in units:
                left = self._left.get(u)
                if left is None:
                    left = self._unit_size(u)
                left -= self._overlap(u, region)
                if left <= 0:
                    self._left.pop(u, None)
                    self._moved.add(u)
                    complete.append(u)
                else:
                    self._left[u] = left
        for u in complete:
            self._move(u)

    def _move(self, idx) -> None:
        """Move the unit ``idx`` from memory into the zip (absent: all fill value, not
        stored, as on disk)."""
        name = self._unit_key(idx)
        got = self._kv.read(name).result()
        if got.state == "value":
            self.target.write(self._entry(name), got.value)
            del self._kv[name]

    # the transaction mode of write_levels writes through TensorStore views and says
    # when a unit is complete

    def with_transaction(self, txn):
        return self._ts.with_transaction(txn)

    def unit_written(self, idx) -> None:
        idx = tuple(int(i) for i in idx)
        with self._lock:
            if idx in self._moved:
                return
            self._moved.add(idx)
            self._left.pop(idx, None)
        self._move(idx)

    def chunk_output(self, meta_name: str = "zarr.json"):
        """write_levels' chunk-by-chunk mode: shards assembled in temporary files next to
        the zip, each appended (and its file deleted) when its last chunk is in."""
        from ._shard_writer import ShardAssembler
        meta = json.loads(self._kv.read(meta_name).result().value)
        folder = self.target.temp_dir() / f"{id(self):x}"
        writer = self.target.writer

        def appended(name, file_path):
            writer.write_from(self._entry(name), file_path, remove=True)
        assembler = ShardAssembler(str(folder), meta=meta, on_finish=appended)
        with self._lock:
            self._assemblers.append(assembler)
        return assembler

    def finish(self) -> None:
        """Move what is still in memory into the zip (units only partly written: the rest
        reads as the fill value) and finish open assembled shards."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
            assemblers = list(self._assemblers)
        for assembler in assemblers:
            assembler.finish()
        for name in self._kv.list().result():
            name = name.decode()
            if name.rsplit("/", 1)[-1] in _METADATA:
                continue
            got = self._kv.read(name).result()
            if got.state == "value":
                self.target.write(self._entry(name), got.value)
            del self._kv[name]
        self.target._done(self)
