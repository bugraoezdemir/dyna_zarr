"""Shards written chunk by chunk, in bounded memory, for local zarr v3 arrays.

TensorStore assembles a shard in a transaction that holds the whole shard until it is
committed (an 844 MiB shard per thread for 960^3 uint8). A shard's file is simple
(``sharding_indexed``, index at the end): the encoded inner chunks one after another,
then an index of ``(offset, nbytes)`` per inner chunk. So it can be written as its chunks
complete, holding only the index (16 bytes per inner chunk): each chunk is encoded
and appended to its shard's file, and the index is written when the shard's last chunk
is in. zarrista-bounded does the same through zarrs' partial encoding.

The array itself (``zarr.json``) is created by the caller's backend as usual; this only
writes its shard files, with the codecs that metadata names. Supported: an inner
pipeline of ``bytes`` plus at most one of ``blosc`` / ``zstd`` / ``gzip``, and an index
of ``bytes`` + ``crc32c`` at the end - what dyna and TensorStore write. Anything else is
refused by :func:`supported` (the caller then keeps TensorStore's own writer).

Chunks equal to the fill value are not written (their index entry says "missing"),
and a shard with no data gets no file - as TensorStore and zarrs do.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional

import numpy as np

_MISSING = np.uint64(2 ** 64 - 1)
_COMPRESSORS = ("blosc", "zstd", "gzip")


def _metadata(path: str) -> Optional[dict]:
    try:
        with open(os.path.join(path, "zarr.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None
    return meta if meta.get("node_type") == "array" and meta.get("zarr_format") == 3 else None


def _why_not(meta: Optional[dict]) -> Optional[str]:
    """Why these shards cannot be written chunk by chunk here, or None if they can."""
    if meta is None:
        return "not a local zarr v3 array"
    codecs = meta.get("codecs") or []
    if len(codecs) != 1 or codecs[0].get("name") != "sharding_indexed":
        return "not sharded (one sharding_indexed codec)"
    conf = codecs[0].get("configuration", {})
    if conf.get("index_location", "end") != "end":
        return "shard index at the start"
    inner = [c.get("name") for c in conf.get("codecs", [])]
    if not inner or inner[0] != "bytes" or len(inner) > 2 or (
            len(inner) == 2 and inner[1] not in _COMPRESSORS):
        return f"inner codecs {inner} (supported: bytes + one of {list(_COMPRESSORS)})"
    index = [c.get("name") for c in conf.get("index_codecs", [])]
    if index != ["bytes", "crc32c"]:
        return f"index codecs {index} (supported: bytes + crc32c)"
    if (meta.get("chunk_key_encoding") or {}).get("name", "default") not in ("default", "v2"):
        return "chunk key encoding"
    try:
        import numcodecs
        numcodecs.CRC32C()
    except Exception:                                   # noqa: BLE001
        return "numcodecs without crc32c"
    return None


def supported(path: str) -> Optional[str]:
    """None when the array at ``path`` can be written chunk by chunk, else why not."""
    return _why_not(_metadata(path))


def _compressor(codec: dict):
    import numcodecs
    name, conf = codec["name"], codec.get("configuration", {})
    if name == "blosc":
        shuffle = {"noshuffle": numcodecs.Blosc.NOSHUFFLE, "shuffle": numcodecs.Blosc.SHUFFLE,
                   "bitshuffle": numcodecs.Blosc.BITSHUFFLE}[conf.get("shuffle", "noshuffle")]
        return numcodecs.Blosc(cname=conf.get("cname", "lz4"), clevel=int(conf.get("clevel", 5)),
                               shuffle=shuffle, blocksize=int(conf.get("blocksize", 0)))
    if name == "zstd":
        return numcodecs.Zstd(level=int(conf.get("level", 0)),
                              checksum=bool(conf.get("checksum", False)))
    return numcodecs.GZip(level=int(conf.get("level", 5)))


class _Shard:
    __slots__ = ("file", "offset", "index", "remaining", "lock")

    def __init__(self, n_inner, remaining):
        self.file = None                      # opened at the first chunk with data
        self.offset = 0
        self.index = np.full(n_inner + (2,), _MISSING, dtype="<u8")
        self.remaining = remaining
        self.lock = threading.Lock()


class ShardAssembler:
    """``append_chunk(region, data)`` writes one whole inner chunk into its shard's file;
    the shard is finished (index written, file closed) when its last chunk is in."""

    def __init__(self, path: str):
        meta = _metadata(path)
        why = _why_not(meta)
        if why is not None:
            raise ValueError(f"{path!r}: cannot write its shards chunk by chunk: {why}")
        conf = meta["codecs"][0]["configuration"]
        self.path = path
        self.shape = tuple(int(s) for s in meta["shape"])
        self.shard = tuple(int(s) for s in meta["chunk_grid"]["configuration"]["chunk_shape"])
        self.inner = tuple(int(s) for s in conf["chunk_shape"])
        self.n_inner = tuple(s // c for s, c in zip(self.shard, self.inner))
        endian = conf["codecs"][0].get("configuration", {}).get("endian", "little")
        self.dtype = np.dtype(meta["data_type"]).newbyteorder("<" if endian == "little" else ">")
        self.fill = np.array(meta.get("fill_value", 0) or 0, dtype=self.dtype)
        self.compressor = _compressor(conf["codecs"][1]) if len(conf["codecs"]) == 2 else None
        import numcodecs
        self._crc = numcodecs.CRC32C()
        enc = meta.get("chunk_key_encoding") or {"name": "default"}
        self._v2_keys = enc.get("name") == "v2"
        self._sep = (enc.get("configuration") or {}).get("separator",
                                                          "." if self._v2_keys else "/")
        self._shards: dict = {}
        self._lock = threading.Lock()

    @property
    def write_unit(self):
        return self.shard

    def _file_path(self, shard_idx) -> str:
        key = self._sep.join(str(i) for i in shard_idx)
        if not self._v2_keys:
            key = "c" + self._sep + key if key else "c"
        return os.path.join(self.path, *key.split("/"))

    def _state(self, shard_idx) -> _Shard:
        with self._lock:
            state = self._shards.get(shard_idx)
            if state is None:
                remaining = 1
                for i, S, c, n in zip(shard_idx, self.shard, self.inner, self.shape):
                    remaining *= -(-(min((i + 1) * S, n) - i * S) // c)
                state = self._shards[shard_idx] = _Shard(self.n_inner, remaining)
            return state

    def append_chunk(self, region, data) -> None:
        start = tuple(int(r.start or 0) for r in region)
        shard_idx = tuple(s // S for s, S in zip(start, self.shard))
        inner_idx = tuple((s - i * S) // c
                          for s, i, S, c in zip(start, shard_idx, self.shard, self.inner))
        chunk = np.asarray(data)
        if chunk.shape != self.inner:                   # an edge chunk: padded to full size
            full = np.full(self.inner, self.fill, dtype=chunk.dtype)
            full[tuple(slice(0, n) for n in chunk.shape)] = chunk
            chunk = full
        encoded = None
        if not np.all(chunk == self.fill):              # empty chunks are not written
            raw = np.ascontiguousarray(chunk, dtype=self.dtype)
            encoded = raw.tobytes() if self.compressor is None else bytes(
                self.compressor.encode(raw))
        state = self._state(shard_idx)
        with state.lock:
            if encoded is not None:
                if state.file is None:
                    target = self._file_path(shard_idx)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    state.file = open(target, "wb")
                state.file.seek(state.offset)
                state.file.write(encoded)
                state.index[inner_idx] = (state.offset, len(encoded))
                state.offset += len(encoded)
            state.remaining -= 1
            if state.remaining == 0:
                self._finish(shard_idx, state)

    def _finish(self, shard_idx, state) -> None:
        """Write the index and close (caller holds ``state.lock``)."""
        if state.file is not None:
            state.file.seek(state.offset)
            state.file.write(bytes(np.asarray(self._crc.encode(state.index.tobytes()))))
            state.file.close()
            state.file = None
        with self._lock:
            self._shards.pop(shard_idx, None)

    def finish(self) -> None:
        """Finish every shard still open (chunks never written read as the fill value)."""
        with self._lock:
            open_shards = list(self._shards.items())
        for shard_idx, state in open_shards:
            with state.lock:
                self._finish(shard_idx, state)
