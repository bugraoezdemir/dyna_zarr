"""One write's threads (Workplan decision DR-5, task DR6b).

``max_threads`` counts every thread a write keeps busy: dyna's region workers, the
engine's threads inside a region, and TensorStore's own pools, which decode the sources
and encode the output. A write opens a :class:`WriteScope`; everything it starts runs
inside it (dyna's worker threads copy the caller's context), and three things read it:

- the output is opened in the scope's TensorStore context, whose ``data_copy_concurrency``
  and ``file_io_concurrency`` are limited to ``max_threads`` (TensorStore's default is
  every core, so ``max_threads=4`` kept about 9 cores busy on a copy);
- a TensorStore source read during the write is reopened once in that context (the
  same store and view; only its two pools change), so decoding it shares the output's
  pools;
- engine reads that use more than one thread run on the scope's one engine ``Pool`` of
  ``max_threads`` threads, shared by all of the write's regions.

Outside a write nothing changes: sources keep their own context and engine reads their
own pools. A write inside a write (a staged writer, write_levels called by io.write)
keeps the outer scope, so the outer ``max_threads`` bounds both.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

_SCOPE: ContextVar[Optional["WriteScope"]] = ContextVar("dyna_zarr_write_scope", default=None)

#: TensorStore drivers whose handles are reopened in a write's context: Zarr arrays in a
#: store (an in-memory or virtual handle is used as it is: reopening would copy its data
#: or cannot be done from its spec)
_REOPEN_DRIVERS = ("zarr", "zarr3")
_NOT_A_STORE = ("memory",)
#: TensorStore's thread pools, which a write limits to its max_threads
_POOLS = ("data_copy_concurrency", "file_io_concurrency")


class WriteScope:
    """The threads of one write: ``max_threads`` for TensorStore and the engine."""

    def __init__(self, max_threads: int):
        self.max_threads = max(1, int(max_threads))
        self._lock = threading.Lock()
        self._ts_context = None
        self._pool = None
        self._handles = {}            # id(handle) -> (handle, its reopened handle)
        #: bytes of the engine pool's chunk cache (0: none); a write over a filter sets
        #: it before its first read (see io._halo_cache_bytes)
        self.cache_bytes = 0

    def ts_context(self):
        """A TensorStore context whose data-copy and file-I/O pools have ``max_threads``
        threads; one per write, shared by its output and its sources."""
        with self._lock:
            if self._ts_context is None:
                import tensorstore as ts
                n = self.max_threads
                self._ts_context = ts.Context({"data_copy_concurrency": {"limit": n},
                                               "file_io_concurrency": {"limit": n}})
            return self._ts_context

    def ts_spec(self, spec: dict) -> dict:
        """``spec`` (a TensorStore spec as JSON) without its own two thread pools, so
        opened in :meth:`ts_context` it takes the write's (one pool for the sources and
        the output) and keeps every other context entry it has (credentials, a cache
        pool)."""
        spec = dict(spec)
        context = {k: v for k, v in (spec.get("context") or {}).items()
                   if k not in _POOLS}
        spec["context"] = context
        return spec

    def source(self, handle):
        """``handle`` (a TensorStore source) reopened in this write's limits, once per
        write; the handle itself when it cannot be (see _REOPEN_DRIVERS)."""
        key = id(handle)
        with self._lock:
            known = self._handles.get(key)
        if known is not None and known[0] is handle:
            return known[1]
        reopened = _reopen(handle, self)
        with self._lock:
            # the handle is kept with its reopened one, so its id cannot be reused
            self._handles[key] = (handle, reopened)
        return reopened

    def pool(self):
        """The engine's thread pool for this write (``max_threads`` threads), with a
        cache of decoded chunks of ``cache_bytes`` when that is set."""
        with self._lock:
            if self._pool is None:
                from . import _engine
                self._pool = _engine.engine().Pool(
                    self.max_threads, cache_mb=self.cache_bytes / 2**20)
            return self._pool

    def want_cache(self, nbytes: int) -> bool:
        """Give the engine pool a chunk cache of ``nbytes`` (the larger of what earlier
        writes in this scope asked for); False once the pool exists without one big
        enough (a write inside a write that started reading)."""
        with self._lock:
            if self._pool is not None:
                return self._pool.cache_bytes >= nbytes
            self.cache_bytes = max(self.cache_bytes, int(nbytes))
            return True


def _reopen(handle, scope: WriteScope):
    try:
        spec = handle.spec(retain_context=True).to_json()
    except Exception:                               # noqa: BLE001 - keep the handle
        return handle
    kv = spec.get("kvstore")
    kv_driver = kv.get("driver") if isinstance(kv, dict) else None
    if spec.get("driver") not in _REOPEN_DRIVERS or kv_driver in _NOT_A_STORE:
        return handle
    import tensorstore as ts
    try:
        return ts.open(scope.ts_spec(spec), read=True, context=scope.ts_context()).result()
    except Exception:                               # noqa: BLE001 - keep the handle
        return handle


def current() -> Optional[WriteScope]:
    """The write this thread works for, or None outside a write."""
    return _SCOPE.get()


@contextmanager
def write_scope(max_threads: int):
    """Run a write within ``max_threads`` threads; an outer write's scope is kept."""
    outer = _SCOPE.get()
    if outer is not None:
        yield outer
        return
    scope = WriteScope(max_threads)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def ts_source(handle):
    """``handle`` as a read inside a write should use it (see WriteScope.source)."""
    scope = _SCOPE.get()
    return handle if scope is None else scope.source(handle)
