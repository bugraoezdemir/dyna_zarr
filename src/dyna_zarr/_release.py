"""Keep a write's memory near ``memory_budget_mb`` whatever the allocator (Workplan
decision DR-5).

glibc gives every thread an arena of its own (up to 8 per core) and keeps what a thread
frees in that arena; after the first large buffer is freed it also raises its mmap
threshold (up to 32 MB), so region buffers of a few MB come from the arenas and stay in
the process once freed. A buffer freed by one thread then cannot serve another. Measured
on Linux (16 cores, a 128 MiB uint16 copy, memory_budget_mb=64): the process grew by
110 MB with 4 threads and 199 MB with 16.

Two things bring it back near the budget (89 -> 66 MB at 4 threads, 199 -> 67-81 MB at
16, at the same speed):

- :func:`limit_arenas`, once per process at the first write: at most ``ARENAS`` arenas,
  so the threads share freed buffers. A ``MALLOC_ARENA_MAX`` set in the environment wins.
  Fixing the mmap threshold instead kept memory as low but made a 16-thread copy 1.5x
  slower (every region buffer a fresh mmap).
- :func:`release` after each region a worker has written: the free pages of every arena
  go back to the system (``malloc_trim``). On macOS the default zone keeps freed large
  blocks the same way (``malloc_zone_pressure_relief``).

Windows' heap gives large blocks back on its own; musl does too. Elsewhere nothing here
does anything.
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import threading
import time

#: glibc arenas once dyna writes (MALLOC_ARENA_MAX in the environment wins)
ARENAS = 2
_M_ARENA_MAX = -8                            # mallopt parameter, glibc's malloc.h


def _glibc():
    try:
        if sys.platform.startswith("linux") and platform.libc_ver()[0] == "glibc":
            return ctypes.CDLL(None)
    except OSError:
        pass
    return None


def _load():
    try:
        libc = _glibc()
        if libc is not None:
            trim = libc.malloc_trim
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            return lambda: trim(0)
        if sys.platform == "darwin":
            relief = ctypes.CDLL("/usr/lib/libSystem.B.dylib").malloc_zone_pressure_relief
            relief.argtypes, relief.restype = [ctypes.c_void_p, ctypes.c_size_t], ctypes.c_size_t
            return lambda: relief(None, 0)
    except (OSError, AttributeError):
        pass
    return None


_RELEASE = _load()
_arenas_lock = threading.Lock()
_arenas_done = False


#: at most one release per this many seconds, by whichever worker comes first: trimming
#: after every region made a 16-thread float32 write 1.3x slower (0.54 -> 0.69 s)
RELEASE_EVERY_S = 0.05
_release_lock = threading.Lock()
_last_release = float("-inf")


def release() -> None:
    """Hand the memory this process has freed back to the system (no-op where the
    allocator does so itself, or cannot be asked). At most every RELEASE_EVERY_S; a
    worker that finds another one releasing goes on."""
    global _last_release
    if _RELEASE is None or time.monotonic() - _last_release < RELEASE_EVERY_S:
        return
    if not _release_lock.acquire(blocking=False):
        return
    try:
        _RELEASE()
        _last_release = time.monotonic()
    finally:
        _release_lock.release()


def limit_arenas() -> None:
    """At most :data:`ARENAS` glibc arenas from now on, once per process (threads then
    share what others freed). No-op off glibc and when ``MALLOC_ARENA_MAX`` is set."""
    global _arenas_done
    if _arenas_done:
        return
    with _arenas_lock:
        if _arenas_done:
            return
        _arenas_done = True
        if "MALLOC_ARENA_MAX" in os.environ:
            return
        libc = _glibc()
        if libc is None:
            return
        try:
            libc.mallopt.argtypes, libc.mallopt.restype = [ctypes.c_int, ctypes.c_int], ctypes.c_int
            libc.mallopt(_M_ARENA_MAX, ARENAS)
        except AttributeError:
            pass


# at import, before the threads that read sources make arenas of their own: set at the
# first write, a 16-thread copy still grew by 103 MB instead of 67-81
limit_arenas()
