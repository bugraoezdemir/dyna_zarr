"""dyna-zarr's native engine: the package ``dyna-zarr-core`` (module ``dyna_zarr_core``).

The engine is written in Rust on zarrs (Workplan WP15, decision DR-1). dyna hands
work to it step by step; its own Python path stays as the reference that the
tests compare with. This module finds the engine and decides whether it is used.

Which path runs, first match wins:

1. ``set_engine(True)`` / ``set_engine(False)`` in this process (``set_engine(None)``
   forgets the choice);
2. the environment variable ``DYNA_ZARR_ENGINE``: ``0`` / ``off`` / ``false`` for
   the Python path, ``1`` / ``on`` / ``true`` to require the engine;
3. otherwise the engine whenever a compatible one is installed.

Requiring the engine when it is missing, or installed in a version this dyna was
not built for, raises :class:`EngineUnavailableError` naming the fix. Without a
requirement, dyna quietly uses the Python path.
"""
from __future__ import annotations

import importlib
import os
from typing import Optional

#: the environment variable that switches the engine on or off
ENV = "DYNA_ZARR_ENGINE"
#: the dyna-zarr-core releases this dyna works with: at least the first, below the second
CORE_RANGE = ((0, 0, 1), (0, 1, 0))
#: what to install
FIX = 'pip install "dyna-zarr-core>=0.0.1,<0.1"'

_ON = {"1", "on", "true", "yes"}
_OFF = {"0", "off", "false", "no"}

_module = None
_problem: Optional[str] = None       # why the engine cannot be used, once looked for
_looked = False
_choice: Optional[bool] = None       # set_engine()


class EngineUnavailableError(ImportError):
    """The native engine was required but cannot be used."""


def _release(version: str) -> tuple:
    """The numeric release of a version string: '0.0.1' -> (0, 0, 1); a pre-release
    or local suffix is dropped, so '0.1.0rc1' -> (0, 1, 0)."""
    parts = []
    for piece in version.split(".")[:3]:
        digits = ""
        for ch in piece:
            if not ch.isdigit():
                break
            digits += ch
        parts.append(int(digits or 0))
    return tuple(parts + [0] * (3 - len(parts)))


def _find():
    """Import the engine once and check its version."""
    global _module, _problem, _looked
    if _looked:
        return _module
    _looked = True
    try:
        module = importlib.import_module("dyna_zarr_core")
    except ImportError as e:
        _problem = f"dyna-zarr-core is not installed ({e})"
        return None
    version = getattr(module, "__version__", "0")
    low, high = CORE_RANGE
    if not (low <= _release(version) < high):
        _problem = (f"dyna-zarr-core {version} is installed, but this dyna-zarr works "
                    f"with {'.'.join(map(str, low))} up to below {'.'.join(map(str, high))}")
        return None
    _module = module
    return module


def available() -> bool:
    """True when a compatible engine is installed (whether or not it is switched on)."""
    return _find() is not None


def version() -> Optional[str]:
    """The installed engine's version, or None when it cannot be used."""
    module = _find()
    return None if module is None else module.__version__


def _requested() -> Optional[bool]:
    """True / False when the engine is explicitly switched on / off, None for auto."""
    if _choice is not None:
        return _choice
    value = os.environ.get(ENV, "").strip().lower()
    if value in _ON:
        return True
    if value in _OFF:
        return False
    if value:
        raise ValueError(f"{ENV}={os.environ[ENV]!r}: use 1 / on / true or 0 / off / false")
    return None


def _unavailable() -> EngineUnavailableError:
    return EngineUnavailableError(f"dyna-zarr's native engine was required, but {_problem}. "
                                  f"Install it with: {FIX}  (or set {ENV}=0 for the Python path)")


def enabled() -> bool:
    """True when dyna uses the engine. Raises EngineUnavailableError when the engine
    is required (set_engine(True) or DYNA_ZARR_ENGINE=1) but cannot be used."""
    wanted = _requested()
    if wanted is False:
        return False
    if _find() is not None:
        return True
    if wanted:
        raise _unavailable()
    return False


def engine():
    """The engine module when dyna uses it, else None (the Python path)."""
    return _module if enabled() else None


def set_engine(on: Optional[bool]) -> None:
    """Use the engine (True), the Python path (False), or decide from the environment
    and what is installed (None). True raises at once if the engine cannot be used."""
    global _choice
    if on is True and _find() is None:
        raise _unavailable()
    _choice = None if on is None else bool(on)
