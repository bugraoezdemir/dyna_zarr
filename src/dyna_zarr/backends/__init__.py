"""Optional storage backends.

The default path (tensorstore + zarr-python) lives in ``io.py``; this package holds
alternatives that are opt-in because they carry an extra dependency. Importing this
package never imports the optional dependency itself -- ask for the backend by name
and handle the ``ImportError``, or check the ``*_AVAILABLE`` flag.
"""

from .zarrista_backend import ZARRISTA_AVAILABLE

__all__ = ["ZARRISTA_AVAILABLE", "zarrista_backend"]
