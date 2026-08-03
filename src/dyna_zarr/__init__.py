"""
dyna_zarr: Lightweight lazy operations on Zarr arrays without task graph overhead.

A thin layer around zarr.Array that enables lazy/dynamic array processing,
similar to dask.array but without the task graph overhead, optimized for
large-scale (terabyte+) Zarr datasets.

Includes efficient TIFF reading via tifffile's zarr bridge with concurrent access support.
"""

__version__ = "0.0.1"
__author__ = "EuBI-Biohub"

# Import core classes
from .dynamic_array import DynamicArray, slice_array
from .codecs import Codecs

# Import operations (now a package/module of flat ops) and io namespaces
from . import operations
from .io import io

# Optional: expose tifffile utilities if available
try:
    from .io import read_file
    from .tiff_reader import read_tiff_lazy, open_tiff_zarr
    __all__ = [
        "DynamicArray",
        "operations",
        "io",
        "slice_array",
        "Codecs",
        "read_file",
        "read_tiff_lazy",
        "open_tiff_zarr",
    ]
except ImportError:
    __all__ = [
        "DynamicArray",
        "operations",
        "io",
        "slice_array",
        "Codecs",
    ]
