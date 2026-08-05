"""Lazy TIFF reading via tifffile's zarr bridge.

A TIFF is opened as a lazy, chunked zarr array through ``tifffile``'s ``aszarr`` bridge and
handed straight to :class:`~dyna_zarr.DynamicArray`. All laziness, slicing, chunk-wise reads
and memory-boundedness then come from ``DynamicArray``'s pull model (``SliceTransform`` /
``_read_direct``) -- exactly as for a zarr store -- so there is no TIFF-specific slice logic
to maintain. The tifffile store stays open for as long as the returned array references it
(released on garbage collection), which is what lazy TIFF reading requires.
"""

import tifffile
import zarr


def open_tiff_zarr(path):
    """Open a TIFF as a lazy zarr array via tifffile's zarr bridge (the raw backend object).

    For a plain single-series TIFF this is a ``zarr.Array``; for a multi-series/multi-level
    file tifffile yields a group, in which case the first array (series/level 0) is returned.
    """
    store = tifffile.imread(str(path), aszarr=True)     # ZarrTiffStore (lazy; reads on access)
    obj = zarr.open(store, mode="r")
    if isinstance(obj, zarr.Group):
        for key in obj:                                 # multi-series/level -> take the first array
            item = obj[key]
            if isinstance(item, zarr.Array):
                return item
        raise ValueError(f"TIFF at {path!r} exposes no readable array via aszarr")
    return obj


def read_tiff_lazy(path):
    """Read a TIFF as a lazy, memory-bounded :class:`~dyna_zarr.DynamicArray`."""
    from .dynamic_array import DynamicArray
    return DynamicArray(open_tiff_zarr(path))
