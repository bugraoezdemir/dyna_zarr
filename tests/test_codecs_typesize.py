"""Blosc ``typesize`` must match the array dtype (Zarr v3), for every dtype and format.

Both blosc shuffle modes permute *within* elements - byte shuffle groups the Nth byte of
every element, bitshuffle does the same per bit - so the element width IS the transform.
Zarr v2 does not store it (blosc infers it from the buffer at encode time), but v3
serializes ``typesize`` into the array metadata. dyna's ``to_v3_config()`` used to omit it,
so every v3 blosc write recorded ``typesize: 1`` and the shuffle degenerated into a
byte-wise permutation of the raw stream: measurably bigger output (12.9 -> 10.4 MiB on a
512^3 int32 label volume) and slower writes (0.83s -> 0.71s).
"""

import json
import os

import numpy as np
import pytest
import zarr

from dyna_zarr import Codecs
from dyna_zarr.io import io


DTYPES = ["uint8", "uint16", "int32", "uint32", "float32", "float64"]


def _blosc_meta(path, zarr_format):
    name = ".zarray" if zarr_format == 2 else "zarr.json"
    meta = json.load(open(os.path.join(path, name)))
    if zarr_format == 2:
        return meta.get("compressor")
    return [c for c in meta["codecs"] if c["name"] == "blosc"][0]["configuration"]


def _source(tmp_path, dtype, shape=(32, 32, 32), chunks=(16, 16, 16)):
    p = tmp_path / f"src_{dtype}.zarr"
    data = (np.arange(int(np.prod(shape))) % 97).reshape(shape).astype(dtype)
    z = zarr.create_array(store=str(p), shape=shape, chunks=chunks, dtype=dtype)
    z[:] = data
    return io.read(str(p)), data


@pytest.mark.parametrize("dtype", DTYPES)
def test_v3_typesize_matches_dtype(tmp_path, dtype):
    """v3 metadata must record the dtype's itemsize, not the old hardcoded 1."""
    src, data = _source(tmp_path, dtype)
    out = tmp_path / f"o3_{dtype}.zarr"
    io.write(src, str(out), chunks=(16, 16, 16), zarr_format=3,
             compressor=Codecs("blosc", cname="zstd", clevel=5, shuffle=2))
    cfg = _blosc_meta(str(out), 3)
    assert cfg["typesize"] == np.dtype(dtype).itemsize
    np.testing.assert_array_equal(zarr.open_array(str(out), mode="r")[...], data)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_roundtrip_both_formats(tmp_path, dtype, zarr_format):
    """Data must survive the round trip in BOTH zarr formats for every dtype."""
    src, data = _source(tmp_path, dtype)
    out = tmp_path / f"rt_{dtype}_{zarr_format}.zarr"
    io.write(src, str(out), chunks=(16, 16, 16), zarr_format=zarr_format,
             compressor=Codecs("blosc", cname="zstd", clevel=5, shuffle=2))
    np.testing.assert_array_equal(zarr.open_array(str(out), mode="r")[...], data)


def test_v2_does_not_carry_typesize(tmp_path):
    """v2 stores no typesize - blosc derives it from the buffer, so it was never buggy."""
    src, _ = _source(tmp_path, "int32")
    out = tmp_path / "o2.zarr"
    io.write(src, str(out), chunks=(16, 16, 16), zarr_format=2,
             compressor=Codecs("blosc", cname="zstd", clevel=5, shuffle=2))
    assert "typesize" not in _blosc_meta(str(out), 2)


def test_explicit_typesize_overrides_dtype(tmp_path):
    """An explicitly passed typesize wins over the dtype-derived default."""
    src, data = _source(tmp_path, "int32")
    out = tmp_path / "explicit.zarr"
    io.write(src, str(out), chunks=(16, 16, 16), zarr_format=3,
             compressor=Codecs("blosc", cname="zstd", clevel=5, shuffle=2, typesize=2))
    assert _blosc_meta(str(out), 3)["typesize"] == 2
    np.testing.assert_array_equal(zarr.open_array(str(out), mode="r")[...], data)


def test_dtype_conversion_uses_output_dtype(tmp_path):
    """typesize must follow the WRITTEN dtype, not the source's."""
    src, _ = _source(tmp_path, "int32")
    out = tmp_path / "conv.zarr"
    io.write(src, str(out), chunks=(16, 16, 16), zarr_format=3, dtype=np.uint16,
             compressor=Codecs("blosc", cname="zstd", clevel=5, shuffle=2))
    assert _blosc_meta(str(out), 3)["typesize"] == 2


def test_to_v3_config_without_dtype_is_safe():
    """Called with no dtype it must still emit a valid (fallback) typesize."""
    cfg = Codecs("blosc", cname="zstd", clevel=5, shuffle=2).to_v3_config()
    blosc = [c for c in cfg if c["name"] == "blosc"][0]["configuration"]
    assert blosc["typesize"] >= 1


def test_bitshuffle_typesize_improves_compression(tmp_path):
    """The point of the fix: correct typesize compresses label data better."""
    shape, chunks = (64, 64, 64), (32, 32, 32)
    data = (np.arange(int(np.prod(shape))) // 61).reshape(shape).astype(np.int32)
    p = tmp_path / "s.zarr"
    z = zarr.create_array(store=str(p), shape=shape, chunks=chunks, dtype="int32")
    z[:] = data
    src = io.read(str(p))

    sizes = {}
    for tag, ts in (("wrong", 1), ("correct", None)):
        out = tmp_path / f"c_{tag}.zarr"
        io.write(src, str(out), chunks=chunks, zarr_format=3,
                 compressor=Codecs("blosc", cname="zstd", clevel=5, shuffle=2, typesize=ts))
        sizes[tag] = sum(os.path.getsize(os.path.join(d, f))
                         for d, _, fs in os.walk(str(out)) for f in fs)
        np.testing.assert_array_equal(zarr.open_array(str(out), mode="r")[...], data)
    assert sizes["correct"] < sizes["wrong"]
