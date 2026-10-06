"""io.write resolves every output setting once, before touching storage.

- format: the INPUT's zarr format unless zarr_format= is given; v3 for sources that have
  none (numpy, TIFF, creation ops). It used to be v3 always on the region writer and v2
  on the staged writers - and format DETECTION reported every array as v2 under zarr 3.
- codecs and shards are inherited from the input (they silently were not for v3 inputs,
  because of that misdetection); an uncompressed input stays uncompressed; a numpy
  source does NOT inherit its in-memory staging store's zarr defaults.
- a rejected write leaves nothing on disk (validation used to run partly AFTER the
  output store had been created).
- a remote URL on the TensorStore writer is refused loudly instead of being taken as a
  local path.
"""
import contextlib
import io as _stdio
import os

import numcodecs
import numpy as np
import pytest
import tifffile
import zarr
from zarr.codecs import BloscCodec

from dyna_zarr import Codecs, DynamicArray, io, operations as ops
from dyna_zarr.backends.zarrista_backend import UnsupportedByBackend


def _write(*args, **kwargs):
    buf = _stdio.StringIO()
    with contextlib.redirect_stdout(buf):
        io.write(*args, **kwargs)
    return buf.getvalue()


def _src(tmp_path, name="src.zarr", zarr_format=3, **kw):
    data = np.random.default_rng(0).random((8, 32, 32)).astype("f4")
    z = zarr.create_array(str(tmp_path / name), shape=data.shape, chunks=(4, 16, 16),
                          dtype="f4", zarr_format=zarr_format, **kw)
    z[...] = data
    return data, io.read(str(tmp_path / name))


def _fmt(path):
    return zarr.open_array(str(path)).metadata.zarr_format


# --------------------------------------------------------------------------- #
# format
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fmt", [2, 3])
def test_format_is_inherited_from_the_input(tmp_path, fmt):
    data, a = _src(tmp_path, zarr_format=fmt)
    assert a.zarr_format == fmt
    _write(a * 2, str(tmp_path / "out.zarr"))
    assert _fmt(tmp_path / "out.zarr") == fmt
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "out.zarr"))[...], data * 2)


@pytest.mark.parametrize("fmt", [2, 3])
def test_explicit_format_wins(tmp_path, fmt):
    _, a = _src(tmp_path, zarr_format=5 - fmt)          # the other format
    _write(a, str(tmp_path / "out.zarr"), zarr_format=fmt)
    assert _fmt(tmp_path / "out.zarr") == fmt


def test_sources_without_a_format_default_to_v3(tmp_path):
    tifffile.imwrite(str(tmp_path / "t.tif"), np.ones((4, 8, 8), "u2"))
    for name, arr in [("numpy", DynamicArray(np.ones((4, 8, 8), "f4"))),
                      ("creation", ops.zeros((4, 8, 8), dtype="f4")),
                      ("tiff", io.read(str(tmp_path / "t.tif")))]:
        assert arr.zarr_format is None, name
        _write(arr, str(tmp_path / f"{name}.zarr"))
        assert _fmt(tmp_path / f"{name}.zarr") == 3, name


@pytest.mark.parametrize("build", ["flatten", "reshape", "cumsum"])
def test_staged_writers_use_the_same_format_rule(tmp_path, build):
    data, a = _src(tmp_path, zarr_format=3)             # staged used to default to v2
    lazy = {"flatten": lambda: ops.flatten(a),
            "reshape": lambda: ops.reshape(a, (8, 1024)),
            "cumsum": lambda: ops.cumsum(a, axis=0)}[build]()
    _write(lazy, str(tmp_path / "out.zarr"))
    assert _fmt(tmp_path / "out.zarr") == 3
    _write(lazy, str(tmp_path / "out2.zarr"), zarr_format=2)
    assert _fmt(tmp_path / "out2.zarr") == 2


# --------------------------------------------------------------------------- #
# codecs and shards
# --------------------------------------------------------------------------- #

def test_v3_codecs_and_shards_are_inherited(tmp_path):
    data, a = _src(tmp_path, compressors=BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle"),
                   shards=(8, 32, 32))
    assert a.shards == (8, 32, 32)
    _write(a + 1, str(tmp_path / "out.zarr"))
    out = zarr.open_array(str(tmp_path / "out.zarr"))
    blosc = out.compressors[0].to_dict()["configuration"]
    assert (blosc["cname"], blosc["clevel"], blosc["shuffle"]) == ("zstd", 3, "bitshuffle")
    assert out.shards == (8, 32, 32) and out.chunks == (4, 16, 16)
    np.testing.assert_allclose(out[...], data + 1)


def test_v2_codecs_are_inherited(tmp_path):
    _, a = _src(tmp_path, zarr_format=2, compressors=numcodecs.Zstd(level=4))
    _write(a, str(tmp_path / "out.zarr"))
    comp = zarr.open_array(str(tmp_path / "out.zarr")).compressors[0]
    assert type(comp).__name__ == "Zstd" and comp.level == 4


def test_uncompressed_input_stays_uncompressed(tmp_path):
    _, a = _src(tmp_path, compressors=None)
    _write(a, str(tmp_path / "out.zarr"))
    assert zarr.open_array(str(tmp_path / "out.zarr")).compressors == ()


def test_numpy_source_does_not_inherit_its_staging_store(tmp_path):
    # The in-memory zarr a numpy array is staged through uses zarr's defaults (zstd
    # level 0); those are an implementation detail, so the write's defaults apply.
    _write(DynamicArray(np.ones((4, 8, 8), "f4")), str(tmp_path / "out.zarr"))
    blosc = zarr.open_array(str(tmp_path / "out.zarr")).compressors[0].to_dict()
    assert blosc["name"] == "blosc" and blosc["configuration"]["cname"] == "lz4"


def test_inherited_shards_that_no_longer_fit_are_dropped_with_a_note(tmp_path):
    _, a = _src(tmp_path, shards=(8, 32, 32))
    printed = _write(a, str(tmp_path / "out.zarr"), chunks=(8, 32, 24))   # 32 % 24 != 0
    assert zarr.open_array(str(tmp_path / "out.zarr")).shards is None
    assert "written unsharded" in printed


def test_explicit_compressor_and_shards_still_win(tmp_path):
    _, a = _src(tmp_path, compressors=BloscCodec(cname="zstd", clevel=3), shards=(8, 32, 32))
    _write(a, str(tmp_path / "out.zarr"), compressor=Codecs("gzip", clevel=2),
           shard_coefficients=(1, 1, 2))
    out = zarr.open_array(str(tmp_path / "out.zarr"))
    assert out.compressors[0].to_dict()["name"] == "gzip"
    assert out.shards == (4, 16, 32)


# --------------------------------------------------------------------------- #
# nothing is written unless the whole write is valid
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", [
    dict(chunks=(4, 16, 16), region_shape=(4, 16, 24)),    # region not a chunk multiple
    dict(zarr_format=4),
    dict(chunks=(4, 16)),                                   # wrong rank
    dict(max_workers=0),
    dict(device="tpu"),
    dict(zarr_format=2, shard_coefficients=(1, 1, 1)),
    dict(zarr_format=2, dimension_names=("z", "y", "x")),  # v2 has no such field
    dict(zarr_format=3, dimension_names=("y", "x")),       # wrong rank
])
def test_rejected_write_leaves_nothing_behind(tmp_path, bad):
    _, a = _src(tmp_path)
    with pytest.raises(ValueError):
        _write(a, str(tmp_path / "out.zarr"), **bad)
    assert not os.path.exists(tmp_path / "out.zarr")


def test_http_output_on_tensorstore_is_refused_not_written_locally(tmp_path, monkeypatch):
    # TensorStore's HTTP store is read-only. (s3:// and gs:// ARE written since #15 -
    # tested against an S3 emulator in test_remote_io.py; a test here must never reach
    # a real cloud endpoint.) Before, any URL was opened as a local PATH.
    monkeypatch.chdir(tmp_path)
    a = DynamicArray(np.ones((4, 8, 8), "f4"))
    with pytest.raises(UnsupportedByBackend, match="backend='zarrista'"):
        _write(a, "https://example.org/out.zarr")
    assert os.listdir(tmp_path) == []                   # no local 'https:' directory


@pytest.mark.parametrize("backend", ["tensorstore",
                                     pytest.param("zarrista", marks=pytest.mark.zarrista)])
def test_dimension_names_are_written_to_v3_metadata(tmp_path, backend):
    # a plain zarr v3 array field, written verbatim (callers such as an OME-Zarr writer
    # decide what the names must be)
    data, a = _src(tmp_path)
    out = tmp_path / "out.zarr"
    _write(a, str(out), zarr_format=3, dimension_names=("z", "y", "x"), backend=backend)
    z = zarr.open_array(str(out))
    assert z.metadata.dimension_names == ("z", "y", "x")
    np.testing.assert_array_equal(z[...], data)
