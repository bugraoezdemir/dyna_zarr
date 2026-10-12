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


def _ts_handle(path, zarr_format):
    import tensorstore as ts
    return ts.open({"driver": "zarr3" if zarr_format == 3 else "zarr",
                    "kvstore": {"driver": "file", "path": str(path)}}).result()


def _codec(path):
    c = zarr.open_array(str(path)).compressors[0]
    cfg = c.to_dict() if hasattr(c, "to_dict") else c.get_config()
    if "configuration" in cfg:
        cfg = {"id": cfg["name"], **cfg["configuration"]}
    return cfg


def test_a_tensorstore_handle_keeps_format_codecs_and_shards(tmp_path):
    # DynamicArray(ts_array): a TensorStore handle the caller opened, not io.read
    data, _ = _src(tmp_path, compressors=BloscCodec(cname="zstd", clevel=3), shards=(8, 32, 32))
    _write(DynamicArray(_ts_handle(tmp_path / "src.zarr", 3)) + 1, str(tmp_path / "out.zarr"))
    out = zarr.open_array(str(tmp_path / "out.zarr"))
    assert out.metadata.zarr_format == 3
    assert (out.shards, out.chunks) == ((8, 32, 32), (4, 16, 16))
    assert (_codec(tmp_path / "out.zarr")["cname"], _codec(tmp_path / "out.zarr")["clevel"]) == ("zstd", 3)
    np.testing.assert_allclose(out[...], data + 1)

    _src(tmp_path, name="v2.zarr", zarr_format=2, compressors=numcodecs.Zstd(level=4))
    _write(DynamicArray(_ts_handle(tmp_path / "v2.zarr", 2)), str(tmp_path / "out2.zarr"))
    assert _fmt(tmp_path / "out2.zarr") == 2
    assert (_codec(tmp_path / "out2.zarr")["id"], _codec(tmp_path / "out2.zarr")["level"]) == ("zstd", 4)


def test_sources_without_a_format_get_v3_blosc_lz4(tmp_path):
    import tensorstore as ts
    _src(tmp_path, compressors=BloscCodec(cname="zstd", clevel=3))
    tifffile.imwrite(str(tmp_path / "t.tif"), np.ones((4, 8, 8), "u2"))
    view = ts.downsample(_ts_handle(tmp_path / "src.zarr", 3), [1, 2, 2], method="mean")
    for name, arr in [("numpy", DynamicArray(np.ones((4, 8, 8), "f4"))),
                      ("creation", ops.zeros((4, 8, 8), dtype="f4")),
                      ("tiff", io.read(str(tmp_path / "t.tif"))),
                      ("ts_view", DynamicArray(view))]:   # virtual: not its source's zstd
        _write(arr, str(tmp_path / f"{name}.zarr"))
        assert _fmt(tmp_path / f"{name}.zarr") == 3, name
        codec = _codec(tmp_path / f"{name}.zarr")
        assert (codec["id"], codec["cname"]) == ("blosc", "lz4"), name


def test_each_default_can_be_overridden_for_a_source_without_a_format(tmp_path):
    a = DynamicArray(np.ones((8, 32, 32), "f4"))
    _write(a, str(tmp_path / "v2.zarr"), zarr_format=2)
    assert _fmt(tmp_path / "v2.zarr") == 2
    _write(a, str(tmp_path / "gz.zarr"), compressor=Codecs("gzip", clevel=2))
    assert _codec(tmp_path / "gz.zarr")["id"] == "gzip"
    _write(a, str(tmp_path / "sh.zarr"), chunks=(4, 16, 16), shard_coefficients=(2, 2, 2))
    assert zarr.open_array(str(tmp_path / "sh.zarr")).shards == (8, 32, 32)


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
    dict(max_threads=0),
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
