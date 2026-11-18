"""The engine's lazy arrays (dyna-zarr-core, Workplan WP15 / DR3) against zarr-python
and NumPy: every codec and dtype dyna reads, regions across chunk edges, missing
chunks, reads into a larger destination, NumPy-style indexing, views and zipped Zarr."""
import os
import zipfile

import numcodecs
import numpy as np
import pytest
import zarr
import zarr.codecs as zc

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")

DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
          "float16", "float32", "float64", "complex64", "complex128"]


def _data(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.random(shape) > 0.5
    if dtype.startswith("complex"):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    if dtype.startswith("float"):
        return rng.normal(scale=100, size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)


def _write(path, data, chunks, zarr_format=3, **kw):
    z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype,
                          zarr_format=zarr_format, **kw)
    z[...] = data
    return z


V3_CODECS = {
    "default": {},
    "blosc-lz4": {"compressors": zc.BloscCodec(cname="lz4", shuffle="shuffle")},
    "blosc-zstd": {"compressors": zc.BloscCodec(cname="zstd", shuffle="bitshuffle")},
    "zstd": {"compressors": zc.ZstdCodec(level=3)},
    "gzip": {"compressors": zc.GzipCodec()},
    "none": {"compressors": None},
}
V2_CODECS = {
    "blosc-lz4": {"compressors": numcodecs.Blosc(cname="lz4")},
    "zlib": {"compressors": numcodecs.Zlib()},
    "zstd": {"compressors": numcodecs.Zstd()},
    "gzip": {"compressors": numcodecs.GZip()},
    "none": {"compressors": None},
}


@pytest.mark.parametrize("codec", list(V3_CODECS))
def test_v3_codecs(tmp_path, codec):
    data = _data((5, 37, 41), "uint16")
    _write(tmp_path / "a", data, (2, 16, 16), **V3_CODECS[codec])
    a = core.Array.open(str(tmp_path / "a"))
    assert (a.shape, a.dtype, a.chunks, a.zarr_format) == ((5, 37, 41), "uint16", (2, 16, 16), 3)
    np.testing.assert_array_equal(a.read([0, 0, 0], [5, 37, 41]), data)
    np.testing.assert_array_equal(a.read([1, 15, 7], [4, 33, 40], threads=4), data[1:4, 15:33, 7:40])


@pytest.mark.parametrize("codec", list(V2_CODECS))
def test_v2_codecs(tmp_path, codec):
    data = _data((5, 37, 41), "int32")
    _write(tmp_path / "a", data, (2, 16, 16), zarr_format=2, **V2_CODECS[codec])
    a = core.Array.open(str(tmp_path / "a"))
    assert a.zarr_format == 2
    np.testing.assert_array_equal(a.read([0, 3, 2], [5, 30, 41], threads=3), data[:, 3:30, 2:])


def test_sharding(tmp_path):
    data = _data((4, 64, 96), "uint8")
    _write(tmp_path / "a", data, (1, 16, 16), shards=(2, 32, 48))
    a = core.Array.open(str(tmp_path / "a"))
    np.testing.assert_array_equal(a.read([1, 10, 20], [4, 60, 90], threads=4), data[1:, 10:60, 20:90])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_every_dtype(tmp_path, dtype, zarr_format):
    data = _data((6, 13), dtype)
    _write(tmp_path / "a", data, (4, 5), zarr_format=zarr_format)
    a = core.Array.open(str(tmp_path / "a"))
    assert a.dtype == dtype
    got = a.read([1, 2], [6, 13])
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(got, data[1:, 2:])


def test_missing_chunks_read_as_the_fill_value(tmp_path):
    z = zarr.create_array(str(tmp_path / "a"), shape=(8, 8), chunks=(4, 4), dtype="float32",
                          fill_value=-7.5)
    z[0:4, 0:4] = 1.0
    a = core.Array.open(str(tmp_path / "a"))
    got = a.read([0, 0], [8, 8])
    assert (got[:4, :4] == 1.0).all() and (got[4:, :] == -7.5).all() and (got[:, 4:] == -7.5).all()


def test_written_by_dyna_tensorstore(tmp_path):
    dz = pytest.importorskip("dyna_zarr")
    data = _data((3, 70, 90), "uint16")
    dz.io.write(dz.from_array(data), str(tmp_path / "a.zarr"))
    a = core.Array.open(str(tmp_path / "a.zarr"))
    np.testing.assert_array_equal(a.read([0, 5, 9], [3, 70, 77], threads=4), data[:, 5:, 9:77])


def test_read_into_a_larger_destination(tmp_path):
    data = _data((4, 20, 30), "float32")
    _write(tmp_path / "a", data, (2, 8, 8))
    a = core.Array.open(str(tmp_path / "a"))
    out = np.zeros((6, 25, 40), "float32")
    res = a.read([1, 2, 3], [3, 12, 23], out=out, at=[2, 5, 10], threads=2)
    assert res is out
    np.testing.assert_array_equal(out[2:4, 5:15, 10:30], data[1:3, 2:12, 3:23])
    out[2:4, 5:15, 10:30] = 0
    assert not out.any()


def test_bad_destinations_and_regions(tmp_path):
    _write(tmp_path / "a", _data((4, 4), "uint16"), (2, 2))
    a = core.Array.open(str(tmp_path / "a"))
    with pytest.raises(TypeError, match="uint16"):
        a.read([0, 0], [2, 2], out=np.zeros((2, 2), "float32"))
    with pytest.raises(ValueError, match="C-contiguous"):
        a.read([0, 0], [2, 2], out=np.zeros((2, 4), "uint16")[:, ::2])
    with pytest.raises(core.ReadError, match="not inside"):
        a.read([0, 0], [5, 2])
    with pytest.raises(core.ReadError, match="does not fit"):
        a.read([0, 0], [2, 2], out=np.zeros((2, 2), "uint16"), at=[1, 0])
    with pytest.raises(core.ReadError, match="not a Zarr array"):
        core.Array.open(str(tmp_path))


KEYS = [
    (slice(None),), (2,), (-1,), (slice(1, 4),), (slice(None, None, 2),),
    (Ellipsis, 3), (1, Ellipsis, slice(2, 9, 3)), (None, 2), (slice(None), None, slice(1, 3)),
    (Ellipsis,), (slice(5, 2),), (0, 0, 0), (slice(-3, None), slice(None, -2), slice(1, 2)),
    (slice(0, 7, 4), slice(3, 20, 5), slice(None, None, 7)),
]


@pytest.mark.parametrize("key", KEYS, ids=str)
def test_indexing_as_numpy(tmp_path, key):
    data = _data((6, 19, 23), "int16")
    _write(tmp_path / "a", data, (4, 5, 6))
    a = core.Array.open(str(tmp_path / "a"))
    np.testing.assert_array_equal(a[key], data[key])
    assert a.index(key).shape == data[key].shape


def test_views_as_numpy(tmp_path):
    d1, d2 = _data((3, 10, 12), "uint16", 1), _data((2, 10, 12), "uint16", 2)
    _write(tmp_path / "a", d1, (2, 4, 5))
    _write(tmp_path / "b", d2, (1, 10, 12))
    a, b = core.Array.open(str(tmp_path / "a")), core.Array.open(str(tmp_path / "b"))
    full = lambda v: v.read([0] * v.ndim, list(v.shape), threads=3)
    np.testing.assert_array_equal(full(a.transpose([2, 0, 1])), d1.transpose(2, 0, 1))
    np.testing.assert_array_equal(full(a.transpose()), d1.T)
    np.testing.assert_array_equal(full(a.astype("float32")), d1.astype("float32"))
    np.testing.assert_array_equal(full(a.astype("bool")), d1.astype(bool))
    np.testing.assert_array_equal(full(core.concatenate([a, b, a], axis=0)),
                                  np.concatenate([d1, d2, d1], axis=0))
    for axis in range(4):
        np.testing.assert_array_equal(full(core.stack([a, a], axis=axis)), np.stack([d1, d1], axis=axis))
    np.testing.assert_array_equal(full(a.expand_dims(3).squeeze(3)), d1)
    c = core.concatenate([a, b], axis=0)
    np.testing.assert_array_equal(c.read([2, 3, 4], [4, 9, 10], threads=2),
                                  np.concatenate([d1, d2])[2:4, 3:9, 4:10])
    assert a.transpose([2, 0, 1]).chunks == (5, 2, 4)
    with pytest.raises(core.ReadError, match="only the join axis"):
        core.concatenate([a, b.transpose([0, 2, 1])], axis=0)
    with pytest.raises(core.ReadError, match="not of length 1"):
        a.squeeze(0)


def test_zipped_zarr(tmp_path):
    data = _data((3, 30, 40), "uint16")
    folder = tmp_path / "img.ome.zarr"
    _write(folder / "0", data, (1, 16, 16))
    zpath = tmp_path / "img.ome.zarr.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(folder):
            for f in files:
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, tmp_path).replace(os.sep, "/"))
    a = core.Array.open_zip(str(zpath), "img.ome.zarr/0")
    assert a.shape == (3, 30, 40)
    np.testing.assert_array_equal(a.read([0, 2, 3], [3, 29, 37], threads=2), data[:, 2:29, 3:37])
    with pytest.raises(core.ReadError):
        core.Array.open_zip(str(zpath), "img.ome.zarr/9")


def test_metadata_is_the_stored_json(tmp_path):
    import json
    _write(tmp_path / "a", _data((4, 4), "uint8"), (2, 2), compressors=zc.ZstdCodec(level=3))
    meta = json.loads(core.Array.open(str(tmp_path / "a")).metadata)
    assert meta["shape"] == [4, 4] and meta["zarr_format"] == 3
    # a codec without configuration may be written as just its name (Zarr v3 allows both)
    names = [c if isinstance(c, str) else c["name"] for c in meta["codecs"]]
    assert "zstd" in names
    assert core.Array.open(str(tmp_path / "a")).transpose().metadata is None
