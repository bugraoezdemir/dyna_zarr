"""Shards written chunk by chunk (``_shard_writer``), for backend='tensorstore'.

A TensorStore transaction holds a whole shard until it commits (844 MiB per thread for a
960^3 uint8 shard). dyna writes the shard file itself instead: each inner chunk is encoded
and appended as it completes, the index written when the shard's last chunk is in. Every
reader (TensorStore, zarr-python, zarrista) must read the result exactly.
"""
import importlib
import itertools
import os
import random
import re

import numpy as np
import pytest
import tensorstore as ts
import zarr

import dyna_zarr
from dyna_zarr.codecs import Codecs
from dyna_zarr.io import io
from dyna_zarr._shard_writer import ShardAssembler, supported

dio = importlib.import_module("dyna_zarr.io")

SHAPE, INNER, COEF = (37, 70, 90), (8, 16, 16), (2, 2, 3)


def _created(path, compressor, dtype="u2"):
    """An empty sharded array, created by TensorStore through io.write."""
    with io.capture_messages():
        io.write(dyna_zarr.from_array(np.zeros(SHAPE, dtype)), str(path), chunks=INNER,
                 shard_coefficients=COEF, zarr_format=3, compressor=compressor)
    for root, _, files in os.walk(os.path.join(path, "c")):
        for f in files:
            os.remove(os.path.join(root, f))
    return str(path)


def _boxes(shape=SHAPE, inner=INNER):
    return [tuple(slice(i * c, min((i + 1) * c, n)) for i, c, n in zip(idx, inner, shape))
            for idx in itertools.product(*[range(-(-n // c)) for n, c in zip(shape, inner)])]


def _files(path):
    return sorted(os.path.relpath(os.path.join(r, f), path)
                  for r, _, fs in os.walk(os.path.join(path, "c")) for f in fs)


@pytest.mark.parametrize("compressor", [None, Codecs("blosc", cname="zstd", shuffle=2),
                                        Codecs("zstd", clevel=3), Codecs("gzip", clevel=5),
                                        Codecs(None)],
                         ids=["blosc-default", "blosc-bitshuffle", "zstd", "gzip", "raw"])
def test_any_order_round_trips_through_every_reader(tmp_path, compressor):
    data = np.random.default_rng(1).integers(0, 60000, SHAPE, dtype="u2")
    data[:8, :16, :16] = 0                                     # one empty chunk
    path = _created(tmp_path / "a.zarr", compressor)
    assert supported(path) is None
    writer = ShardAssembler(path)
    boxes = _boxes()
    random.Random(0).shuffle(boxes)                            # any order, edges included
    for b in boxes:
        writer.append_chunk(b, data[b])
    writer.finish()
    spec = {"driver": "zarr3", "kvstore": {"driver": "file", "path": path}}
    np.testing.assert_array_equal(ts.open(spec).result().read().result(), data)
    np.testing.assert_array_equal(zarr.open_array(path, mode="r")[...], data)
    if dio._bounded_partial_writes() or importlib.util.find_spec("zarrista"):
        np.testing.assert_array_equal(io.read(path, backend="zarrista")[...].compute(), data)


def test_empty_chunks_and_shards_are_not_written(tmp_path):
    data = np.zeros(SHAPE, "u2")
    data[33:, 65:, 80:] = 7                                    # only the last shard has data
    path = _created(tmp_path / "a.zarr", None)
    writer = ShardAssembler(path)
    for b in _boxes():
        writer.append_chunk(b, data[b])
    writer.finish()
    assert _files(path) == [os.path.join("c", "2", "2", "1")]   # one shard file
    np.testing.assert_array_equal(zarr.open_array(path, mode="r")[...], data)


def test_metadata_it_does_not_write_is_refused(tmp_path):
    path = str(tmp_path / "plain.zarr")
    zarr.create_array(path, shape=(4, 4), chunks=(2, 2), dtype="u1", zarr_format=3)
    assert "not sharded" in supported(path)
    with pytest.raises(ValueError, match="not sharded"):
        ShardAssembler(path)
    assert "not a local zarr v3 array" in supported(str(tmp_path / "missing.zarr"))


# --- write_levels with backend='tensorstore' --------------------------------------------------

LEVELS = [(1, 1, 1), (2, 2, 2), (4, 4, 4)]


def _levels_out(tmp_path, shard_coefficients=(2, 4, 4)):
    return [dict(path=str(tmp_path / f"L{k}.zarr"), factors=f, chunks=(8, 16, 16),
                 shard_coefficients=shard_coefficients) for k, f in enumerate(LEVELS)]


def _source(tmp_path):
    data = np.random.default_rng(3).integers(0, 250, (48, 128, 160), dtype="uint8")
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=data.shape, chunks=(8, 32, 32),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    return data, io.read(str(tmp_path / "src.zarr"))


def _expected(data, f, method):
    if method == "simple":
        return data[tuple(slice(None, None, k) for k in f)]
    return ts.downsample(ts.array(data), list(f), method=method).read().result()


@pytest.mark.parametrize("method", ["simple", "mean"])
def test_tensorstore_big_shards_are_appended_chunk_by_chunk(tmp_path, method):
    data, src = _source(tmp_path)
    outs = _levels_out(tmp_path)
    with io.capture_messages() as messages:
        n = io.write_levels(src, outs, method=method, region_size_mb=0.02, zarr_format=3)
    assert n == len(outs)
    assert any("appended to their shards" in m for m in messages), messages
    peak = float(re.search(r"peaked at ([0-9.]+) MiB",
                           next(m for m in messages if "peaked at" in m)).group(1))
    shard_mib = 16 * 64 * 64 / 2**20                           # one level-0 shard
    assert peak < 4 * shard_mib                                # never shards per thread
    for o in outs:
        np.testing.assert_array_equal(zarr.open_array(o["path"])[...],
                                      _expected(data, o["factors"], method))


def test_tensorstore_transactions_remain_where_shards_cannot_be_written(tmp_path, monkeypatch):
    # e.g. a remote output or a codec the shard writer does not encode
    monkeypatch.setattr(dio, "_shard_assembly_ok", lambda specs: False)
    data, src = _source(tmp_path)
    outs = _levels_out(tmp_path)
    with io.capture_messages() as messages:
        io.write_levels(src, outs, method="simple", region_size_mb=0.02, zarr_format=3)
    assert not any("appended to their shards" in m for m in messages)
    assert any("transaction" in m for m in messages), messages
    for o in outs:
        np.testing.assert_array_equal(zarr.open_array(o["path"])[...],
                                      _expected(data, o["factors"], "simple"))


def test_tensorstore_tiles_within_the_budget(tmp_path):
    data, src = _source(tmp_path)
    outs = _levels_out(tmp_path, shard_coefficients=(2, 8, 10))   # 1 shard across a plane
    with io.capture_messages() as messages:
        io.write_levels(src, outs, method="mean", region_size_mb=0.02, zarr_format=3,
                        max_threads=4, memory_budget_mb=1.5)
    tiled = next(m for m in messages if "outputs in one pass" in m)
    assert "tiles of" in tiled, tiled
    est = float(re.search(r"open chunks ([0-9.]+) \+", tiled).group(1))
    peak = float(re.search(r"peaked at ([0-9.]+) MiB",
                           next(m for m in messages if "peaked at" in m)).group(1))
    assert peak <= est + 0.5, (peak, est)
    for o in outs[:sum("outputs in one pass" in m for m in messages) and len(outs)]:
        a = zarr.open_array(o["path"])
        if a.nchunks_initialized:
            np.testing.assert_array_equal(a[...], _expected(data, o["factors"], "mean"))


# --- io.write: shards larger than a region ---------------------------------------------------

def _big_shard_kw(**kw):
    return dict(chunks=(8, 16, 16), shard_coefficients=(2, 8, 10), zarr_format=3,
                region_size_mb=0.02, **kw)


@pytest.mark.parametrize("backend", ["tensorstore", "zarrista"])
def test_io_write_big_shards_chunk_by_chunk(tmp_path, backend):
    if backend == "zarrista" and not dio._bounded_partial_writes():
        pytest.skip("needs zarrista-bounded")
    data, src = _source(tmp_path)
    out = str(tmp_path / f"out_{backend}.zarr")
    with io.capture_messages() as messages:
        io.write(src, out, backend=backend, **_big_shard_kw())
    assert any("written chunk by chunk" in m for m in messages), messages
    assert any("appended to their shards" in m for m in messages)
    np.testing.assert_array_equal(zarr.open_array(out)[...], data)
    a = zarr.open_array(out)
    assert a.chunks == (8, 16, 16) and a.shards == (16, 128, 160)


def test_io_write_big_shards_keeps_the_chain_cast_and_observe(tmp_path):
    from scipy import ndimage
    from dyna_zarr import operations as ops
    data, src = _source(tmp_path)
    chain = ops.gaussian_filter(src.astype("float32"), 1.5) * 2      # a halo, a wider dtype
    ext = io.Extrema(axis=None)
    out = str(tmp_path / "out.zarr")
    with io.capture_messages() as messages:
        io.write(chain, out, dtype="uint16", observe=ext, **_big_shard_kw())
    assert any("written chunk by chunk" in m for m in messages)
    want = np.clip(np.rint(ndimage.gaussian_filter(data.astype("float32"), 1.5,
                                                   mode="reflect") * 2), 0, 65535)
    got = zarr.open_array(out)[...]
    assert got.dtype == np.uint16
    np.testing.assert_allclose(got, want, atol=1)
    lo, hi = ext.min[0], ext.max[0]
    assert lo == got.min() and hi == got.max()


def test_io_write_big_shards_tiles_within_the_budget(tmp_path):
    data, src = _source(tmp_path)
    out = str(tmp_path / "out.zarr")
    with io.capture_messages() as messages:
        io.write(src, out, max_threads=4, memory_budget_mb=0.6, **_big_shard_kw())
    tiled = next(m for m in messages if "outputs in one pass" in m)
    assert "tiles of" in tiled, tiled
    est = float(re.search(r"open chunks ([0-9.]+) \+", tiled).group(1))
    peak = float(re.search(r"peaked at ([0-9.]+) MiB",
                           next(m for m in messages if "peaked at" in m)).group(1))
    assert peak <= est + 0.5, (peak, est)
    np.testing.assert_array_equal(zarr.open_array(out)[...], data)


@pytest.mark.parametrize("why", ["region_shape", "small_shards", "v2"])
def test_io_write_keeps_its_own_loop_otherwise(tmp_path, why):
    data, src = _source(tmp_path)
    out = str(tmp_path / "out.zarr")
    kw = _big_shard_kw()
    if why == "region_shape":
        kw = dict(kw, region_size_mb=None, region_shape=(16, 128, 160))
    elif why == "small_shards":
        kw = dict(kw, region_size_mb=8)
    else:
        kw = dict(chunks=(8, 16, 16), zarr_format=2, region_size_mb=0.02)
    with io.capture_messages() as messages:
        io.write(src, out, **kw)
    assert not any("written chunk by chunk" in m for m in messages)
    np.testing.assert_array_equal(zarr.open_array(out)[...], data)
