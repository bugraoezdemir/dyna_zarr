"""io.write_levels: a source and its downsampled levels from ONE read of the source.

Every output must equal what one io.write per level would store (``x[::f]`` for
'simple', ``coarsen(x, f, method)`` otherwise), every output chunk (shard) must be
written whole and exactly once, and only as many leading outputs as fit the region
budget are written - the return value says how many.
"""
import importlib
import itertools
import threading

import numcodecs
import numpy as np
import pytest
import tensorstore as ts
import zarr

import dyna_zarr
from dyna_zarr import operations as ops
from dyna_zarr.io import io

dio = importlib.import_module("dyna_zarr.io")

SHAPE = (3, 23, 41, 37)          # odd extents: partial edge windows at every level
CHUNKS = (1, 8, 16, 16)
FACTORS = [(1, 1, 1, 1), (1, 2, 2, 2), (1, 4, 4, 4), (1, 8, 8, 8)]


@pytest.fixture(scope="module")
def data():
    return np.random.default_rng(0).integers(0, 1000, size=SHAPE, dtype="uint16")


@pytest.fixture
def source(data, tmp_path):
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=CHUNKS,
                          dtype=data.dtype)
    z[...] = data
    return io.read(str(tmp_path / "src.zarr"))


def _outputs(tmp_path, factors=FACTORS, **extra):
    return [dict(path=str(tmp_path / f"L{i}.zarr"), factors=f, **extra)
            for i, f in enumerate(factors)]


def _expected(data, f, method):
    if method == "simple":
        return data[tuple(slice(None, None, k) for k in f)]
    return ts.downsample(ts.array(data), list(f), method=method).read().result()


@pytest.mark.parametrize("method", ["simple", "mean", "median", "min", "max", "mode"])
@pytest.mark.parametrize("region_size_mb", [None, 0.05])
def test_every_level_equals_its_own_write(source, data, tmp_path, method, region_size_mb):
    # the same chunks at every level: deeper levels need larger regions, so a small
    # budget splits the write into passes
    outs = _outputs(tmp_path, chunks=CHUNKS)
    done = 0
    src, src_f = source, (1, 1, 1, 1)
    while done < len(outs):            # the caller's loop: the rest from a stored level
        rel = [dict(o, factors=tuple(a // b for a, b in zip(o["factors"], src_f)))
               for o in outs[done:]]
        n = io.write_levels(src, rel, method=method, region_size_mb=region_size_mb)
        assert 1 <= n <= len(rel)
        done += n
        src, src_f = io.read(outs[0]["path"]), (1, 1, 1, 1)
    for o in outs:
        np.testing.assert_array_equal(zarr.open_array(o["path"])[...],
                                      _expected(data, o["factors"], method))


def test_returns_how_many_leading_outputs_fit(source, tmp_path):
    # default level chunks are the source's divided by the factor, so every level
    # aligns to the source grid and all of them fit
    assert io.write_levels(source, _outputs(tmp_path / "a"), region_size_mb=0.01) == 4
    # the same chunks at every level: deeper levels need larger regions; a tiny budget
    # still writes the first output
    n = io.write_levels(source, _outputs(tmp_path / "b", chunks=CHUNKS),
                        region_size_mb=0.01)
    assert 1 <= n < len(FACTORS)
    assert not (tmp_path / "b" / f"L{n}.zarr").exists()      # nothing beyond n created


@pytest.mark.parametrize("sharded", [False, True])
def test_every_write_unit_is_written_whole_and_once(source, tmp_path, monkeypatch, sharded):
    writes, lock = {}, threading.Lock()
    orig = dio._open_output

    def spy(spec):
        handle = orig(spec)
        unit = spec.shards if spec.shards is not None else spec.chunks

        class View:
            def __getitem__(self, key):
                with lock:
                    writes.setdefault(spec.path if hasattr(spec, "path") else id(spec),
                                      (spec.shape, unit, []))[2].append(key)
                return handle[key]
        return View()
    monkeypatch.setattr(dio, "_open_output", spy)
    extra = dict(shard_coefficients=(1, 2, 2, 2)) if sharded else {}
    n = io.write_levels(source, _outputs(tmp_path, **extra), region_size_mb=0.5,
                        zarr_format=3)
    assert n == len(FACTORS) and len(writes) == n
    for shape, unit, keys in writes.values():
        seen = set()
        for key in keys:
            for s, u, n_ in zip(key, unit, shape):
                assert s.start % u == 0 and (s.stop % u == 0 or s.stop == n_), (key, unit)
            cells = itertools.product(*[range(s.start // u, -(-s.stop // u))
                                        for s, u in zip(key, unit)])
            for c in cells:
                assert c not in seen, f"write unit {c} written twice"
                seen.add(c)
        grid = itertools.product(*[range(-(-n_ // u)) for n_, u in zip(shape, unit)])
        assert seen == set(grid)                  # and every one written


def test_levels_inherit_the_source_codecs_and_format(source, tmp_path):
    z = zarr.create_array(str(tmp_path / "zs.zarr"), shape=SHAPE, chunks=CHUNKS,
                          dtype="uint16", zarr_format=2,
                          compressors=numcodecs.Zstd(level=3))
    z[...] = 1
    io.write_levels(io.read(str(tmp_path / "zs.zarr")), _outputs(tmp_path))
    for i in range(len(FACTORS)):
        a = zarr.open_array(str(tmp_path / f"L{i}.zarr"))
        assert a.metadata.zarr_format == 2
        assert a.metadata.compressor.codec_id == "zstd"


def test_explicit_chunks_codecs_and_shards_per_output(source, data, tmp_path):
    outs = [dict(path=str(tmp_path / "L0.zarr"), factors=FACTORS[0], chunks=(1, 8, 32, 32),
                 shard_coefficients=(1, 2, 1, 1)),
            dict(path=str(tmp_path / "L1.zarr"), factors=FACTORS[1], chunks=(1, 4, 8, 8))]
    io.write_levels(source, outs, compressor=dyna_zarr.Codecs("zstd", clevel=5),
                    zarr_format=3, method="mean")
    a0, a1 = (zarr.open_array(o["path"]) for o in outs)
    assert a0.chunks == (1, 8, 32, 32) and a0.shards == (1, 16, 32, 32)
    assert a1.chunks == (1, 4, 8, 8) and a1.shards is None
    assert all(c.to_dict()["name"] in ("bytes", "zstd") for c in a1.metadata.codecs)
    np.testing.assert_array_equal(a1[...], _expected(data, FACTORS[1], "mean"))


def test_dtype_is_a_saturating_cast(tmp_path):
    x = np.array([[-5.6, 0.4], [300.0, np.nan]], dtype="float32")
    io.write_levels(x, [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1))], dtype="uint8")
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "L0.zarr"))[...],
                                  [[0, 0], [255, 0]])


def test_numpy_and_zarr_sources(data, tmp_path):
    for i, src in enumerate([data, zarr.array(data, chunks=CHUNKS)]):
        out = _outputs(tmp_path / str(i), FACTORS[:2])
        io.write_levels(src, out, method="max")
        np.testing.assert_array_equal(zarr.open_array(out[1]["path"])[...],
                                      _expected(data, FACTORS[1], "max"))


def test_reducer_replaces_coarsen_but_not_stride(source, tmp_path):
    calls = []

    def reducer(block, f, method):
        calls.append(method)
        return ops.downsampling.coarsen_block(block, f, method)
    io.write_levels(source, _outputs(tmp_path / "a", FACTORS[:2]), method="min",
                    reducer=reducer)
    assert calls and set(calls) == {"min"}
    calls.clear()
    io.write_levels(source, _outputs(tmp_path / "b", FACTORS[:2]), reducer=reducer)
    assert calls == []


@pytest.mark.parametrize("bad, match", [
    (dict(method="gaussian"), "method"),
    (dict(outputs=[]), "at least one output"),
    (dict(outputs=[dict(path="x", factors=(1, 2))]), "factors"),
    (dict(outputs=[dict(path="x", factors=(1, 0, 1, 1))]), "factors"),
])
def test_refusals(source, tmp_path, bad, match):
    kw = dict(outputs=_outputs(tmp_path)) | bad
    with pytest.raises(ValueError, match=match):
        io.write_levels(source, **kw)


@pytest.mark.parametrize("method", ["simple", "mean", "median", "max", "mode"])
def test_shards_larger_than_the_budget_are_read_in_blocks(data, tmp_path, monkeypatch,
                                                          method):
    # 1 GB shards at a 64 MB budget, scaled down: shards of 4x4x4 chunks, a budget of a
    # few chunks. A region (whole shards of every output) is READ in budget-sized blocks
    # and its shards are committed once; reading it whole held factor**ndim shards per
    # worker (8 GB regions -> Windows error 1450 on a 17 GB conversion).
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=(1, 4, 8, 8),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    src = io.read(str(tmp_path / "src.zarr"))
    reads, lock = [], threading.Lock()
    orig = type(src)._read_direct

    def spy(self, key):
        out = orig(self, key)
        with lock:
            reads.append(out.nbytes)
        return out
    monkeypatch.setattr(type(src), "_read_direct", spy)
    budget_mb = 0.02
    outs = _outputs(tmp_path / "o", chunks=(1, 4, 8, 8), shard_coefficients=(1, 4, 4, 4))
    done, base = 0, src
    while done < len(outs):
        rel = [dict(o, factors=tuple(a // b for a, b in zip(o["factors"], FACTORS[0])))
               for o in outs[done:]]
        n = io.write_levels(base, rel, method=method, region_size_mb=budget_mb,
                            zarr_format=3)
        done += n
        base = io.read(outs[0]["path"])
    # never a whole region: at most the budget, or one output chunk's footprint in the
    # source where that alone is larger (factor 8 x chunk 8 spans a whole test plane)
    footprint = max(int(np.prod([min(k * c, s) for k, c, s in zip(f, (1, 4, 8, 8), SHAPE)]))
                    for f in FACTORS) * data.itemsize
    assert max(reads) <= max(budget_mb * 2**20, footprint)
    whole_shard_region = int(np.prod([min(2 * 4 * c, s) for c, s in zip((1, 4, 8, 8), SHAPE)]))
    assert max(reads) < whole_shard_region * data.itemsize * 2
    for o in outs:
        a = zarr.open_array(o["path"])
        assert a.shards is not None
        np.testing.assert_array_equal(a[...], _expected(data, o["factors"], method))


@pytest.mark.parametrize("method", ["simple", "mean", "mode"])
def test_a_stack_of_planes_into_big_shards(data, tmp_path, monkeypatch, method):
    # The source's chunks are whole planes (a TIFF stack), wider than a shard: blocks are
    # whole planes (reading part of one reads all of it), and a shard - filled over many
    # blocks - is committed whenever it holds the budget. A tiny budget makes every shard
    # commit many times, from concurrent workers.
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=(1, 1, 41, 37),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    src = io.read(str(tmp_path / "src.zarr"))
    reads, lock = [], threading.Lock()
    orig = type(src)._read_direct

    def spy(self, key):
        with lock:
            reads.append(key)
        return orig(self, key)
    monkeypatch.setattr(type(src), "_read_direct", spy)
    outs = _outputs(tmp_path / "o", chunks=(1, 4, 8, 8), shard_coefficients=(1, 2, 2, 2))
    with io.capture_messages() as messages:
        assert io.write_levels(src, outs, method=method, region_size_mb=0.002,
                               zarr_format=3) == len(outs)
    assert any("whole source chunks" in m for m in messages)
    assert all(k[2] == slice(0, 41) and k[3] == slice(0, 37) for k in reads)  # whole planes
    for o in outs:
        np.testing.assert_array_equal(zarr.open_array(o["path"])[...],
                                      _expected(data, o["factors"], method))


def _bounded():
    return dio._bounded_partial_writes()


@pytest.mark.zarrista
@pytest.mark.parametrize("planes", [False, True], ids=["3d-chunks", "planes"])
@pytest.mark.parametrize("method", ["simple", "mean", "mode"])
def test_big_shards_appended_chunk_by_chunk(data, tmp_path, planes, method):
    # zarrista-bounded: shards larger than the budget are filled by appending whole inner
    # chunks (zarrs partial encoding), all levels in one pass. Every level must match, and
    # no shard may hold dead space (a partial chunk written twice would leave its old copy).
    if not _bounded():
        pytest.skip("needs zarrista-bounded (in-place shard appends)")
    import zarrista
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE,
                          chunks=(1, 1, 41, 37) if planes else (1, 4, 8, 8),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    outs = _outputs(tmp_path / "o", chunks=(1, 4, 8, 8), shard_coefficients=(1, 4, 4, 4))
    with io.capture_messages() as messages:
        n = io.write_levels(io.read(str(tmp_path / "src.zarr")), outs, method=method,
                            region_size_mb=0.002, zarr_format=3, backend="zarrista")
    assert n == len(outs)
    assert any("appended to their shards" in m for m in messages), messages
    for o in outs:
        a = zarr.open_array(o["path"])
        np.testing.assert_array_equal(a[...], _expected(data, o["factors"], method))
        za = zarrista.Array.open(zarrista.store.FilesystemStore(o["path"]), "/")
        grid = [range(g) for g in za.chunk_grid_shape]
        assert not any(za.compact_chunk(list(idx)) for idx in itertools.product(*grid))


@pytest.mark.zarrista
def test_big_shards_stay_whole_shard_without_bounded_zarrista(data, tmp_path, monkeypatch):
    # upstream zarrista would rewrite the whole shard per append: the mode is not used
    monkeypatch.setattr(dio, "_bounded_partial_writes", lambda: False)
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=SHAPE, chunks=(1, 4, 8, 8),
                          dtype=data.dtype, zarr_format=3)
    z[...] = data
    outs = _outputs(tmp_path / "o", FACTORS[:2], chunks=(1, 4, 8, 8),
                    shard_coefficients=(1, 4, 4, 4))
    with io.capture_messages() as messages:
        io.write_levels(io.read(str(tmp_path / "src.zarr")), outs, region_size_mb=0.002,
                        zarr_format=3, backend="zarrista")
    assert not any("appended to their shards" in m for m in messages)


def test_dask_is_refused(data, tmp_path):
    da = pytest.importorskip("dask.array")
    with pytest.raises(ValueError, match="from_array"):
        io.write_levels(da.from_array(data), _outputs(tmp_path))


# --- SliceTransform.read_amplification: strided levels are budgeted by what they read ---

@pytest.mark.parametrize("key, amp", [
    ((slice(None),) * 4, 1),
    ((slice(None), slice(None, None, 2), slice(None, None, 2), slice(None, None, 2)), 8),
    ((0, slice(None, None, 4), slice(None), slice(None)), 4),
    # a step longer than a chunk reads one chunk per element, not `step` elements
    ((slice(None), slice(None, None, 16), slice(None), slice(None)), 8),
    ((slice(None), np.newaxis, slice(None, None, 2), slice(None), slice(None)), 2),
])
def test_slice_read_amplification(source, key, amp):
    assert source[key]._transform.read_amplification == amp


def test_strided_write_regions_are_budgeted_by_their_read(source, tmp_path, capsys):
    io.write(source[:, ::2, ::2, ::2], str(tmp_path / "s.zarr"), region_size_mb=0.05)
    io.write(source, str(tmp_path / "p.zarr"), region_size_mb=0.05)
    out = capsys.readouterr().out
    assert dio._chain_read_amplification(source[:, ::2, ::2, ::2]) == 8
    assert "Region" in out or "region" in out
