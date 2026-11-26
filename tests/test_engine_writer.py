"""The engine's writer (Workplan DR8, scope decided 2026-11-25): with the engine on,
local Zarr v2 / v3 / sharded v3 outputs are encoded and stored by zarrs, TensorStore
still creating the array. The outputs are the ones TensorStore writes: the same
metadata and the same values."""
import json
import os
import sys

import numpy as np
import pytest
import zarr

from dyna_zarr import _engine, io
from dyna_zarr.operations.downsampling import coarsen

pytestmark = pytest.mark.engine


@pytest.fixture(autouse=True)
def engine_on(monkeypatch):
    if not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", True)
    monkeypatch.delenv("DYNA_ZARR_ENGINE_WRITER", raising=False)


DTYPES = ["uint8", "int16", "uint16", "int32", "float32", "float64", "bool", "complex64"]


def _data(dtype, shape=(6, 70, 90)):
    rng = np.random.default_rng(2)
    if dtype == "bool":
        return rng.integers(0, 2, shape).astype(bool)
    if dtype.startswith("complex"):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    if dtype.startswith("float"):
        return rng.normal(size=shape).astype(dtype)
    return rng.integers(0, 100, shape).astype(dtype)


def _meta(path):
    for name in ("zarr.json", ".zarray"):
        p = os.path.join(path, name)
        if os.path.exists(p):
            return json.load(open(p))
    raise AssertionError(f"no metadata in {path}")


def _both(tmp_path, monkeypatch, write):
    """``write(path)`` with the engine writer and with TensorStore's."""
    write(str(tmp_path / "engine"))
    monkeypatch.setenv("DYNA_ZARR_ENGINE_WRITER", "0")
    write(str(tmp_path / "ts"))
    monkeypatch.delenv("DYNA_ZARR_ENGINE_WRITER")
    return str(tmp_path / "engine"), str(tmp_path / "ts")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", ["v2", "v3", "sharded"])
def test_the_engine_writes_what_tensorstore_writes(tmp_path, monkeypatch, dtype, fmt):
    if fmt == "v2" and dtype.startswith("complex"):
        pytest.skip("TensorStore cannot read zarr-python's v2 complex fill value (the source)")
    src = tmp_path / "src.zarr"
    kw = {"zarr_format": 2} if fmt == "v2" else {}
    if fmt == "sharded":
        kw["shards"] = (6, 32, 64)
    zarr.create_array(str(src), shape=(6, 70, 90), chunks=(2, 16, 32), dtype=dtype,
                      **kw)[:] = _data(dtype)
    a = io.read(str(src))
    e, t = _both(tmp_path, monkeypatch, lambda p: io.write(a, p, max_threads=3))
    assert _meta(e) == _meta(t)
    np.testing.assert_array_equal(np.asarray(zarr.open_array(e)[:]),
                                  np.asarray(zarr.open_array(t)[:]))
    np.testing.assert_array_equal(np.asarray(zarr.open_array(e)[:]), _data(dtype))


def test_write_levels_and_the_pipeline_use_it(tmp_path, monkeypatch, capsys):
    src = tmp_path / "src.zarr"
    zarr.create_array(str(src), shape=(8, 256, 300), chunks=(4, 64, 64),
                      dtype="uint16")[:] = _data("uint16", (8, 256, 300))
    a = io.read(str(src))

    def levels(p):
        io.write_levels(a, [{"path": p + "/0", "factors": (1, 1, 1)},
                            {"path": p + "/1", "factors": (1, 2, 2)}],
                        method="mean", max_threads=4)
    e, t = _both(tmp_path, monkeypatch, levels)
    for lvl in ("0", "1"):
        np.testing.assert_array_equal(np.asarray(zarr.open_array(f"{e}/{lvl}")[:]),
                                      np.asarray(zarr.open_array(f"{t}/{lvl}")[:]))
    e2, t2 = _both(tmp_path / "p", monkeypatch,
                   lambda p: io.write(coarsen(a, (1, 2, 2)), p, max_threads=6,
                                      engine="pipeline"))
    np.testing.assert_array_equal(np.asarray(zarr.open_array(e2)[:]),
                                  np.asarray(zarr.open_array(t2)[:]))
    assert "the engine cannot write" not in capsys.readouterr().out


def test_it_is_used_only_with_the_engine_on(tmp_path, monkeypatch):
    seen = []
    mod = sys.modules["dyna_zarr.io"]          # dyna_zarr.io is a namespace; the module
    orig = mod._engine_output

    def spy(spec, handle):
        out = orig(spec, handle)
        seen.append(type(out).__name__)
        return out
    monkeypatch.setattr(mod, "_engine_output", spy)
    a = io.read(_small(tmp_path))
    io.write(a, str(tmp_path / "on"), max_threads=2)
    monkeypatch.setattr(_engine, "_choice", False)
    io.write(a, str(tmp_path / "off"), max_threads=2)
    assert seen == ["_EngineOutput", "TensorStore"]


def _small(tmp_path):
    src = tmp_path / "small.zarr"
    zarr.create_array(str(src), shape=(4, 40, 50), chunks=(2, 16, 16),
                      dtype="uint16")[:] = _data("uint16", (4, 40, 50))
    return str(src)


def test_the_writer_refuses_what_it_cannot_store(tmp_path):
    core = pytest.importorskip("dyna_zarr_core")
    src = _small(tmp_path)
    w = core.Writer(src)
    assert w.shape == (4, 40, 50) and w.dtype == "uint16" and w.unit == (2, 16, 16)
    with pytest.raises(TypeError):
        w.store([0, 0, 0], np.zeros((2, 16, 16), "int16"))
    with pytest.raises(ValueError):
        w.store([0, 0, 0], np.zeros((2, 16, 32), "uint16")[:, :, ::2])     # not contiguous
    with pytest.raises(ValueError):
        w.store([0, 0, 0], np.zeros((2, 16, 16), ">u2"))
    with pytest.raises(Exception):
        w.store([3, 0, 0], np.zeros((2, 16, 16), "uint16"))                # past the end
    w.store([0, 0, 0], np.full((2, 16, 16), 7, "uint16"), threads=2)
    assert (np.asarray(zarr.open_array(src)[:2, :16, :16]) == 7).all()


def test_big_shards_chunk_by_chunk_go_through_the_engine(tmp_path, monkeypatch):
    """write_levels' bounded mode (shards larger than a region) appends inner chunks
    to their shards with the engine writer (zarrs' partial encoding), not dyna's Python
    shard assembler; the shards hold no dead space."""
    import dyna_zarr._shard_writer as sw

    def refused(*a, **k):
        raise AssertionError("the Python shard assembler was used")
    src = tmp_path / "src.zarr"
    data = _data("uint16", (16, 128, 128))
    zarr.create_array(str(src), shape=data.shape, chunks=(16, 64, 64),
                      dtype="uint16")[:] = data
    outs = lambda root: [{"path": f"{root}/0", "factors": (1, 1, 1),
                          "shard_coefficients": (1, 2, 2), "chunks": (4, 32, 32)},
                         {"path": f"{root}/1", "factors": (1, 2, 2),
                          "shard_coefficients": (1, 2, 2), "chunks": (4, 32, 32)}]
    a = io.read(str(src))
    monkeypatch.setattr(sw, "ShardAssembler", refused)
    with io.capture_messages() as messages:
        io.write_levels(a, outs(tmp_path / "e"), method="mean", region_size_mb=0.02,
                        zarr_format=3, max_threads=4)
    assert any("appended to their shards" in m for m in messages), messages
    monkeypatch.undo()
    monkeypatch.setattr(_engine, "_choice", True)
    monkeypatch.setenv("DYNA_ZARR_ENGINE_WRITER", "0")
    io.write_levels(a, outs(tmp_path / "t"), method="mean", region_size_mb=0.02,
                    zarr_format=3, max_threads=4)
    for lvl in ("0", "1"):
        e, t = str(tmp_path / "e" / lvl), str(tmp_path / "t" / lvl)
        assert _meta(e) == _meta(t)
        np.testing.assert_array_equal(np.asarray(zarr.open_array(e)[:]),
                                      np.asarray(zarr.open_array(t)[:]))
        size = lambda d: sum(os.path.getsize(os.path.join(r, f))
                             for r, _, fs in os.walk(d) for f in fs)
        assert size(e) <= size(t) * 1.02, (lvl, size(e), size(t))


@pytest.mark.parametrize("fmt", ["v2", "v3", "sharded"])
def test_zip_outputs_are_written_by_the_engine(tmp_path, monkeypatch, fmt):
    """Inside a zip the engine writer stores the units as files next to it, each moved
    into the zip when complete: the same entries, metadata and values as TensorStore's."""
    import zipfile
    from dyna_zarr import _zip
    seen = []
    orig = _zip.ZipOutput._engine_output

    def spy(self, metadata):
        out = orig(self, metadata)
        seen.append(out is not None)
        return out
    monkeypatch.setattr(_zip.ZipOutput, "_engine_output", spy)
    src = tmp_path / "src.zarr"
    kw = {"zarr_format": 2} if fmt == "v2" else {}
    if fmt == "sharded":
        kw["shards"] = (6, 32, 64)
    zarr.create_array(str(src), shape=(6, 70, 90), chunks=(2, 16, 32), dtype="uint16",
                      **kw)[:] = _data("uint16")
    a = io.read(str(src))

    def write(name):
        with io.open_zip(tmp_path / name) as z:
            io.write(a, z / "0", max_threads=3)
            io.write_levels(a, [{"path": z / "L1", "factors": (1, 2, 2)}], method="mean",
                            max_threads=3)
    write("e.zip")
    assert seen and all(seen)
    monkeypatch.setenv("DYNA_ZARR_ENGINE_WRITER", "0")
    write("t.zip")
    with zipfile.ZipFile(tmp_path / "e.zip") as ze, zipfile.ZipFile(tmp_path / "t.zip") as zt:
        # entries sit under the zip's own name ("e/...", "t/...")
        inner = lambda z: {n.split("/", 1)[1]: n for n in z.namelist() if "/" in n}
        ne, nt = inner(ze), inner(zt)
        assert sorted(ne) == sorted(nt)
        for name in ne:
            if name.endswith((".zarray", "zarr.json")):
                assert json.loads(ze.read(ne[name])) == json.loads(zt.read(nt[name])), name
    for key in ("0", "L1"):
        e = io.read(str(tmp_path / "e.zip" / key)).compute()
        t = io.read(str(tmp_path / "t.zip" / key)).compute()
        np.testing.assert_array_equal(np.asarray(e), np.asarray(t))
    assert not any(p.name.endswith("-units") for p in tmp_path.rglob("*"))
