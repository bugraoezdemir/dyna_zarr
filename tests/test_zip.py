"""Zarr written straight into a zip (io.open_zip): every write mode, read-back while
writing, the layouts, and the refusals."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("zarr_zip")

from dyna_zarr import from_array, io                                   # noqa: E402
from dyna_zarr.codecs import Codecs                                     # noqa: E402

DATA = (np.random.default_rng(1).integers(0, 60, (24, 96, 128))
        + np.arange(128)).astype("uint16")
LEVELS = [dict(path="0", factors=(1, 1, 1)), dict(path="1", factors=(1, 2, 2)),
          dict(path="2", factors=(1, 4, 4))]


def _source():
    return from_array(DATA, chunks=(8, 32, 32))


def _folder(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in Path(root).rglob("*") if p.is_file()}


def _zipped(zip_path, prefix=""):
    with zipfile.ZipFile(zip_path) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        assert len(names) == len(set(names)), "an entry is listed twice"
        return {n[len(prefix):]: zf.read(n) for n in names if n.startswith(prefix)}


def _levels(tmp_path, name, outputs, **kw):
    """Write ``outputs`` into a folder and into a zip; return both, as {file: bytes}."""
    folder = tmp_path / f"{name}.zarr"
    io.write_levels(_source(), [dict(o, path=str(folder / o["path"])) for o in outputs], **kw)
    zip_path = tmp_path / f"{name}.ome.zarr.zip"
    with io.open_zip(zip_path) as z:
        with io.capture_messages() as messages:
            io.write_levels(_source(), [dict(o, path=z / o["path"]) for o in outputs], **kw)
        back = io.read(z / "0").compute()             # read back before the zip is closed
    np.testing.assert_array_equal(back, DATA)
    assert not Path(str(zip_path) + ".part").exists()
    assert not Path(str(zip_path) + ".part.shards").exists()
    return _folder(folder), _zipped(zip_path, f"{name}.ome.zarr/"), "\n".join(messages)


@pytest.mark.parametrize("fmt", [2, 3])
def test_whole_units_are_byte_identical_to_a_folder_write(tmp_path, fmt):
    outputs = [dict(o, chunks=(8, 32, 32),
                    shard_coefficients=(1, 2, 2) if fmt == 3 else None) for o in LEVELS]
    folder, zipped, _ = _levels(tmp_path, f"v{fmt}", outputs, zarr_format=fmt, max_threads=4)
    assert zipped == folder


@pytest.mark.parametrize("mode", ["appended to their shards", "transaction"])
def test_big_shards_decode_the_same(tmp_path, mode):
    """Write units larger than a region: shards assembled chunk by chunk in temporary
    files (zarr v3, with a memory budget), or units built in TensorStore transactions
    (zarr v2 chunks, which the shard writer does not write), committed several times
    each. Chunks land in a shard in the order threads finish them, so the bytes differ
    from run to run even in a folder: the data is compared."""
    kw = dict(max_threads=3, region_size_mb=0.05)
    if mode == "transaction":
        kw["zarr_format"] = 2
        outputs = [dict(o, chunks=(24, 48, 64)) for o in LEVELS]
    else:
        kw.update(zarr_format=3, memory_budget_mb=50,
                  compressor=Codecs("blosc", cname="zstd", clevel=3))
        outputs = [dict(o, chunks=(4, 16, 16), shard_coefficients=(6, 6, 8)) for o in LEVELS]
    folder, zipped, messages = _levels(tmp_path, "big", outputs, **kw)
    assert mode in messages
    assert sorted(zipped) == sorted(folder)
    zip_path = tmp_path / "big.ome.zarr.zip"
    for level in ("0", "1", "2"):
        np.testing.assert_array_equal(io.read(str(zip_path / level)).compute(),
                                      io.read(str(tmp_path / "big.zarr" / level)).compute())


def test_units_committed_several_times_go_in_once_whole(tmp_path):
    """A source read in whole planes fills each unit over many blocks, committed every
    region_size_mb: the unit goes into the zip after its last commit, never before."""
    outputs = [dict(o, chunks=(24, 48, 64)) for o in LEVELS[:2]]
    zip_path = tmp_path / "planar.ome.zarr.zip"
    with io.open_zip(zip_path) as z:
        with io.capture_messages() as messages:
            io.write_levels(from_array(DATA, chunks=(1, 96, 128)),
                            [dict(o, path=z / o["path"]) for o in outputs],
                            zarr_format=2, max_threads=4, region_size_mb=0.05)
    assert "written every" in "\n".join(messages)
    _zipped(zip_path)                                           # each entry listed once
    np.testing.assert_array_equal(io.read(str(zip_path / "0")).compute(), DATA)
    np.testing.assert_array_equal(io.read(str(zip_path / "1")).compute(), DATA[:, ::2, ::2])


def test_io_write_is_byte_identical(tmp_path):
    kw = dict(chunks=(8, 32, 32), shard_coefficients=(1, 2, 2))
    io.write(_source(), str(tmp_path / "w.zarr"), **kw)
    with io.open_zip(tmp_path / "w.ome.zarr.zip") as z:
        io.write(_source(), z / "", **kw)
    assert _zipped(tmp_path / "w.ome.zarr.zip", "w.ome.zarr/") == _folder(tmp_path / "w.zarr")


def test_sink_parts_of_units_and_rfc9(tmp_path):
    """Blocks covering parts of chunks: the chunks go in when covered, the rest at close.
    An .ozx has the store at the zip's root and RFC-9's comment."""
    with io.open_zip(tmp_path / "s.ozx") as z:
        sink = io.create_sink(z / "0", DATA.shape, DATA.dtype, chunks=(8, 32, 32),
                              zarr_format=3)
        sink[0:5, 0:40, :] = DATA[0:5, 0:40, :]
        sink[5:, :, :] = DATA[5:, :, :]
        sink[0:5, 40:, :] = DATA[0:5, 40:, :]
        z.write("zarr.json", json.dumps({"zarr_format": 3, "node_type": "group",
                                         "attributes": {"ome": {"version": "0.5"}}}).encode())
    np.testing.assert_array_equal(io.read(str(tmp_path / "s.ozx" / "0")).compute(), DATA)
    with zipfile.ZipFile(tmp_path / "s.ozx") as zf:
        assert json.loads(zf.comment)["ome"]["version"] == "0.5"
        assert zf.namelist()[:2] == ["zarr.json", "0/zarr.json"]


def test_zarr_python_reads_the_closed_zip(tmp_path):
    zarr = pytest.importorskip("zarr")
    with io.open_zip(tmp_path / "img.ome.zarr.zip") as z:
        io.write_levels(_source(), [dict(o, chunks=(8, 32, 32), shard_coefficients=(1, 2, 2))
                                    for o in [dict(LEVELS[0], path=z / "0"),
                                              dict(LEVELS[1], path=z / "1")]], zarr_format=3)
        group = zarr.open_group(z.zarr_store(), mode="a", zarr_format=3)   # beside the arrays
        group.attrs["ome"] = {"version": "0.5"}
    store = zarr.storage.ZipStore(str(tmp_path / "img.ome.zarr.zip"), mode="r")
    group = zarr.open_group(store, mode="r", path="img.ome.zarr")
    np.testing.assert_array_equal(group["0"][:], DATA)
    np.testing.assert_array_equal(group["1"][:], DATA[:, ::2, ::2])


def test_an_error_leaves_nothing(tmp_path):
    with pytest.raises(RuntimeError):
        with io.open_zip(tmp_path / "x.ome.zarr.zip") as z:
            io.write(_source(), z / "0", chunks=(8, 32, 32))
            raise RuntimeError("stop")
    assert list(tmp_path.iterdir()) == []


def test_a_zip_not_opened_is_refused(tmp_path):
    with pytest.raises(ValueError, match="open_zip"):
        io.write(_source(), str(tmp_path / "x.ome.zarr.zip" / "0"))
    assert list(tmp_path.iterdir()) == []


def test_refusals_inside_an_open_zip(tmp_path):
    with io.open_zip(tmp_path / "x.ozx") as z:
        io.write(_source(), z / "0", chunks=(8, 32, 32), zarr_format=3)
        with pytest.raises(io.OutputExistsError if hasattr(io, "OutputExistsError")
                           else FileExistsError, match="cannot replace"):
            io.write(_source(), z / "0", overwrite=True)
        with pytest.raises(ValueError, match="Zarr v3"):
            io.write(_source(), z / "1", zarr_format=2)
        with pytest.raises(ValueError, match="tensorstore"):
            io.write(_source(), z / "2", backend="zarrista")
        z.write("zarr.json", b'{"zarr_format":3,"node_type":"group","attributes":{}}')


def test_a_unit_is_written_once(tmp_path):
    with io.open_zip(tmp_path / "x.ome.zarr.zip") as z:
        sink = io.create_sink(z / "0", DATA.shape, DATA.dtype, chunks=(8, 32, 32))
        sink[0:8, 0:32, 0:32] = DATA[0:8, 0:32, 0:32]
        with pytest.raises(ValueError, match="already in the zip"):
            sink[0:8, 0:32, 0:32] = DATA[0:8, 0:32, 0:32]
