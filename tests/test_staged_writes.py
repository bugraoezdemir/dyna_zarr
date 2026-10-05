"""Staged writes (an outermost flatten / reshape / scan) honour every io.write setting.

They used to branch off at the top of io.write, before anything was resolved, receive 6
of its 20 arguments and create their output with ``zarr.open(mode="w")``. So they wrote
zarr v2 by default, dropped backend=/compressor/shards, silently REPLACED an existing
store, skipped all validation, staged their intermediates next to the output (so could
not write remotely) and fell back to the non-bounded region writer for a URL. Now they
write into the same resolved output the region pipeline opens, stage in the local temp
directory, and check free space first.
"""
import contextlib
import io as _stdio
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

from dyna_zarr.io import OutputExistsError
from dyna_zarr import Codecs, io, operations as ops
from dyna_zarr import rechunk as rechunk_mod
from dyna_zarr.rechunk import StagingSpaceError, staging_bytes

KINDS = ["flatten", "reshape", "cumsum"]
BACKENDS = ["tensorstore", pytest.param("zarrista", marks=pytest.mark.zarrista)]


def _write(*args, **kwargs):
    with contextlib.redirect_stdout(_stdio.StringIO()):
        return io.write(*args, **kwargs)


@pytest.fixture
def src(tmp_path):
    data = np.random.default_rng(0).random((8, 32, 32)).astype("f4")
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=data.shape, chunks=(4, 16, 16),
                          dtype="f4")
    z[...] = data
    return data, io.read(str(tmp_path / "src.zarr"))


def _lazy(kind, a):
    return {"flatten": lambda: ops.flatten(a),
            "reshape": lambda: ops.reshape(a, (16, 512)),
            "cumsum": lambda: ops.cumsum(a, axis=1)}[kind]()


def _ref(kind, data):
    return {"flatten": lambda: data.ravel(),
            "reshape": lambda: data.reshape(16, 512),
            "cumsum": lambda: np.cumsum(data, axis=1)}[kind]()


# --------------------------------------------------------------------------- #
# every output setting reaches the staged writers
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("fmt", [2, 3])
@pytest.mark.parametrize("kind", KINDS)
def test_staged_write_honours_format_codecs_dtype_backend(tmp_path, src, kind, fmt, backend):
    data, a = src
    out = tmp_path / "out.zarr"
    _write(_lazy(kind, a), str(out), zarr_format=fmt, backend=backend,
           compressor=Codecs("zstd", clevel=5), dtype="float64")
    z = zarr.open_array(str(out))
    assert z.metadata.zarr_format == fmt
    assert z.dtype == np.float64
    comp = z.compressors[0]
    level = comp.level if fmt == 2 else comp.to_dict()["configuration"]["level"]
    assert level == 5                                    # used to come out as level 0
    np.testing.assert_allclose(z[...], _ref(kind, data).astype("f8"), rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_staged_write_honours_sharding(tmp_path, src, kind):
    data, a = src
    lazy = _lazy(kind, a)
    chunks = tuple(max(1, s // 4) for s in lazy.shape)
    _write(lazy, str(tmp_path / "out.zarr"), zarr_format=3, chunks=chunks,
           shard_coefficients=(2,) * lazy.ndim)
    z = zarr.open_array(str(tmp_path / "out.zarr"))
    assert z.chunks == chunks and z.shards == tuple(2 * c for c in chunks)
    np.testing.assert_allclose(z[...], _ref(kind, data), rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_staged_write_respects_overwrite(tmp_path, src, kind):
    data, a = src
    out = tmp_path / "out.zarr"
    _write(a[:, :, :4] * 0, str(tmp_path / "decoy.zarr"))       # unrelated array
    shutil.copytree(tmp_path / "decoy.zarr", out)
    # zarr.open(mode="w") used to REPLACE whatever was there, with no error.
    with pytest.raises(OutputExistsError):
        _write(_lazy(kind, a), str(out))
    assert zarr.open_array(str(out)).shape == (8, 32, 4)         # untouched
    _write(_lazy(kind, a), str(out), overwrite=True)
    np.testing.assert_allclose(zarr.open_array(str(out))[...], _ref(kind, data), rtol=1e-5)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("knob", [dict(region_shape=(1,)), dict(memory_budget_mb=64),
                                  dict(num_readers=2)])
def test_region_pipeline_knobs_are_refused_not_ignored(tmp_path, src, kind, knob):
    _, a = src
    lazy = _lazy(kind, a)
    if "region_shape" in knob:
        knob = dict(region_shape=lazy.shape)
    with pytest.raises(ValueError, match="does not apply"):
        _write(lazy, str(tmp_path / "out.zarr"), **knob)
    assert not (tmp_path / "out.zarr").exists()


@pytest.mark.parametrize("kind", KINDS)
def test_staged_write_validates_before_writing(tmp_path, src, kind):
    _, a = src
    with pytest.raises(ValueError):
        _write(_lazy(kind, a), str(tmp_path / "out.zarr"), zarr_format=4)
    assert not (tmp_path / "out.zarr").exists()


# --------------------------------------------------------------------------- #
# local staging: location, cleanup, free space
# --------------------------------------------------------------------------- #

@pytest.fixture
def staging_dir(tmp_path, monkeypatch):
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(stage))
    return stage


@pytest.mark.parametrize("kind", ["flatten", "reshape"])
def test_staging_uses_the_temp_dir_and_is_removed(tmp_path, src, staging_dir, kind, monkeypatch):
    data, a = src
    made = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(*args, **kwargs):
        made.append(real_mkdtemp(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(rechunk_mod.tempfile, "mkdtemp", recording_mkdtemp)
    (tmp_path / "outdir").mkdir()
    _write(_lazy(kind, a), str(tmp_path / "outdir" / "out.zarr"))
    assert made and all(Path(d).is_relative_to(staging_dir) for d in made)
    assert list(staging_dir.iterdir()) == []                     # cleaned up
    assert os.listdir(tmp_path / "outdir") == ["out.zarr"]      # nothing next to the output
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "outdir" / "out.zarr"))[...],
                               _ref(kind, data), rtol=1e-5)


@pytest.mark.parametrize("kind", ["flatten", "reshape"])
def test_staging_is_removed_when_the_write_fails(tmp_path, src, staging_dir, kind, monkeypatch):
    _, a = src

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(rechunk_mod, "_copy", boom)
    with pytest.raises(RuntimeError, match="disk on fire"):
        _write(_lazy(kind, a), str(tmp_path / "out.zarr"))
    assert list(staging_dir.iterdir()) == []


@pytest.mark.parametrize("kind", ["flatten", "reshape"])
def test_insufficient_staging_space_is_refused_before_writing(tmp_path, src, kind, monkeypatch):
    _, a = src
    real = shutil.disk_usage
    monkeypatch.setattr(rechunk_mod.shutil, "disk_usage",
                        lambda p: real(p)._replace(free=1024))
    with pytest.raises(StagingSpaceError, match="TMPDIR") as exc:
        _write(_lazy(kind, a), str(tmp_path / "out.zarr"))
    assert "Nothing has been written" in str(exc.value)
    assert not (tmp_path / "out.zarr").exists()


def test_scan_needs_no_staging_space(tmp_path, src, monkeypatch):
    data, a = src
    real = shutil.disk_usage
    monkeypatch.setattr(rechunk_mod.shutil, "disk_usage",
                        lambda p: real(p)._replace(free=0))
    _write(ops.cumsum(a, axis=0), str(tmp_path / "out.zarr"))
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "out.zarr"))[...],
                               np.cumsum(data, axis=0), rtol=1e-5)


def test_staging_estimate(src):
    data, a = src
    nbytes = data.nbytes
    assert staging_bytes("scan", a) == 0
    assert staging_bytes("flatten", a) in (nbytes, 2 * nbytes)
    assert staging_bytes("reshape", a) == staging_bytes("flatten", a) + nbytes
    # 96 float32 per unit -> flat-contiguous chunks (1, 3, 32), which nest with the
    # source's (4, 16, 16) in neither direction: the rechunk intermediate coexists too.
    assert staging_bytes("flatten", a, max_mem=96 * 4) == 2 * nbytes
    # 1024 per unit -> (4, 32, 32), a multiple of (4, 16, 16): no intermediate.
    assert staging_bytes("flatten", a, max_mem=4096 * 4) == nbytes


# --------------------------------------------------------------------------- #
# remote output and file:// URLs
# --------------------------------------------------------------------------- #

@pytest.fixture
def fake_s3(tmp_path, monkeypatch):
    """s3://bucket/... served by an obstore LocalStore under tmp_path/bucket.

    Only the transport is local: the write goes through exactly the remote code path
    (zarrista's async API over an obstore store) that a real bucket uses."""
    pytest.importorskip("zarrista")
    from obstore.store import LocalStore
    from dyna_zarr.backends import zarrista_backend as zb

    root = tmp_path / "bucket"
    root.mkdir()

    def local_store(url, storage_options=None):
        key = str(url).split("://", 1)[1].split("/", 1)[1]      # drop scheme + bucket
        (root / key).mkdir(parents=True, exist_ok=True)
        return LocalStore(prefix=str(root / key))

    monkeypatch.setattr(zb, "_obstore", local_store)
    return root


@pytest.mark.zarrista
@pytest.mark.parametrize("kind", KINDS)
def test_staged_write_to_remote_store(tmp_path, src, fake_s3, kind):
    data, a = src
    _write(_lazy(kind, a), "s3://bucket/sub/out.zarr", backend="zarrista",
           compressor=Codecs("zstd", clevel=5))
    z = zarr.open_array(str(fake_s3 / "sub" / "out.zarr"))
    assert z.metadata.zarr_format == 3
    np.testing.assert_allclose(z[...], _ref(kind, data), rtol=1e-5)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("staged", [False, True])
def test_file_url_writes_to_the_named_path(tmp_path, src, backend, staged, monkeypatch):
    # A file:// URL used to reach zarrista as Path('file:///C:/x').resolve() -> the
    # drive-RELATIVE 'C:x' on Windows: a "successful" write under the working directory.
    data, a = src
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    target = tmp_path / "named" / "out.zarr"
    lazy = ops.flatten(a) if staged else a + 1
    _write(lazy, target.as_uri(), backend=backend)
    expected = data.ravel() if staged else data + 1
    np.testing.assert_allclose(zarr.open_array(str(target))[...], expected, rtol=1e-6)
    assert os.listdir(cwd) == []
