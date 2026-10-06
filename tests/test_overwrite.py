"""io.write's overwrite rule is the same for every backend, write path and location.

Before: TensorStore raised ALREADY_EXISTS on an existing array, but zarrista silently
wrote into it (and over a group's metadata); a remote overwrite=True was not applied at
all; and files that were not a zarr store were written into. Now, decided once in
_resolve_output before anything is written:

- nothing there (or an empty directory): write;
- a zarr array or group there: OutputExistsError, or with overwrite=True delete it
  (everything under it) and write;
- files that are not a zarr store: OutputExistsError, even with overwrite=True.
"""
import contextlib
import io as _stdio

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io, operations as ops
from dyna_zarr.io import OutputExistsError

NEW = np.arange(4 * 16 * 16, dtype="f4").reshape(4, 16, 16)
BACKENDS = ["tensorstore", pytest.param("zarrista", marks=pytest.mark.zarrista)]
LOCATIONS = ["local", "s3"]
EXISTING = ["none", "array", "larger_array", "group", "other_files"]


def _write(*args, **kwargs):
    with contextlib.redirect_stdout(_stdio.StringIO()):
        return io.write(*args, **kwargs)


class _Loc:
    """An output location, local or in an S3 bucket, with raw key access to plant
    things there and list what is left."""

    def __init__(self, kind, tmp_path, request):
        self.kind = kind
        if kind == "local":
            self.root, self.opts = tmp_path, None
        else:
            name, self.opts = request.getfixturevalue("bucket")
            self.root = f"s3://{name}"

    def path(self, name):
        return str(self.root / name) if self.kind == "local" else f"{self.root}/{name}"

    def put(self, name, key, data):
        if self.kind == "local":
            p = self.root / name / key
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        else:
            import obstore
            from obstore.store import from_url
            obstore.put(from_url(self.root, **self.opts), f"{name}/{key}", data)

    def keys(self, name):
        if self.kind == "local":
            base = self.root / name
            return sorted(str(p.relative_to(base)).replace("\\", "/")
                          for p in base.rglob("*") if p.is_file()) if base.exists() else []
        import obstore
        from obstore.store import from_url
        store = from_url(f"{self.root}/{name}", **self.opts)
        return sorted(o["path"] for batch in obstore.list(store) for o in batch)


@pytest.fixture(params=LOCATIONS)
def loc(request, tmp_path):
    if request.param == "s3":
        pytest.importorskip("obstore", reason="the test plants S3 keys with obstore")
    return _Loc(request.param, tmp_path, request)


def _plant(loc, name, kind):
    if kind == "array":
        _write(DynamicArray(np.full((4, 16, 16), -1, "f4")), loc.path(name),
               storage_options=loc.opts)
    elif kind == "larger_array":
        _write(DynamicArray(np.full((8, 32, 32), -1, "f4")), loc.path(name),
               chunks=(2, 8, 8), storage_options=loc.opts)
    elif kind == "group":
        loc.put(name, "zarr.json", b'{"zarr_format": 3, "node_type": "group", "attributes": {}}')
        loc.put(name, "child/zarr.json", b'{"zarr_format": 3, "node_type": "group", "attributes": {}}')
    elif kind == "other_files":
        loc.put(name, "notes.txt", b"keep me")


def _lazy(staged):
    a = DynamicArray(NEW)
    return ops.reshape(ops.reshape(a, (4, 256)), (4, 16, 16)) if staged else a


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("staged", [False, True], ids=["region", "staged"])
@pytest.mark.parametrize("existing", EXISTING)
@pytest.mark.parametrize("overwrite", [False, True], ids=["keep", "overwrite"])
def test_one_rule_everywhere(loc, backend, staged, existing, overwrite):
    _plant(loc, "out.zarr", existing)
    before = loc.keys("out.zarr")
    refused = existing == "other_files" or (existing != "none" and not overwrite)

    if refused:
        with pytest.raises(OutputExistsError):
            _write(_lazy(staged), loc.path("out.zarr"), backend=backend,
                   overwrite=overwrite, storage_options=loc.opts)
        assert loc.keys("out.zarr") == before                  # untouched
        return

    _write(_lazy(staged), loc.path("out.zarr"), backend=backend, overwrite=overwrite,
           storage_options=loc.opts)
    back = io.read(loc.path("out.zarr"), storage_options=loc.opts).compute()
    np.testing.assert_array_equal(back, NEW)
    if existing == "group":                                     # replaced, not written into
        assert not any(k.startswith("child/") for k in loc.keys("out.zarr"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_replacing_a_larger_array_leaves_no_stale_chunks(loc, backend):
    _plant(loc, "out.zarr", "larger_array")                      # 4x4x4 = 64 chunks
    _write(DynamicArray(NEW), loc.path("out.zarr"), backend=backend, overwrite=True,
           chunks=(4, 16, 16), storage_options=loc.opts)         # 1 chunk
    keys = loc.keys("out.zarr")
    assert len(keys) == 2, keys                                  # zarr.json + one chunk


@pytest.mark.parametrize("backend", BACKENDS)
def test_overwrite_deletes_only_its_own_prefix(loc, backend):
    for sibling in ("out.zarr2", "out.zarr_backup"):
        loc.put(sibling, "notes.txt", b"keep me")
    _plant(loc, "out.zarr", "array")
    _write(DynamicArray(NEW), loc.path("out.zarr"), backend=backend, overwrite=True,
           storage_options=loc.opts)
    assert loc.keys("out.zarr2") == ["notes.txt"]
    assert loc.keys("out.zarr_backup") == ["notes.txt"]


def test_an_empty_directory_counts_as_nothing(tmp_path):
    (tmp_path / "out.zarr").mkdir()
    _write(DynamicArray(NEW), str(tmp_path / "out.zarr"))
    np.testing.assert_array_equal(io.read(str(tmp_path / "out.zarr")).compute(), NEW)


def test_writing_a_level_into_an_existing_group_is_fine(tmp_path):
    # the OME-Zarr case: image.zarr is a group, image.zarr/0 does not exist yet
    zarr.open_group(str(tmp_path / "image.zarr"), mode="w")
    _write(DynamicArray(NEW), str(tmp_path / "image.zarr" / "0"))
    np.testing.assert_array_equal(io.read(str(tmp_path / "image.zarr" / "0")).compute(), NEW)
    assert zarr.open_group(str(tmp_path / "image.zarr"), mode="r") is not None


def test_error_is_also_a_value_error_and_says_what_to_do(tmp_path):
    _write(DynamicArray(NEW), str(tmp_path / "out.zarr"))
    with pytest.raises(ValueError, match="overwrite=True"):     # what ALREADY_EXISTS was
        _write(DynamicArray(NEW), str(tmp_path / "out.zarr"))
    zarr.open_group(str(tmp_path / "g.zarr"), mode="w")
    with pytest.raises(FileExistsError, match="whole group"):
        _write(DynamicArray(NEW), str(tmp_path / "g.zarr"))


def test_nothing_is_created_when_refused(tmp_path):
    (tmp_path / "out.zarr").mkdir()
    (tmp_path / "out.zarr" / "notes.txt").write_text("keep me")
    with pytest.raises(OutputExistsError, match="not a zarr store"):
        _write(DynamicArray(NEW), str(tmp_path / "out.zarr"), overwrite=True)
    assert [p.name for p in (tmp_path / "out.zarr").iterdir()] == ["notes.txt"]
