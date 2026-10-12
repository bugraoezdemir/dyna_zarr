"""What a failure leaves behind.

- write_array pauses Python's cyclic garbage collector while regions are in flight. A
  region that raised on engine="pipeline" left it off for the rest of the process (a long
  eubi-bridge run or a GUI then never collected cycles) - and, before that, hung: a failed
  reader never sent the end-of-input sentinels, so the writers waited forever.
- io.read fell back from TensorStore to zarr-python with `except Exception: pass`. An
  explicit backend='tensorstore' now fails loudly; the default still falls back (zarr-python
  reads a few stores TensorStore cannot) but warns with the reason.
"""
import gc
import sys
import threading

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, io

dio = sys.modules["dyna_zarr.io"]          # the module (``io`` is the namespace class)


def _src(tmp_path):
    data = np.arange(4 * 32 * 32, dtype="f4").reshape(4, 32, 32)
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=data.shape, chunks=(2, 16, 16),
                          dtype="f4")
    z[...] = data
    return data


@pytest.fixture
def gc_on():
    was = gc.isenabled()
    gc.enable()
    yield
    if not was:
        gc.disable()


def _boom(block):
    if block.size > 64:              # not on the tiny sample dyna infers the dtype from
        raise RuntimeError("region failed")
    return block


@pytest.mark.parametrize("engine", ["fused", "pipeline"])
def test_a_failed_region_raises_and_restores_gc(tmp_path, gc_on, engine):
    _src(tmp_path)
    arr = io.read(str(tmp_path / "src.zarr")).map_blocks(_boom)
    outcome = {}

    def run():                       # in a thread, so a hang fails the test, not CI
        try:
            io.write(arr, str(tmp_path / "out.zarr"), engine=engine)
        except Exception as exc:
            outcome["error"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive(), f"engine={engine!r} hung on a failed region"
    assert isinstance(outcome.get("error"), RuntimeError)
    assert "region failed" in str(outcome["error"])
    assert gc.isenabled()


def test_gc_stays_off_if_it_was_off(tmp_path):
    _src(tmp_path)
    was = gc.isenabled()
    gc.disable()
    try:
        io.write(io.read(str(tmp_path / "src.zarr")), str(tmp_path / "out.zarr"))
        assert not gc.isenabled()          # the caller's choice is kept
    finally:
        if was:
            gc.enable()


def _tensorstore_fails(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("tensorstore cannot open this")
    monkeypatch.setattr(dio, "read_file", fail)


def test_default_backend_falls_back_with_a_warning(tmp_path, monkeypatch):
    data = _src(tmp_path)
    _tensorstore_fails(monkeypatch)
    with pytest.warns(RuntimeWarning, match="reading it with zarr-python instead"):
        arr = io.read(str(tmp_path / "src.zarr"))
    np.testing.assert_array_equal(arr.compute(), data)


def test_explicit_tensorstore_fails_loudly(tmp_path, monkeypatch):
    _src(tmp_path)
    _tensorstore_fails(monkeypatch)
    with pytest.raises(OSError, match="tensorstore cannot open this"):
        io.read(str(tmp_path / "src.zarr"), backend="tensorstore")


def test_the_documented_exceptions_import_from_the_package():
    import dyna_zarr
    from dyna_zarr.backends.zarrista_backend import UnsupportedByBackend
    assert dyna_zarr.OutputExistsError is dio.OutputExistsError
    assert dyna_zarr.RepeatedReductionWarning is dio.RepeatedReductionWarning
    assert dyna_zarr.UnsupportedByBackend is UnsupportedByBackend
