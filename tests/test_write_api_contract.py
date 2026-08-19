"""Contract tests for the write API's signature and its concurrency/overwrite knobs.

`io.write` is a hand-maintained wrapper around `write_array`. It once forwarded its
arguments POSITIONALLY with a trailing `**kwargs`, so a parameter it did not name
(`region_shape`) still reached `write_array` by accident of argument order while
`inspect.signature` reported it unsupported - and a genuine typo was swallowed silently.
These tests pin the wrapper to its target and cover the knobs that replaced the old
independent queue/in-flight settings.
"""

import inspect
import warnings

import numpy as np
import pytest
import zarr

import importlib

from dyna_zarr.io import io

# NB: `from dyna_zarr import io` gives the io CLASS, which shadows the module of the same
# name inside the package. Reach the module explicitly to get at write_array.
dyna_io_module = importlib.import_module("dyna_zarr.io")


def _src(tmp_path, shape=(32, 32, 32), chunks=(8, 8, 8), dtype="int32"):
    p = tmp_path / "src.zarr"
    z = zarr.open(str(p), mode="w", shape=shape, chunks=chunks, dtype=dtype)
    z[:] = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    return io.read(str(p))


def test_write_signature_parity():
    """io.write must expose exactly write_array's parameters, names and defaults alike.

    This is the guard for the root cause: two hand-duplicated signatures with nothing
    checking they stay in sync.
    """
    wrapper = inspect.signature(io.write).parameters
    target = inspect.signature(dyna_io_module.write_array).parameters

    assert list(wrapper) == list(target)
    for name, tp in target.items():
        assert wrapper[name].default == tp.default, f"default drift on {name!r}"


def test_no_var_kwargs_on_write_api():
    """Neither function may absorb unknown keywords - that is what hid the bug."""
    for fn in (io.write, dyna_io_module.write_array):
        kinds = [p.kind for p in inspect.signature(fn).parameters.values()]
        assert inspect.Parameter.VAR_KEYWORD not in kinds


def test_unknown_kwarg_raises(tmp_path):
    """A typo must fail loudly rather than silently falling back to region_size_mb."""
    arr = _src(tmp_path)
    with pytest.raises(TypeError):
        io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
                 regionshape=(16, 16, 16), zarr_format=2)


def test_region_shape_is_honored(tmp_path, capsys):
    """region_shape must reach the writer and drive the region grid."""
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2)
    out = capsys.readouterr().out
    assert "(explicit region_shape)" in out
    assert "Region: (16, 16, 16)" in out


def test_region_shape_must_divide_chunks(tmp_path):
    arr = _src(tmp_path)
    with pytest.raises(ValueError, match="multiple"):
        io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
                 region_shape=(12, 12, 12), zarr_format=2)


# --- concurrency budget ---------------------------------------------------------------

def test_max_workers_is_total_live_regions(tmp_path, capsys):
    """max_workers is the TOTAL live-region budget: the derived stages must sum to it."""
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2, max_workers=8, num_readers=4)
    line = [l for l in capsys.readouterr().out.splitlines() if "Live-region budget" in l]
    assert line, "budget line not printed"
    text = line[0]
    assert "budget: 8" in text
    readers = int(text.split("readers=")[1].split(",")[0])
    queue = int(text.split("queue=")[1].split(",")[0])
    inflight = int(text.split("inflight=")[1].split(",")[0])
    assert readers == 4
    assert readers + queue + inflight == 8


def test_memory_budget_clamps_workers(tmp_path, capsys):
    """A tight budget lowers the live-region count below what was requested."""
    arr = _src(tmp_path)
    # region 16^3 int32 = 16 KiB; budget 0.0625 MiB = 64 KiB -> 4 live regions
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2,
             max_workers=32, memory_budget_mb=0.0625)
    line = [l for l in capsys.readouterr().out.splitlines() if "Live-region budget" in l][0]
    assert "budget: 4" in line


def test_memory_budget_is_a_ceiling_not_a_target(tmp_path, capsys):
    """A generous budget must never RAISE concurrency above the requested max_workers."""
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2,
             max_workers=4, memory_budget_mb=4096)
    line = [l for l in capsys.readouterr().out.splitlines() if "Live-region budget" in l][0]
    assert "budget: 4" in line


def test_impossible_budget_warns_and_floors(tmp_path):
    """A budget too small for a minimal pipeline warns loudly instead of deadlocking."""
    arr = _src(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
                 region_shape=(16, 16, 16), zarr_format=2,
                 max_workers=8, memory_budget_mb=0.001)
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("cannot fit" in m for m in msgs), msgs
    # and it still produced correct output
    assert zarr.open(str(tmp_path / "o.zarr"), mode="r").shape == (32, 32, 32)


def test_write_correct_under_tight_budget(tmp_path):
    """Data must round-trip exactly at the pipeline floor (1 reader/1 queue/1 write)."""
    src = _src(tmp_path)
    expected = np.asarray(src[:, :, :])
    out = tmp_path / "o.zarr"
    io.write(src, str(out), chunks=(8, 8, 8), region_shape=(16, 16, 16),
             zarr_format=2, max_workers=3)
    np.testing.assert_array_equal(zarr.open(str(out), mode="r")[...], expected)


# --- overwrite ------------------------------------------------------------------------

def test_overwrite_replaces_existing(tmp_path):
    out = tmp_path / "o.zarr"
    a = _src(tmp_path, shape=(32, 32, 32))
    io.write(a, str(out), chunks=(8, 8, 8), zarr_format=2)
    assert zarr.open(str(out), mode="r").shape == (32, 32, 32)

    smaller = _src(tmp_path / "b", shape=(16, 16, 16))
    io.write(smaller, str(out), chunks=(8, 8, 8), zarr_format=2, overwrite=True)
    assert zarr.open(str(out), mode="r").shape == (16, 16, 16)


def test_overwrite_refuses_non_zarr_directory(tmp_path):
    """A mistyped output_path must not delete an unrelated tree."""
    victim = tmp_path / "not_a_store"
    victim.mkdir()
    (victim / "precious.txt").write_text("keep me")

    arr = _src(tmp_path)
    with pytest.raises(ValueError, match="does not look like a zarr store"):
        io.write(arr, str(victim), chunks=(8, 8, 8), zarr_format=2, overwrite=True)
    assert (victim / "precious.txt").read_text() == "keep me"
