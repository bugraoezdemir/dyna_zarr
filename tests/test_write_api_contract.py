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


# --- concurrency budget: pipeline engine ----------------------------------------------

def _budget_line(out):
    line = [l for l in out.splitlines() if "Live-region budget" in l]
    assert line, "budget line not printed"
    return line[0]


def test_max_threads_is_total_live_regions(tmp_path, capsys):
    """pipeline: max_threads is the TOTAL live-region budget; the stages sum to it."""
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8), engine="pipeline",
             region_shape=(16, 16, 16), zarr_format=2, max_threads=8, num_readers=4)
    text = _budget_line(capsys.readouterr().out)
    assert "budget: 8" in text
    readers = int(text.split("readers=")[1].split(",")[0])
    queue = int(text.split("queue=")[1].split(",")[0])
    inflight = int(text.split("inflight=")[1].split(",")[0])
    assert readers == 4
    assert readers + queue + inflight == 8


def test_memory_budget_clamps_workers(tmp_path, capsys):
    """pipeline: a tight budget lowers the live-region count below what was requested.

    Region 16^3 int32: read 16 KiB, out 16 KiB. A reader holds 2*16 + 16 = 48 KiB, a
    queued region 16 KiB, an in-flight write 32 KiB. 5 live regions = 2 readers, 1
    queued, 2 in flight = 176 KiB; 6 = 3/1/2 = 224 KiB. A 192 KiB budget -> 5."""
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8), engine="pipeline",
             region_shape=(16, 16, 16), zarr_format=2,
             max_threads=32, memory_budget_mb=192 / 1024)
    line = _budget_line(capsys.readouterr().out)
    assert "budget: 5" in line and "readers=2, queue=1, inflight=2" in line


def test_memory_budget_is_a_ceiling_not_a_target(tmp_path, capsys):
    """pipeline: a generous budget must never RAISE concurrency above max_threads."""
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8), engine="pipeline",
             region_shape=(16, 16, 16), zarr_format=2,
             max_threads=4, memory_budget_mb=4096)
    assert "budget: 4" in _budget_line(capsys.readouterr().out)


# --- concurrency budget: fused engine (the default) ------------------------------------

def _fused_workers(out):
    line = [l for l in out.splitlines() if "Engine: fused" in l]
    assert line, "fused engine line not printed"
    return int(line[0].split("Engine: fused, ")[1].split(" threads")[0])


def test_fused_is_the_default_and_max_threads_is_threads(tmp_path, capsys):
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2, max_threads=5)
    out = capsys.readouterr().out
    assert _fused_workers(out) == 5
    assert "Live-region budget" not in out


def test_fused_memory_budget_clamps_workers(tmp_path, capsys):
    """A fused worker holds 2*read + out: 16^3 int32 -> 2*16 + 16 = 48 KiB, so a 192 KiB
    budget affords exactly 4 workers, and 191 KiB only 3."""
    arr = _src(tmp_path)
    for budget_kib, expected in ((192, 4), (191, 3)):
        io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8), overwrite=True,
                 region_shape=(16, 16, 16), zarr_format=2,
                 max_threads=32, memory_budget_mb=budget_kib / 1024)
        assert _fused_workers(capsys.readouterr().out) == expected


def test_fused_budget_counts_the_widest_dtype(tmp_path, capsys):
    """A narrowing chain carries its source dtype: int32 -> uint8 reads 16 KiB per region
    and writes 4 KiB, so a worker costs 2*16 + 4 = 36 KiB, not 4 KiB."""
    arr = _src(tmp_path).astype("uint8")
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2,
             max_threads=32, memory_budget_mb=72 / 1024)
    assert _fused_workers(capsys.readouterr().out) == 2


def test_fused_budget_counts_the_halo(tmp_path, capsys):
    """A halo op reads the padded region, and the budget must pay for that."""
    from dyna_zarr import operations as ops
    arr = ops.gaussian_filter(_src(tmp_path).astype("float32"), 1.0)
    depth = dyna_io_module._chain_halo_depth(arr, 3)
    assert depth and min(depth) > 0
    padded = int(np.prod([min(16 + 2 * d, 32) for d in depth])) * 4
    worker = 2 * padded + 16 ** 3 * 4
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2,
             max_threads=32, memory_budget_mb=4 * worker / 2**20)
    assert _fused_workers(capsys.readouterr().out) == 4


def test_fused_memory_budget_is_a_ceiling_not_a_target(tmp_path, capsys):
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2,
             max_threads=4, memory_budget_mb=4096)
    assert _fused_workers(capsys.readouterr().out) == 4


def test_fused_floor_is_one_worker(tmp_path, capsys):
    """Unlike the pipeline (floor 3 live regions), fused can run a single region."""
    arr = _src(tmp_path)
    io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8),
             region_shape=(16, 16, 16), zarr_format=2, max_threads=1)
    assert _fused_workers(capsys.readouterr().out) == 1


@pytest.mark.parametrize("engine", ["fused", "pipeline"])
def test_impossible_budget_warns_and_floors(tmp_path, engine):
    """A budget too small for the floor warns loudly instead of deadlocking."""
    arr = _src(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        io.write(arr, str(tmp_path / "o.zarr"), chunks=(8, 8, 8), engine=engine,
                 region_shape=(16, 16, 16), zarr_format=2,
                 max_threads=8, memory_budget_mb=0.001)
    msgs = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("cannot fit" in m for m in msgs), msgs
    # and it still produced correct output
    assert zarr.open(str(tmp_path / "o.zarr"), mode="r").shape == (32, 32, 32)


@pytest.mark.parametrize("engine,workers", [("fused", 1), ("pipeline", 3)])
def test_write_correct_at_the_floor(tmp_path, engine, workers):
    """Data must round-trip exactly at each engine's floor."""
    src = _src(tmp_path)
    expected = np.asarray(src[:, :, :])
    out = tmp_path / "o.zarr"
    io.write(src, str(out), chunks=(8, 8, 8), region_shape=(16, 16, 16),
             zarr_format=2, max_threads=workers, engine=engine)
    np.testing.assert_array_equal(zarr.open(str(out), mode="r")[...], expected)


# --- engine selection -----------------------------------------------------------------

def test_unknown_engine_raises_before_writing(tmp_path):
    arr = _src(tmp_path)
    with pytest.raises(ValueError, match="engine must be one of"):
        io.write(arr, str(tmp_path / "o.zarr"), engine="fast")
    assert not (tmp_path / "o.zarr").exists()


def test_num_readers_refused_under_fused(tmp_path):
    """The fused engine has no readers: refuse num_readers rather than ignore it."""
    arr = _src(tmp_path)
    with pytest.raises(ValueError, match="engine='pipeline' only"):
        io.write(arr, str(tmp_path / "o.zarr"), num_readers=2)
    assert not (tmp_path / "o.zarr").exists()


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
    with pytest.raises(ValueError, match="not a zarr store"):
        io.write(arr, str(victim), chunks=(8, 8, 8), zarr_format=2, overwrite=True)
    assert (victim / "precious.txt").read_text() == "keep me"
