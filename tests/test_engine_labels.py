"""Labels of a tiled mask (tilewise-ccl's Phase B) read by the native engine (Workplan
WP15, task DR7 step 8).

The engine's labeler numbers components as scipy.ndimage.label does; label_tiles maps
each tile's local labels through its table. Engine and Python path must give the same
labels, dtype and shape, whole and region by region and written by io.write, and equal
a per-tile scipy reference.
"""
import itertools

import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from dyna_zarr import _engine, _lowering, io
from dyna_zarr import operations as o

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")
if not hasattr(core.Array, "label_tiles"):
    pytest.skip("this dyna-zarr-core has no tiled labels", allow_module_level=True)


@pytest.fixture(autouse=True)
def _engine_on():
    saved = _engine._choice
    _engine.set_engine(True)
    yield
    _engine._choice = saved


# ---------------------------------------------------------------------------- the labeler

def _engine_label(fg, connectivity):
    b, n = core.filters.label(np.ascontiguousarray(fg, dtype=np.uint8).tobytes(),
                              list(fg.shape), connectivity)
    return np.frombuffer(b, dtype=np.uint32).reshape(fg.shape), n


@pytest.mark.parametrize("shape", [(1,), (17,), (1, 9), (13, 11), (2, 1, 7), (9, 10, 11),
                                   (3, 4, 5, 6), (64, 64)])
@pytest.mark.parametrize("density", [0.1, 0.45, 0.6, 0.95])
def test_labels_equal_scipy(shape, density):
    """Every connectivity of the rank, sparse to dense masks: scipy's labels and count."""
    rng = np.random.default_rng(int(density * 100) + len(shape))
    fg = rng.random(shape) < density
    for connectivity in range(0, len(shape) + 1):
        struct = ndi.generate_binary_structure(len(shape), connectivity)
        want, n = ndi.label(fg, structure=struct)
        got, m = _engine_label(fg, connectivity)
        assert m == n, connectivity
        np.testing.assert_array_equal(got.astype(np.int64), want, err_msg=f"{connectivity}")


def test_spirals_and_late_merges_number_as_scipy():
    """Components that merge late in the scan (U shapes, spirals) keep scipy's order."""
    fg = np.zeros((40, 40), bool)
    for k in range(0, 20, 4):
        fg[k, k:40 - k] = fg[k:40 - k, 40 - k - 1] = True
        fg[40 - k - 1, k + 2:40 - k] = fg[k + 2:40 - k, k + 2] = True
    fg[::7, ::5] = True
    for c in (1, 2):
        want, n = ndi.label(fg, structure=ndi.generate_binary_structure(2, c))
        got, m = _engine_label(fg, c)
        assert m == n
        np.testing.assert_array_equal(got.astype(np.int64), want)


# ------------------------------------------------------------------------ through dyna

SHAPE = (9, 70, 61)


def _zarr(path, data, chunks=(3, 16, 16)):
    z = zarr.create_array(str(path), shape=data.shape, chunks=chunks, dtype=data.dtype)
    z[...] = data
    return io.read(str(path))


def _tables(fg, tile, connectivity, dtype, seed=0):
    """A table per tile, as Phase A would give: its local labels mapped to final ones
    (here distinct random ids, background 0), and the per-tile reference labels."""
    rng = np.random.default_rng(seed)
    struct = ndi.generate_binary_structure(fg.ndim, connectivity)
    grid = [-(-s // t) for s, t in zip(fg.shape, tile)]
    tables, want = {}, np.zeros(fg.shape, dtype)
    for tidx in itertools.product(*[range(g) for g in grid]):
        sl = tuple(slice(i * t, min((i + 1) * t, s)) for i, t, s in zip(tidx, tile, fg.shape))
        local, n = ndi.label(fg[sl], structure=struct)
        tb = np.zeros(n + 1, dtype)
        tb[1:] = rng.integers(1, 2**30, size=n)
        tables[tidx] = tb
        want[sl] = tb[local]
    return tables, want


def _keys(shape):
    a, b, c = shape
    return [
        (slice(None), slice(None), slice(None)),
        (slice(1, a - 1), slice(5, b - 3), slice(c // 3, c)),
        (a // 2, slice(None), slice(None, None, 2)),
        (slice(None, None, 3), b - 1, slice(1, 2)),
        (slice(a - 1, a), slice(b - 2, b), slice(c - 3, c)),
    ]


def _both(fn):
    _engine.set_engine(True)
    e = fn()
    _engine.set_engine(False)
    p = fn()
    _engine.set_engine(True)
    return e, p


MASKS = {
    "threshold": lambda d, tmp: _zarr(tmp / "u.zarr", d) > 128,
    "uint8": lambda d, tmp: _zarr(tmp / "u.zarr", (d > 128).astype("uint8") * d),
    "float": lambda d, tmp: _zarr(tmp / "f.zarr", np.where(d > 128, d / 7.0, -d - 0.0)
                                  .astype("float32")),
}


@pytest.mark.parametrize("dtype", ["int32", "int64"])
@pytest.mark.parametrize("tile", [(3, 32, 32), (9, 70, 61), (4, 17, 23), (1, 64, 8)])
@pytest.mark.parametrize("connectivity", [1, 2, 3])
@pytest.mark.parametrize("mask", list(MASKS))
def test_label_tiles_reads_as_python_and_scipy(tmp_path, mask, connectivity, tile, dtype):
    d = np.random.default_rng(3).integers(0, 256, SHAPE, dtype=np.uint8)
    d = ndi.uniform_filter(d, 3)                      # blobs, not salt and pepper
    m = MASKS[mask](d, tmp_path)
    fg = np.asarray(m.compute()) > 0
    tables, want = _tables(fg, tile, connectivity, dtype)
    v = o.label_tiles(m, tables, tile, connectivity, dtype)
    assert _lowering.lowered(v) is not None
    for key in _keys(SHAPE):
        e, p = _both(lambda: np.asarray(v._read_direct(key)))
        assert e.dtype == p.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(e, p, err_msg=f"{key}")
        np.testing.assert_array_equal(e, want[key], err_msg=f"{key}")


def test_label_tiles_written_as_python(tmp_path):
    """io.write with output chunks unlike the tiles and a small budget: the same labels."""
    d = ndi.uniform_filter(np.random.default_rng(5).integers(0, 256, (12, 96, 80),
                                                              dtype=np.uint8), 3)
    m = _zarr(tmp_path / "a.zarr", d, chunks=(4, 32, 32)) > 120
    tables, want = _tables(np.asarray(m.compute()), (6, 48, 40), 2, "int32")
    v = o.label_tiles(m, tables, (6, 48, 40), 2)
    out = {}
    for on in (True, False):
        _engine.set_engine(on)
        path = tmp_path / f"out-{on}.zarr"
        io.write(v, str(path), chunks=(3, 24, 20), max_threads=3, memory_budget_mb=24)
        out[on] = zarr.open_array(str(path), mode="r")[...]
    np.testing.assert_array_equal(out[True], out[False])
    np.testing.assert_array_equal(out[True], want)


def test_tables_that_do_not_fit_fail_on_both_paths(tmp_path):
    d = np.zeros((4, 10, 10), np.uint8)
    d[:, 1, 1] = d[:, 5, 5] = 200                      # two components per tile
    m = _zarr(tmp_path / "a.zarr", d, chunks=(2, 10, 10)) > 0
    tables = {(i, 0, 0): np.array([0, 9], np.int32) for i in range(2)}
    v = o.label_tiles(m, tables, (2, 10, 10), 1)
    assert _lowering.lowered(v) is not None
    with pytest.raises(core.ReadError, match="tables do not come from this mask"):
        v.compute()
    _engine.set_engine(False)
    with pytest.raises(IndexError):
        v.compute()


def test_label_tiles_refuses_wrong_arguments(tmp_path):
    m = _zarr(tmp_path / "a.zarr", np.zeros((4, 10, 10), np.uint8)) > 0
    with pytest.raises(ValueError, match="tiles of"):
        o.label_tiles(m, {}, (2, 10), 1)
    with pytest.raises(ValueError, match="int32 or int64"):
        o.label_tiles(m, {}, (2, 10, 10), 1, dtype="float32")
