"""Arrays with NO chunk grid (chunks=None) behave like any other array.

reshape/flatten/pad/scan outputs - and broadcast results with no full-shape operand -
have no grid. Before: stack and swap_axes crashed on them; slicing and squeeze INVENTED a
(1, ..., 1) grid, which io.write then preserved (one-voxel chunks on disk); squeeze to 0-d
reported chunks (1,) for shape (); and the staged writers treated a grid-less source as
ONE whole-array chunk, so it was read in a single read whatever region_size_mb said.

Rule: no grid in, no grid out - a transform derives a valid grid of the right rank or
reports None, never an invented one; io.write picks its default chunk for None.
"""
import contextlib
import io as _stdio
import threading

import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from dyna_zarr import DynamicArray, io, operations as ops
from dyna_zarr.rechunk import staging_bytes

X = np.random.default_rng(0).random((6, 8, 10)).astype("f4")


def _gridded():
    return DynamicArray(zarr.array(X, chunks=(3, 4, 5)))


def _gridless():
    a = ops.reshape(ops.reshape(_gridded(), (6, 80)), (6, 8, 10))   # same data, no grid
    assert a.chunks is None
    return a


# (name, lazy op, numpy reference)
OPS = [
    ("slice", lambda a: a[1:5, ::2, 3], lambda x: x[1:5, ::2, 3]),
    ("slice_reversed", lambda a: a[::-1, 2:6], lambda x: x[::-1, 2:6]),
    ("newaxis", lambda a: a[None, 1:3], lambda x: x[None, 1:3]),
    ("expand_dims", lambda a: ops.expand_dims(a, 1), lambda x: np.expand_dims(x, 1)),
    ("squeeze", lambda a: ops.squeeze(a[2:3]), lambda x: np.squeeze(x[2:3])),
    ("squeeze_to_0d", lambda a: ops.squeeze(a[1:2, 3:4, 5:6]), lambda x: np.squeeze(x[1:2, 3:4, 5:6])),
    ("stack", lambda a: ops.stack([a, a + 1], axis=1), lambda x: np.stack([x, x + 1], axis=1)),
    ("concatenate", lambda a: ops.concatenate([a, a], axis=2), lambda x: np.concatenate([x, x], 2)),
    ("transpose", lambda a: ops.transpose(a, (2, 0, 1)), lambda x: x.transpose(2, 0, 1)),
    ("swap_axes", lambda a: ops.swap_axes(a, 0, 2), lambda x: np.swapaxes(x, 0, 2)),
    ("pad", lambda a: ops.pad(a, 1), lambda x: np.pad(x, 1)),
    ("tile", lambda a: ops.tile(a, (1, 2, 1)), lambda x: np.tile(x, (1, 2, 1))),
    ("roll", lambda a: ops.roll(a, 3, 2), lambda x: np.roll(x, 3, 2)),
    ("flip", lambda a: ops.flip(a, 1), lambda x: np.flip(x, 1)),
    ("rot90", lambda a: ops.rot90(a, 1, (1, 2)), lambda x: np.rot90(x, 1, (1, 2))),
    ("pointwise", lambda a: np.sqrt(a) * 2, lambda x: np.sqrt(x) * 2),
    ("broadcast", lambda a: a - a.mean(axis=0, keepdims=True), lambda x: x - x.mean(0, keepdims=True)),
    ("reduce", lambda a: ops.max(a, axis=1), lambda x: x.max(1)),
    ("gaussian", lambda a: ops.gaussian_filter(a, 1), lambda x: ndi.gaussian_filter(x, 1)),
    ("cumsum", lambda a: ops.cumsum(a, axis=2), lambda x: np.cumsum(x, 2)),
]
IDS = [o[0] for o in OPS]


@pytest.mark.parametrize("name, op, ref", OPS, ids=IDS)
def test_ops_on_a_gridless_array_neither_crash_nor_invent_a_grid(name, op, ref):
    r = op(_gridless())
    assert r.chunks is None, f"{name} invented a grid {r.chunks} for a grid-less input"
    np.testing.assert_allclose(np.asarray(r.compute()), ref(X), rtol=1e-5)


@pytest.mark.parametrize("name, op, ref", OPS, ids=IDS)
def test_gridded_results_have_a_grid_of_the_right_rank(name, op, ref):
    r = op(_gridded())
    if r.chunks is not None:
        assert len(r.chunks) == r.ndim, f"{name}: chunks {r.chunks} for shape {r.shape}"
        assert all(int(c) >= 1 for c in r.chunks)
    np.testing.assert_allclose(np.asarray(r.compute()), ref(X), rtol=1e-5)


@pytest.mark.parametrize("name, op, ref", [o for o in OPS if o[0] != "squeeze_to_0d"],
                         ids=[i for i in IDS if i != "squeeze_to_0d"])
def test_gridless_results_write_with_real_default_chunks(tmp_path, name, op, ref):
    r = op(_gridless())
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(r, str(tmp_path / "out.zarr"))
    z = zarr.open_array(str(tmp_path / "out.zarr"))
    np.testing.assert_allclose(z[...], ref(X), rtol=1e-5)
    # the invented (1, ..., 1) grid used to be preserved: one-voxel chunks on disk
    assert int(np.prod(z.chunks)) > 1 or int(np.prod(z.shape)) <= 1, z.chunks


def test_concatenate_and_stack_keep_a_grid_only_when_all_inputs_share_it():
    # They used to take arrays[0]'s grid, so the result depended on argument order.
    g1 = _gridded()
    g2 = DynamicArray(zarr.array(X, chunks=(2, 8, 5)))              # a different grid
    none = _gridless()
    assert ops.concatenate([g1, g1]).chunks == (3, 4, 5)
    assert ops.stack([g1, g1], axis=1).chunks == (3, 1, 4, 5)
    for pair in ([g1, g2], [g2, g1], [g1, none], [none, g1]):
        assert ops.concatenate(pair).chunks is None, pair
        assert ops.stack(pair).chunks is None, pair
        np.testing.assert_array_equal(ops.concatenate(pair).compute(), np.concatenate([X, X]))


# --------------------------------------------------------------------------- #
# staged writes of grid-less sources stay within the memory budget
# --------------------------------------------------------------------------- #

@pytest.fixture
def largest_source_read(monkeypatch):
    lock = threading.Lock()
    biggest = {"n": 0}
    orig = DynamicArray._read_direct

    def rd(self, key):
        out = orig(self, key)
        if self._transform is None:
            with lock:
                biggest["n"] = max(biggest["n"], int(np.asarray(out).size))
        return out

    monkeypatch.setattr(DynamicArray, "_read_direct", rd)
    return biggest


@pytest.mark.parametrize("build", [
    ("flatten of grid-less", lambda a: ops.flatten(ops.reshape(a, (32, 256 * 256)) + 0)),
    ("reshape of grid-less", lambda a: ops.reshape(ops.pad(a, 0), (64, 128 * 256))),
    ("reshape of gridded", lambda a: ops.reshape(a, (64, 128 * 256))),
], ids=lambda b: b[0])
def test_staged_writes_read_within_the_budget(tmp_path, largest_source_read, build):
    v = np.random.default_rng(1).random((32, 256, 256)).astype("f4")          # 8 MiB
    z = zarr.create_array(str(tmp_path / "a.zarr"), shape=v.shape, chunks=(8, 64, 64), dtype="f4")
    z[...] = v
    lazy = build[1](io.read(str(tmp_path / "a.zarr")))
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(lazy, str(tmp_path / "out.zarr"), region_size_mb=1)
    # Grid-less sources used to be read whole (8 MiB) in one read; reshape's staged 1-D
    # copy used a fixed 16 MiB chunk. Every read is now at most one 1 MiB budget.
    assert largest_source_read["n"] * 4 <= 1 << 20
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "out.zarr"))[...],
                                  v.reshape(lazy.shape))


def test_gridless_source_needs_no_rechunk_intermediate():
    a = ops.reshape(ops.reshape(DynamicArray(zarr.array(np.zeros((34, 258, 258), "f4"),
                                                        chunks=(8, 64, 64))),
                                (34, 258 * 258)), (34, 258, 258))
    assert a.chunks is None
    # read in the staging grid itself, so the grids nest (an arbitrary default grid
    # against 258-wide rows made (1, 2, 2) intermediate chunks - millions of files)
    assert staging_bytes("flatten", a, max_mem=1 << 20) == int(np.prod(a.shape)) * 4
