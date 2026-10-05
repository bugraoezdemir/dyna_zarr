"""ops.pad: exact against numpy.pad, and bounded reads at the borders.

A window touching a padded border used to read the WHOLE input axis (pad(a)[0:3] read
all of `a`), so every edge region of a padded write pulled whole axes. For the copy modes
(constant scalar / empty / edge / reflect / symmetric / wrap) each padded position maps
to one input index, so a window now reads only the span of input it maps to. Statistic
modes (mean/median/maximum/minimum/linear_ramp), reflect_type='odd' and per-axis
constants genuinely need whole axes and keep that behaviour.
"""
import threading

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, operations as ops

COPY_MODES = [("constant", {}), ("constant", {"constant_values": 7.5}), ("edge", {}),
              ("reflect", {}), ("symmetric", {}), ("wrap", {})]
WHOLE_AXIS_MODES = [("mean", {}), ("median", {}), ("maximum", {}), ("minimum", {}),
                    ("linear_ramp", {"end_values": 3}), ("reflect", {"reflect_type": "odd"}),
                    ("constant", {"constant_values": ((1, 2), (3, 4), (5, 6))})]


def _random_key(rng, shape):
    key = []
    for size in shape:
        c = rng.integers(0, 4)
        if c == 0:
            key.append(int(rng.integers(0, size)))
        elif c == 1:
            key.append(slice(None))
        else:
            a, b = sorted(int(v) for v in rng.integers(0, size + 1, size=2))
            key.append(slice(a, b, int(rng.integers(1, 3))))
    return tuple(key)


@pytest.mark.parametrize("mode, kw", COPY_MODES + WHOLE_AXIS_MODES,
                         ids=[f"{m}{'-' + next(iter(k)) if k else ''}" for m, k in COPY_MODES + WHOLE_AXIS_MODES])
@pytest.mark.parametrize("pad_width", [1, ((2, 0), (0, 3), (1, 1)), ((9, 4), (12, 2), (3, 15))],
                         ids=["uniform", "asymmetric", "wider-than-axis"])
def test_pad_matches_numpy_on_random_windows(mode, kw, pad_width):
    x = np.random.default_rng(0).random((5, 7, 6)).astype("f4")
    a = DynamicArray(zarr.array(x, chunks=(2, 3, 4)))
    ref = np.pad(x, pad_width, mode=mode, **kw)
    p = ops.pad(a, pad_width, mode=mode, **kw)
    assert p.shape == ref.shape
    np.testing.assert_allclose(p.compute(), ref, rtol=1e-5)
    rng = np.random.default_rng(1)
    for _ in range(60):
        k = _random_key(rng, ref.shape)
        np.testing.assert_allclose(np.asarray(p[k].compute()), ref[k], rtol=1e-5, err_msg=str(k))


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("mode, kw", COPY_MODES, ids=[m + ("-value" if k else "") for m, k in COPY_MODES])
def test_pad_low_rank(ndim, mode, kw):
    x = np.arange(np.prod((9,) * ndim), dtype="f4").reshape((9,) * ndim)
    a = DynamicArray(zarr.array(x, chunks=(4,) * ndim))
    ref = np.pad(x, 3, mode=mode, **kw)
    rng = np.random.default_rng(2)
    for _ in range(40):
        k = _random_key(rng, ref.shape)
        np.testing.assert_allclose(np.asarray(ops.pad(a, 3, mode=mode, **kw)[k].compute()), ref[k])


@pytest.fixture
def source_reads(monkeypatch):
    lock = threading.Lock()
    reads = []
    orig = DynamicArray._read_direct

    def rd(self, key):
        out = orig(self, key)
        if self._transform is None:
            with lock:
                reads.append(np.asarray(out).shape)
        return out

    monkeypatch.setattr(DynamicArray, "_read_direct", rd)
    return reads


@pytest.mark.parametrize("mode, kw", [m for m in COPY_MODES if m[0] != "wrap"],
                         ids=[m[0] + ("-value" if m[1] else "") for m in COPY_MODES if m[0] != "wrap"])
def test_bordered_windows_read_only_what_they_need(source_reads, mode, kw):
    x = np.random.default_rng(3).random((32, 64, 64)).astype("f4")
    a = DynamicArray(zarr.array(x, chunks=(8, 16, 16)))
    p = ops.pad(a, 2, mode=mode, **kw)
    ref = np.pad(x, 2, mode=mode, **kw)
    for key in [(slice(0, 4),), (slice(0, 4), slice(0, 6), slice(60, 68)),
                (slice(34, 36), 0, slice(None, 5))]:
        source_reads.clear()
        np.testing.assert_allclose(np.asarray(p[key].compute()), ref[key], rtol=1e-6)
        (shape,) = source_reads
        # Along every axis the input read spans at most as many positions as the window
        # asks for there (copy modes map monotonically, wrap aside). It used to read the
        # WHOLE 32/64-long input axis for any window touching the border.
        full = key + (slice(None),) * (ref.ndim - len(key))
        for a, (k, read_len) in enumerate(zip(full, shape)):
            asked = 1 if isinstance(k, int) else len(range(*k.indices(ref.shape[a])))
            assert read_len <= max(1, asked), (key, a, shape)


def test_wrap_reads_only_the_far_edge_it_needs(source_reads):
    x = np.random.default_rng(4).random((64, 8)).astype("f4")
    a = DynamicArray(zarr.array(x, chunks=(16, 8)))
    p = ops.pad(a, ((3, 0), (0, 0)), mode="wrap")
    np.testing.assert_allclose(np.asarray(p[0:3].compute()), np.pad(x, ((3, 0), (0, 0)), mode="wrap")[0:3])
    assert source_reads == [(3, 8)]                     # rows 61..63 only


def test_whole_axis_modes_still_read_the_whole_axis(source_reads):
    x = np.random.default_rng(5).random((32, 8)).astype("f4")
    a = DynamicArray(zarr.array(x, chunks=(8, 8)))
    np.testing.assert_allclose(np.asarray(ops.pad(a, 1, mode="mean")[0:3].compute()),
                               np.pad(x, 1, mode="mean")[0:3], rtol=1e-6)
    assert source_reads[0][0] == 32                     # the statistic needs every row


def test_padded_write_is_bounded(tmp_path, source_reads):
    import contextlib
    import io as _stdio
    from dyna_zarr import io
    x = np.random.default_rng(6).random((32, 256, 256)).astype("f4")          # 8 MiB
    z = zarr.create_array(str(tmp_path / "a.zarr"), shape=x.shape, chunks=(8, 64, 64), dtype="f4")
    z[...] = x
    a = io.read(str(tmp_path / "a.zarr"))
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(ops.pad(a, 1, mode="reflect"), str(tmp_path / "p.zarr"),
                 chunks=(8, 64, 64), region_size_mb=1)
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "p.zarr"))[...],
                               np.pad(x, 1, mode="reflect"))
    # every read is about one region, never a whole axis of the 8 MiB array
    assert max(int(np.prod(s)) for s in source_reads) * 4 <= 2 << 20
