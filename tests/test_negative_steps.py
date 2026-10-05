"""Negative-step slicing matches NumPy on every source.

``a[5:1:-1]`` used to pass the negative step straight to the source array: zarr raised
NegativeStepError. Composed/normalised negative slices also mark "before index 0" as
``stop=-1`` - a ``range()`` bound, but "the last element" in slice syntax - so passing
them on could read wrong data rather than fail. SliceTransform now reads the equivalent
positive-step range and flips the result, so upstream never sees a negative step.
"""
import contextlib
import io as _stdio

import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from dyna_zarr import DynamicArray, io, operations as ops

SHAPE = (6, 9, 11)
DATA = np.arange(np.prod(SHAPE), dtype="f4").reshape(SHAPE)


def _random_key(rng, shape):
    key = []
    for size in shape:
        c = rng.integers(0, 5)
        if c == 0:
            key.append(int(rng.integers(-size, size)))
        elif c == 1:
            key.append(slice(None, None, -int(rng.integers(1, 4))))
        elif c == 2:
            a, b = (int(v) for v in rng.integers(-size - 2, size + 2, size=2))
            key.append(slice(a, b, -int(rng.integers(1, 4))))
        elif c == 3:
            a, b = sorted(int(v) for v in rng.integers(0, size + 1, size=2))
            key.append(slice(a, b, int(rng.integers(1, 3))))
        else:
            key.append(slice(None))
    return tuple(key)


def _sources(tmp_path):
    z = zarr.create_array(str(tmp_path / "a.zarr"), shape=SHAPE, chunks=(2, 4, 5), dtype="f4")
    z[...] = DATA
    yield "zarr", DynamicArray(zarr.open_array(str(tmp_path / "a.zarr")))
    yield "tensorstore", io.read(str(tmp_path / "a.zarr"))
    yield "numpy", DynamicArray(DATA.copy())


@pytest.fixture
def sources(tmp_path):
    return dict(_sources(tmp_path))


@pytest.mark.parametrize("seed", range(4))
def test_random_negative_step_keys_match_numpy(sources, seed):
    rng = np.random.default_rng(seed)
    keys = [_random_key(rng, SHAPE) for _ in range(40)]
    for name, a in sources.items():
        for k in keys:
            got, want = a[k], DATA[k]
            assert got.shape == want.shape, (name, k)
            np.testing.assert_array_equal(np.asarray(got.compute()), want, err_msg=f"{name} {k}")


def test_chained_slices_with_negative_steps(sources):
    chains = [
        [(slice(None, None, -1),), (slice(1, 4),)],
        [(slice(5, 0, -2), slice(None, None, -1)), (slice(None, None, -1), 3)],
        [(slice(None), slice(8, 1, -3)), (1, slice(None, None, -1), slice(10, 2, -4))],
        [(slice(None, None, -1),) * 3, (slice(None, None, -1),) * 3],     # back to identity
        [(slice(2, 5),), (slice(None, None, -1), slice(4, None, -1))],
    ]
    for name, a in sources.items():
        for chain in chains:
            got, want = a, DATA
            for k in chain:
                got, want = got[k], want[k]
            np.testing.assert_array_equal(np.asarray(got.compute()), want, err_msg=f"{name} {chain}")


def test_empty_negative_selections(sources):
    for name, a in sources.items():
        for k in [(slice(1, 3, -1),), (slice(None), slice(0, 5, -2)), (slice(-1, -1, -1),)]:
            assert a[k].shape == DATA[k].shape, (name, k)
            assert np.asarray(a[k].compute()).size == 0


def test_generated_source_negative_steps():
    r = ops.random(SHAPE, seed=3)
    whole = r.compute()
    for k in [(slice(None, None, -1),), (slice(4, 0, -2), 7, slice(None, None, -3))]:
        np.testing.assert_array_equal(np.asarray(r[k].compute()), whole[k])


def test_ops_over_a_reversed_view(sources):
    a = sources["tensorstore"]
    rev = a[::-1, :, ::-2]
    ref = DATA[::-1, :, ::-2]
    np.testing.assert_allclose((rev * 2 + 1).compute(), ref * 2 + 1)
    np.testing.assert_allclose(ops.gaussian_filter(rev, 1).compute(),
                               ndi.gaussian_filter(ref, 1), rtol=1e-5)
    assert float(rev.max()) == ref.max()


def test_write_a_reversed_view_region_wise(tmp_path, sources):
    a = sources["tensorstore"]
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(a[::-1, 1:8, ::-1], str(tmp_path / "rev.zarr"), chunks=(2, 3, 5),
                 region_shape=(2, 3, 5))
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "rev.zarr"))[...],
                                  DATA[::-1, 1:8, ::-1])


@pytest.mark.zarrista
def test_zarrista_source_negative_steps(tmp_path):
    pytest.importorskip("zarrista")
    z = zarr.create_array(str(tmp_path / "a.zarr"), shape=SHAPE, chunks=(2, 4, 5), dtype="f4")
    z[...] = DATA
    a = io.read(str(tmp_path / "a.zarr"), backend="zarrista")
    rng = np.random.default_rng(9)
    for _ in range(40):
        k = _random_key(rng, SHAPE)
        np.testing.assert_array_equal(np.asarray(a[k].compute()), DATA[k], err_msg=str(k))
