"""``...`` in a DynamicArray index behaves as in NumPy (it used to raise TypeError)."""
import numpy as np
import pytest

import dyna_zarr

X = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)

KEYS = [Ellipsis, (Ellipsis,), (Ellipsis, 1), (0, Ellipsis), (0, Ellipsis, 2),
        (Ellipsis, slice(1, 3)), (1, Ellipsis, slice(None, None, 2)), (Ellipsis, None),
        (None, Ellipsis, 0), (1, 2, 3, 4, Ellipsis), (slice(1, 2), Ellipsis, -1)]


@pytest.mark.parametrize("key", KEYS, ids=str)
def test_matches_numpy(key):
    a = dyna_zarr.from_array(X, chunks=(1, 2, 2, 5))
    out = a[key]
    assert out.shape == X[key].shape
    np.testing.assert_array_equal(np.asarray(out), X[key])


def test_two_ellipses_raise_like_numpy():
    a = dyna_zarr.from_array(X)
    with pytest.raises(IndexError, match="single ellipsis"):
        a[..., 0, ...]


def test_too_many_indices_with_ellipsis():
    a = dyna_zarr.from_array(X)
    with pytest.raises(IndexError):
        a[0, 0, 0, 0, 0, ...]
