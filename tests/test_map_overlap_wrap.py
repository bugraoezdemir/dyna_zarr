"""map_overlap with boundary="wrap" on the Python path: every region read equals scipy on
the whole array.

The halo stops at an array edge and the function applies the boundary mode itself,
which is right for every mode but wrap: scipy would wrap the block around itself, not
the array, so a region at an edge read the other side of its own block. A wrap read now
takes the halo from the far end of the axis (_wrapped_runs). A whole read was right
before, so the region reads are what these tests check.
"""
import numpy as np
import pytest
import scipy.ndimage as ndi

from dyna_zarr import DynamicArray, _engine
from dyna_zarr import operations as o
from dyna_zarr.operations.neighborhood import _wrapped_runs, map_overlap


@pytest.fixture(autouse=True)
def _python_path():
    saved = _engine._choice
    _engine.set_engine(False)
    yield
    _engine._choice = saved


def _keys(shape):
    a, b, c = shape
    return [
        (slice(None), slice(None), slice(None)),
        (slice(0, 2), slice(0, 5), slice(c - 4, c)),
        (slice(1, a - 1), slice(2, b - 3), slice(c // 3, c)),
        (a // 2, slice(None), slice(None, None, 2)),
        (slice(None, None, 3), b - 1, slice(1, 2)),
        (0, 0, slice(0, 5)),
        (slice(a - 1, a), slice(b - 2, b), slice(c - 3, c)),
        (-1, slice(b - 7, b), slice(0, 3)),
    ]


def _data(shape, dtype="float32", seed=0):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return rng.normal(scale=100, size=shape).astype(dtype)
    return rng.integers(0, 1000, size=shape).astype(dtype)


W = np.random.default_rng(3).uniform(-1, 1, (3, 2, 5))

#: name -> (dyna's filter, scipy's, dtype)
FILTERS = {
    "gaussian": (lambda a: o.gaussian_filter(a, (1.0, 2.0, 1.5), boundary="wrap"),
                 lambda d: ndi.gaussian_filter(d, (1.0, 2.0, 1.5), mode="wrap"), "float32"),
    "gaussian-large": (lambda a: o.gaussian_filter(a, (0.5, 4.0, 6.0), boundary="wrap"),
                       lambda d: ndi.gaussian_filter(d, (0.5, 4.0, 6.0), mode="wrap"),
                       "float64"),
    "uniform-int": (lambda a: o.uniform_filter(a, (3, 4, 5), boundary="wrap"),
                    lambda d: ndi.uniform_filter(d, (3, 4, 5), mode="wrap"), "int32"),
    "median": (lambda a: o.median_filter(a, 3, boundary="wrap"),
               lambda d: ndi.median_filter(d, 3, mode="wrap"), "uint16"),
    "minimum": (lambda a: o.minimum_filter(a, (2, 5, 3), boundary="wrap"),
                lambda d: ndi.minimum_filter(d, (2, 5, 3), mode="wrap"), "float32"),
    "laplace": (lambda a: o.laplace(a, boundary="wrap"),
                lambda d: ndi.laplace(d, mode="wrap"), "int16"),
    "convolve-even": (lambda a: o.convolve(a, W, boundary="wrap"),
                      lambda d: ndi.convolve(d, W, mode="wrap"), "float64"),
    "gradient": (lambda a: o.gaussian_gradient_magnitude(a, 1.2, boundary="wrap"),
                 lambda d: ndi.gaussian_gradient_magnitude(d, 1.2, mode="wrap"), "float32"),
}


@pytest.mark.parametrize("shape", [(7, 45, 38), (2, 24, 9)])
@pytest.mark.parametrize("name", list(FILTERS))
def test_wrap_regions_equal_scipy_on_the_whole_array(name, shape):
    build, ref, dtype = FILTERS[name]
    data = _data(shape, dtype)
    v = build(DynamicArray(data))
    want = ref(data)
    for key in _keys(shape):
        got = np.asarray(v[key].compute())
        np.testing.assert_array_equal(got, want[key], err_msg=f"{name} {key}")


def test_a_user_function_with_dask_periodic_wraps_around_the_array():
    """dask's name ("periodic") through DynamicArray.map_overlap, with align: the cells
    widen the core, the halo still wraps around the array."""
    data = _data((6, 30, 20), "float64", seed=1)
    f = lambda b: ndi.correlate1d(b, [1.0, -2.0, 0.5, 3.0], axis=2, mode="wrap")
    want = f(data)
    periodic = DynamicArray(data).map_overlap(f, depth=(0, 0, 2), boundary="periodic")
    aligned = map_overlap(DynamicArray(data), f, (0, 0, 2), boundary="wrap", align=4)
    for v in (periodic, aligned):
        for key in _keys(data.shape):
            np.testing.assert_array_equal(np.asarray(v[key].compute()), want[key])


def test_a_wrap_read_at_an_edge_reads_the_halo_not_the_axis():
    """Only the halo comes from the far end: the block is the core plus the depth on
    each side, not the whole axis."""
    seen = []

    def record(block):
        seen.append(block.shape)
        return block

    data = np.zeros((4, 200, 300), "float32")
    v = map_overlap(DynamicArray(data), record, (1, 3, 5), boundary="wrap", dtype="float32")
    v[:, 0:10, 290:300].compute()
    assert seen == [(4, 16, 20)]


@pytest.mark.parametrize("lo, hi, size, want", [
    (-3, 5, 10, [(7, 10), (0, 5)]),
    (6, 14, 10, [(6, 10), (0, 4)]),
    (2, 7, 10, [(2, 7)]),
    (1, 14, 10, [(1, 10), (0, 4)]),
    (-4, 9, 3, [(2, 3), (0, 3), (0, 3), (0, 3), (0, 3)]),
])
def test_wrapped_runs(lo, hi, size, want):
    runs = _wrapped_runs(lo, hi, size)
    assert [(r.start, r.stop) for r in runs] == want
    assert np.array_equal(np.concatenate([np.arange(size)[r] for r in runs]),
                          np.arange(lo, hi) % size)
