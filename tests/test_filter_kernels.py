"""The engine's filter kernels against scipy.ndimage (Workplan DR7, decision DR-3 a).

The kernels live in the Rust crate dyna-zarr-filters; dyna_zarr_core.filters runs them
on whole NumPy arrays. They must give scipy's values bit for bit: every mode, every
numeric type, odd and short shapes, any axis, symmetric, antisymmetric and general
weights, and any number of threads.

Integers out of range: scipy's C casts wrap on x86-64 (a uint8 400.0 gives 144) and
saturate on ARM. The kernels follow x86-64, so cases whose float result leaves the
output type's range are compared on x86-64 only.
"""
import platform

import numpy as np
import pytest
import scipy.ndimage as ndi
from scipy.ndimage._filters import _gaussian_kernel1d

core = pytest.importorskip("dyna_zarr_core")
if not hasattr(core, "filters"):
    pytest.skip("this dyna-zarr-core has no filters", allow_module_level=True)
filters = core.filters

pytestmark = pytest.mark.engine

X86 = platform.machine().lower() in ("x86_64", "amd64", "x86", "i686")
MODES = ["reflect", "mirror", "nearest", "wrap", "constant"]
TYPES = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64",
         "float32", "float64"]
SHAPES = [(1,), (2,), (3,), (13,), (1, 7), (2, 3), (5, 1, 4), (3, 4, 5), (2, 1, 3, 2, 3)]


def _data(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return (rng.standard_normal(shape) * 50).astype(dtype)
    info = np.iinfo(dtype)
    lo, hi = max(info.min, -1000), min(info.max, 1000)
    return rng.integers(lo, hi, size=shape, endpoint=True).astype(dtype)


def _weights(kind, size, seed):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1, 1, size)
    if kind == "symmetric":
        w = (w + w[::-1]) / 2
    elif kind == "antisymmetric":
        w = (w - w[::-1]) / 2
    return w


def _out_of_range(x, w, axis, mode, cval, origin, dtype):
    """Whether scipy's float result leaves the integer type's range (x86-64 only)."""
    if np.dtype(dtype).kind == "f":
        return False
    f = ndi.correlate1d(x, w, axis, output=np.float64, mode=mode, cval=cval, origin=origin)
    info = np.iinfo(dtype)
    return bool((f < info.min).any() or (f >= info.max + 1.0).any())


def _ours_1d(x, w, axis, mode, cval=0.0, origin=0, out_dtype=None, threads=1):
    out = np.empty(x.shape, out_dtype or x.dtype)
    filters.correlate1d(x, list(w), axis, out, mode=mode, cval=cval, origin=origin,
                        threads=threads)
    return out


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind, size", [("symmetric", 1), ("symmetric", 5),
                                        ("antisymmetric", 3), ("general", 4),
                                        ("general", 7), ("symmetric", 31)])
def test_correlate1d_equals_scipy(dtype, mode, kind, size):
    w = _weights(kind, size, seed=size)
    for k, shape in enumerate(SHAPES):
        x = _data(shape, dtype, seed=k)
        for axis in range(len(shape)):
            for origin in sorted({0, -(size // 2), (size - 1) // 2}):
                cval = 3.5
                if not X86 and _out_of_range(x, w, axis, mode, cval, origin, dtype):
                    continue
                want = ndi.correlate1d(x, w, axis, mode=mode, cval=cval, origin=origin)
                got = _ours_1d(x, w, axis, mode, cval, origin)
                np.testing.assert_array_equal(got, want, err_msg=f"{shape} axis {axis} "
                                              f"origin {origin}")


@pytest.mark.parametrize("dtype", ["uint8", "int16", "float32"])
def test_float_output_of_an_integer_or_float32_input(dtype):
    x = _data((6, 9), dtype, seed=3)
    w = _weights("general", 4, seed=4)
    for out in ("float32", "float64"):
        want = ndi.correlate1d(x, w, 1, output=np.dtype(out), mode="mirror")
        np.testing.assert_array_equal(_ours_1d(x, w, 1, "mirror", out_dtype=out), want)


def test_threads_change_nothing():
    x = _data((40, 30, 50), "float32", seed=5)
    w = _weights("symmetric", 9, seed=6)
    for axis in range(3):
        one = _ours_1d(x, w, axis, "reflect")
        np.testing.assert_array_equal(_ours_1d(x, w, axis, "reflect", threads=4), one)
        np.testing.assert_array_equal(one, ndi.correlate1d(x, w, axis))


@pytest.mark.skipif(not X86, reason="scipy saturates out-of-range casts on ARM")
def test_integer_overflow_wraps_as_scipy():
    x = np.array([0, 0, 200, 255, 0, 0, 255], np.uint8)
    for w in ([-1.0, 0.0, 1.0], [2.0, 2.0, 2.0], [-3.0, 1.0]):
        np.testing.assert_array_equal(_ours_1d(x, w, 0, "reflect"), ndi.correlate1d(x, w))


def _gaussian_with_scipy_weights(x, sigma, order, mode, truncate=4.0, out_dtype=None):
    """Gaussian as the engine will run it: scipy's own weights, one pass per axis."""
    cur = x
    out = np.empty(x.shape, out_dtype or x.dtype)
    passes = [a for a in range(x.ndim) if sigma[a] > 1e-15]
    if not passes:
        out[...] = x
        return out
    for a in passes:
        w = _gaussian_kernel1d(sigma[a], order[a], int(truncate * sigma[a] + 0.5))[::-1]
        nxt = np.empty(x.shape, out.dtype)
        filters.correlate1d(cur, list(w), a, nxt, mode=mode[a])
        cur = nxt
    return cur


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "float32", "float64"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sigma, order", [((2.0, 0.0, 1.5), (0, 0, 0)),
                                          ((1.0, 1.0, 1.0), (0, 1, 0)),
                                          ((0.7, 3.0, 0.4), (2, 0, 1)),
                                          ((0.0, 0.0, 0.0), (0, 0, 0))])
def test_gaussian_with_scipy_weights_equals_scipy(dtype, mode, sigma, order):
    x = _data((5, 17, 11), dtype, seed=7)
    if not X86 and np.dtype(dtype).kind != "f" and any(order):
        pytest.skip("derivatives leave the range of integer types")
    want = ndi.gaussian_filter(x, sigma, order=order, mode=mode)
    got = _gaussian_with_scipy_weights(x, sigma, order, [mode] * 3)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("sigma, order, radius", [(0.3, 0, 1), (1.0, 0, 4), (2.0, 0, 8),
                                                  (5.5, 0, 22), (20.0, 0, 80), (1.5, 1, 6),
                                                  (2.0, 2, 8), (3.0, 3, 12)])
def test_own_kernel_is_scipys_to_the_last_bits(sigma, order, radius):
    """The crate's own kernel. scipy builds it with numpy's exp and, for derivatives, a
    BLAS product, both of which can differ in the last bits between machines (on a
    Windows x86-64 machine with OpenBLAS, order 2 was fused into a multiply-add while
    orders 0, 1 and 3 matched exactly), so the engine passes scipy's own weights. The
    crate's stay within 2 ulp."""
    want = _gaussian_kernel1d(sigma, order, radius)[::-1]
    got = np.array(filters.gaussian_weights(sigma, order, radius))
    np.testing.assert_array_max_ulp(got, want, maxulp=2)


@pytest.mark.parametrize("n", [1, 2, 7, 8, 9, 16, 17, 127, 128, 129, 200, 513, 1000])
def test_numpy_sum_is_numpys(n):
    a = np.random.default_rng(n).standard_normal(n) * 10 ** np.arange(n) % 7
    assert filters.numpy_sum(list(a)) == a.sum()


def test_numpy_sum_of_negative_zeros():
    """numpy starts from 0.0, so negative zeros sum to 0.0, not -0.0."""
    for n in (1, 3, 9, 200):
        want = np.full(n, -0.0).sum()
        got = filters.numpy_sum([-0.0] * n)
        assert (got, np.signbit(got)) == (want, np.signbit(want)), n


def test_own_gaussian_filter_matches_scipy():
    x = _data((6, 20, 15), "float64", seed=8)
    out = np.empty_like(x)
    filters.gaussian_filter(x, [1.0, 2.0, 0.0], out, [0, 0, 0], ["reflect"] * 3)
    np.testing.assert_allclose(out, ndi.gaussian_filter(x, [1.0, 2.0, 0.0]), rtol=1e-13,
                               atol=1e-12)


def test_errors_name_the_way_out():
    x = np.zeros((4, 5), np.float32)
    with pytest.raises(filters.FilterError, match="share memory"):
        filters.correlate1d(x, [1.0], 0, x)
    with pytest.raises(filters.FilterError, match="output int16"):
        filters.correlate1d(x, [1.0], 0, np.zeros((4, 5), np.int16))
    with pytest.raises(filters.FilterError, match="origin"):
        filters.correlate1d(x, [1.0, 1.0, 1.0], 0, np.zeros_like(x), origin=2)
    with pytest.raises(filters.FilterError, match="mode"):
        filters.correlate1d(x, [1.0], 0, np.zeros_like(x), mode="reflect101")
    with pytest.raises(filters.FilterError, match="C-contiguous"):
        filters.correlate1d(x.T, [1.0], 0, np.zeros((5, 4), np.float32))
    with pytest.raises(filters.FilterError, match="float16 is not supported"):
        filters.correlate1d(x.astype("float16"), [1.0], 0, np.zeros((4, 5), np.float16))


# ----------------------------------------------------------------- step 3 (DR7)
# Passes are given as (axis, kind, values, origin, mode) to filters.chains.


def _same_bits(got, want, msg=""):
    """Equal values, NaN where NaN, and the same sign of zero."""
    np.testing.assert_array_equal(got, want, err_msg=msg)
    if got.dtype.kind == "f":
        np.testing.assert_array_equal(np.signbit(got), np.signbit(want), err_msg=msg)


def _floats_with_specials(shape, dtype, seed):
    x = _data(shape, dtype, seed)
    flat = x.reshape(-1)
    if flat.size > 4:
        flat[1], flat[flat.size // 2] = np.nan, -0.0
        flat[-2] = 0.0
    return x


def _chains_out(x, chains, combine=None, threads=1):
    out = np.empty(x.shape, x.dtype)
    filters.chains(x, chains, out, combine=combine, threads=threads)
    return out


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("kind", ["minimum", "maximum"])
@pytest.mark.parametrize("size, origin", [(1, 0), (2, 0), (3, 0), (4, -1), (5, 2), (8, -4)])
def test_minimum_and_maximum_equal_scipy(dtype, mode, kind, size, origin):
    """Sliding minima and maxima, NaN and signed zeros included (floats)."""
    f1d = ndi.minimum_filter1d if kind == "minimum" else ndi.maximum_filter1d
    for k, shape in enumerate(SHAPES):
        make = _floats_with_specials if np.dtype(dtype).kind == "f" else _data
        x = make(shape, dtype, seed=k)
        for axis in range(len(shape)):
            if size // 2 + origin >= size or size // 2 + origin < 0:
                continue
            want = f1d(x, size, axis, mode=mode, cval=3.5, origin=origin)
            out = np.empty_like(x)
            filters.chains(x, [[(axis, kind, [size], origin, mode)]], out, cval=3.5)
            _same_bits(out, want, f"{shape} axis {axis}")


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("mode", MODES)
def test_separable_minimum_maximum_and_grey_morphology_equal_scipy(dtype, mode):
    make = _floats_with_specials if np.dtype(dtype).kind == "f" else _data
    x = make((5, 9, 12), dtype, seed=11)
    sizes = (2, 3, 4)
    mn = _chains_out(x, [[(a, "minimum", [s], 0, mode) for a, s in enumerate(sizes)]])
    _same_bits(mn, ndi.minimum_filter(x, sizes, mode=mode))
    _same_bits(mn, ndi.grey_erosion(x, size=sizes, mode=mode))
    # grey_dilation: the origin negated, one lower for even sizes
    dil = [(a, "maximum", [s], -(1 - s % 2), mode) for a, s in enumerate(sizes)]
    _same_bits(_chains_out(x, [dil]), ndi.grey_dilation(x, size=sizes, mode=mode))
    mx = _chains_out(x, [[(a, "maximum", [s], 0, mode) for a, s in enumerate(sizes)]])
    _same_bits(mx, ndi.maximum_filter(x, sizes, mode=mode))


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size, origin", [(2, 0), (3, 0), (4, -1), (5, 1), (9, 0)])
def test_uniform_equals_scipy(dtype, mode, size, origin):
    """The running sum in scipy's order: equal for floats too on the whole array."""
    for k, shape in enumerate(SHAPES):
        x = _data(shape, dtype, seed=k)
        for axis in range(len(shape)):
            if not X86 and _out_of_range(x, np.full(size, 1 / size), axis, mode, 3.5,
                                         origin, dtype):
                continue
            want = ndi.uniform_filter1d(x, size, axis, mode=mode, cval=3.5, origin=origin)
            out = np.empty_like(x)
            filters.chains(x, [[(axis, "uniform", [size], origin, mode)]], out, cval=3.5)
            _same_bits(out, want, f"{shape} axis {axis}")


@pytest.mark.parametrize("dtype", TYPES)
@pytest.mark.parametrize("mode", MODES)
def test_laplace_equals_scipy(dtype, mode):
    """generic_laplace: one [1, -2, 1] pass per axis on the input, added in the type."""
    for k, shape in enumerate(SHAPES):
        x = _data(shape, dtype, seed=k)
        if not X86 and any(_out_of_range(x, [1, -2, 1], a, mode, 3.5, 0, dtype)
                           for a in range(x.ndim)):
            continue
        chains = [[(a, "correlate", [1.0, -2.0, 1.0], 0, mode)] for a in range(x.ndim)]
        out = np.empty_like(x)
        filters.chains(x, chains, out, combine="sum", cval=3.5)
        _same_bits(out, ndi.laplace(x, mode=mode, cval=3.5), f"{shape}")


def _gaussian_chain(ndim, sigma, order, mode, truncate=4.0):
    """scipy's gaussian_filter as passes, with scipy's own weights."""
    chain = []
    for a in range(ndim):
        if sigma[a] > 1e-15:
            w = _gaussian_kernel1d(sigma[a], order[a], int(truncate * float(sigma[a]) + 0.5))
            chain.append((a, "correlate", [float(v) for v in w[::-1]], 0, mode))
    return chain


@pytest.mark.parametrize("dtype", ["uint8", "int16", "uint16", "int32", "float32", "float64"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sigma", [(1.0, 1.5, 2.0), (0.0, 1.0, 0.7)])
def test_gaussian_laplace_equals_scipy(dtype, mode, sigma):
    x = _data((5, 19, 14), dtype, seed=12)
    if not X86 and np.dtype(dtype).kind != "f":
        pytest.skip("second derivatives leave the range of integer types")
    chains = []
    for d in range(3):
        order = [2 if a == d else 0 for a in range(3)]
        chains.append(_gaussian_chain(3, sigma, order, mode))
    out = np.empty_like(x)
    filters.chains(x, chains, out, combine="sum")
    _same_bits(out, ndi.gaussian_laplace(x, sigma, mode=mode))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("sigma", [(1.0, 1.5, 2.0), (0.0, 1.0, 0.7)])
def test_gaussian_gradient_magnitude_equals_scipy(dtype, mode, sigma):
    x = _data((5, 19, 14), dtype, seed=13)
    chains = []
    for d in range(3):
        order = [1 if a == d else 0 for a in range(3)]
        chains.append(_gaussian_chain(3, sigma, order, mode))
    for threads in (1, 4):
        out = np.empty_like(x)
        filters.chains(x, chains, out, combine="hypot", threads=threads)
        _same_bits(out, ndi.gaussian_gradient_magnitude(x, sigma, mode=mode))


def test_a_gradient_magnitude_of_integers_is_refused():
    x = np.zeros((4, 5), np.uint16)
    chains = [[(0, "correlate", [-1.0, 0.0, 1.0], 0, "reflect")]]
    with pytest.raises(filters.FilterError, match="floating-point"):
        filters.chains(x, chains, np.empty_like(x), combine="hypot")


def test_running_passes_on_threads_change_nothing():
    x = _floats_with_specials((30, 40, 50), "float32", seed=14)
    for kind, size in (("minimum", 5), ("maximum", 4), ("uniform", 7)):
        for axis in range(3):
            chain = [[(axis, kind, [size], 0, "mirror")]]
            _same_bits(_chains_out(x, chain, threads=4), _chains_out(x, chain))
