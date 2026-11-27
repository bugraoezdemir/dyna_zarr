"""Downsampling in dyna's engine (dyna-zarr-core, Workplan WP15 / DR5): "stride" and
"mean" against tensorstore.downsample (which ozp's levels follow), "gaussian" against
scipy.ndimage.gaussian_filter followed by every f-th sample."""
import numpy as np
import pytest
import zarr

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")
ts = pytest.importorskip("tensorstore")
nd = pytest.importorskip("scipy.ndimage")

DTYPES = ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
          "float16", "float32", "float64", "complex64", "complex128"]


def _data(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if dtype == "bool":
        return rng.random(shape) > 0.5
    if dtype.startswith("complex"):
        return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    if dtype.startswith("float"):
        return rng.normal(scale=100, size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)


def _store(tmp_path, data, chunks, name="a"):
    z = zarr.create_array(str(tmp_path / name), shape=data.shape, chunks=chunks, dtype=data.dtype)
    z[...] = data
    return core.Array.open(str(tmp_path / name))


def _ts(data, factors, method):
    return np.asarray(ts.downsample(ts.array(data), list(factors), method=method).read().result())


def _all(v, threads=4):
    return v.read([0] * v.ndim, list(v.shape), threads=threads)


FACTORS = [(2, 2, 2), (3, 2, 1), (1, 3, 4), (5, 5, 5)]


@pytest.mark.parametrize("factors", FACTORS, ids=str)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mean_and_stride_as_tensorstore(tmp_path, dtype, factors):
    data = _data((13, 22, 31), dtype)                       # odd sizes: partial edge windows
    a = _store(tmp_path, data, (4, 8, 8))
    for method, ts_method in (("mean", "mean"), ("stride", "stride")):
        v = a.downsample(list(factors), method)
        want = _ts(data, factors, ts_method)
        assert v.shape == want.shape
        got = _all(v)
        if method == "mean" and dtype.startswith("complex"):
            # TensorStore divides a complex sum by the count as one complex number. On
            # macOS (libc++) that rounds differently from dividing each part, by up to
            # one unit in the last place when the count is not a power of two; on Linux
            # and Windows it equals dividing each part, as the engine does.
            for part in ("real", "imag"):
                np.testing.assert_array_max_ulp(getattr(got, part), getattr(want, part), 1)
        else:
            np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("factors", FACTORS, ids=str)
@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_complex_mean_is_the_mean_of_each_part(tmp_path, dtype, factors):
    """Exact on every platform: each part of a complex mean is the float mean of that
    part, which equals tensorstore.downsample's float mean bit for bit
    (test_mean_and_stride_as_tensorstore)."""
    data = _data((13, 22, 31), dtype)
    got = _all(_store(tmp_path, data, (4, 8, 8), "c").downsample(list(factors)))
    for part in ("real", "imag"):
        p = np.ascontiguousarray(getattr(data, part))
        want = _all(_store(tmp_path, p, (4, 8, 8), part).downsample(list(factors)))
        np.testing.assert_array_equal(getattr(got, part), want)


def test_integer_means_are_exact_at_64_bits(tmp_path):
    for dtype, base in (("int64", 2**62), ("uint64", 2**64 - 5)):
        data = (np.full((4, 6), base, dtype=dtype) + np.arange(24, dtype=dtype).reshape(4, 6) % 3)
        a = _store(tmp_path, data, (2, 3), dtype)
        np.testing.assert_array_equal(_all(a.downsample([2, 2])), _ts(data, (2, 2), "mean"))


def _gaussian_ref(data, factors, sigma, mode, truncate=4.0, cval=0.0):
    """scipy's filter in float64 on the whole array, every f-th sample, rounded once."""
    f = nd.gaussian_filter(data.astype(np.float64), sigma, mode=mode, truncate=truncate, cval=cval)
    f = f[tuple(slice(None, None, k) for k in factors)]
    if np.dtype(data.dtype).kind in "iu":
        info = np.iinfo(data.dtype)
        return np.clip(np.rint(f), info.min, info.max).astype(data.dtype)
    return f.astype(data.dtype)


@pytest.mark.parametrize("mode", ["reflect", "mirror", "nearest", "constant"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "float32", "float64"])
@pytest.mark.parametrize("factors", [(2, 2, 2), (1, 3, 2), (4, 4, 4)], ids=str)
def test_gaussian_as_scipy(tmp_path, dtype, factors, mode):
    data = _data((19, 26, 33), dtype, seed=3)
    a = _store(tmp_path, data, (5, 8, 8))
    sigma = [f / 3 if f > 1 else 0.0 for f in factors]
    got = _all(a.downsample(list(factors), "gaussian", mode=mode, cval=5.0))
    want = _gaussian_ref(data, factors, sigma, mode, cval=5.0)
    if np.dtype(dtype).kind == "f":
        np.testing.assert_allclose(got, want, rtol=1e-6 if dtype == "float32" else 1e-12,
                                   atol=1e-12 * float(np.abs(want).max()))
    else:
        np.testing.assert_array_equal(got, want)


def test_gaussian_with_a_given_sigma_and_a_wide_kernel(tmp_path):
    data = _data((8, 40, 50), "float64", seed=4)
    a = _store(tmp_path, data, (4, 16, 16))
    got = _all(a.downsample([1, 2, 2], "gaussian", sigma=[0.0, 3.0, 1.5], truncate=3.0))
    want = _gaussian_ref(data, (1, 2, 2), [0.0, 3.0, 1.5], "reflect", 3.0)
    # Close to scipy, not bit for bit (dyna's own Gaussian goes through the filters, which
    # are): the engine computes its weights in Rust and sums in its own order, so a
    # value much smaller than its neighbours can differ by a few units in the last place
    # (scipy 1.10: 8 of them on one value of 4000). Hence a tolerance on the data's scale.
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14 * np.abs(want).max())
    # a kernel wider than the array: reflect folds several times
    small = _store(tmp_path, data[:2, :3, :4].copy(), (2, 3, 4), "small")
    got = _all(small.downsample([1, 1, 2], "gaussian", sigma=[0.0, 4.0, 4.0]))
    np.testing.assert_allclose(got, _gaussian_ref(data[:2, :3, :4], (1, 1, 2), [0.0, 4.0, 4.0], "reflect"), rtol=1e-12)


@pytest.mark.parametrize("method", ["mean", "gaussian", "stride"])
def test_regions_threads_and_splitting_change_nothing(tmp_path, method):
    data = _data((20, 70, 90), "uint16", seed=6)
    a = _store(tmp_path, data, (5, 16, 16))
    whole = _all(a.downsample([2, 3, 2], method), threads=1)
    tiny = a.downsample([2, 3, 2], method, max_read_mb=0.01)
    np.testing.assert_array_equal(_all(tiny, threads=8), whole)
    v = a.downsample([2, 3, 2], method)
    np.testing.assert_array_equal(v.read([2, 5, 7], [9, 20, 40], threads=4), whole[2:9, 5:20, 7:40])


def test_refusals(tmp_path):
    b = _store(tmp_path, np.eye(6, dtype=bool), (3, 3))
    with pytest.raises(core.ReadError, match="stride or mean"):
        b.downsample([2, 2], "gaussian")
    with pytest.raises(core.ReadError, match="method"):
        b.downsample([2, 2], "median")
    with pytest.raises(core.ReadError, match="factors"):
        b.downsample([2], "mean")
    with pytest.raises(core.ReadError, match="factors"):
        b.downsample([0, 2], "mean")
    u = _store(tmp_path, _data((6, 6), "uint8"), (3, 3), "u")
    with pytest.raises(core.ReadError, match="mode"):
        u.downsample([2, 2], "gaussian", mode="wrap")
    with pytest.raises(core.ReadError, match="sigma"):
        u.downsample([2, 2], "gaussian", sigma=[1.0])
