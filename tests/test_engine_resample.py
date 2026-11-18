"""Affine resampling in dyna's engine (dyna-zarr-core, Workplan WP15 / DR4) against
scipy.ndimage.affine_transform: the same matrix convention (output index -> input
index), interpolation orders 0 / 1 and edge modes "constant" / "grid-constant"."""
import numpy as np
import pytest
import zarr

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")
nd = pytest.importorskip("scipy.ndimage")

ORDER = {"nearest": 0, "linear": 1}


def _data(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.dtype(dtype).kind == "f":
        return rng.normal(scale=100, size=shape).astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, size=shape, dtype=dtype, endpoint=True)


def _store(tmp_path, data, chunks, name="a"):
    z = zarr.create_array(str(tmp_path / name), shape=data.shape, chunks=chunks, dtype=data.dtype)
    z[...] = data
    return core.Array.open(str(tmp_path / name))


def _affine(rng, n, shape):
    """Rotation, anisotropic scale, shear and a shift (pixel units), as a full
    homogeneous matrix mapping output to input indices, roughly centred on the image."""
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    a = q @ np.diag(rng.uniform(0.6, 1.6, n)) @ (np.eye(n) + np.triu(rng.uniform(-0.3, 0.3, (n, n)), 1))
    centre = (np.asarray(shape) - 1) / 2
    offset = centre - a @ centre + rng.uniform(-3, 3, n)
    m = np.eye(n + 1)
    m[:n, :n], m[:n, n] = a, offset
    return m


def _scipy(data, m, shape, interpolation, mode, fill):
    return nd.affine_transform(data, m, output_shape=tuple(shape), order=ORDER[interpolation],
                               mode=mode, cval=fill, prefilter=False, output=data.dtype)


def _compare(got, want, interpolation):
    """Identical to scipy: nearest and integer outputs bit for bit, floats to rounding
    (measured 2026-11-18: 0 of 3.4 million pixels differ; floats within 1e-13)."""
    if interpolation == "nearest" or np.dtype(want.dtype).kind != "f":
        np.testing.assert_array_equal(got, want)
    else:
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * float(np.abs(want).max() or 1))


@pytest.mark.parametrize("mode", ["constant", "grid-constant"])
@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int16", "float32", "float64"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_as_scipy(tmp_path, ndim, dtype, interpolation, mode):
    rng = np.random.default_rng(ndim * 10 + len(dtype))
    shape = [31, 45, 38][-ndim:] if ndim == 3 else [57, 63]
    data = _data(shape, dtype)
    a = _store(tmp_path, data, [8] * ndim)
    for k in range(3):
        m = _affine(rng, ndim, shape)
        out_shape = [s + int(rng.integers(-5, 6)) for s in shape]
        fill = 7
        got = a.resample(m.tolist(), out_shape, interpolation, mode, fill).read([0] * ndim, out_shape, threads=4)
        _compare(got, _scipy(data, m, out_shape, interpolation, mode, fill), interpolation)


def test_regions_threads_and_splitting_change_nothing(tmp_path):
    rng = np.random.default_rng(5)
    data = _data((20, 60, 70), "float32")
    a = _store(tmp_path, data, (5, 16, 16))
    m = _affine(rng, 3, data.shape)
    v = a.resample(m.tolist(), [22, 64, 66], "linear")
    whole = v.read([0, 0, 0], [22, 64, 66], threads=1)
    tiny = a.resample(m.tolist(), [22, 64, 66], "linear", max_read_mb=0.01)
    np.testing.assert_array_equal(tiny.read([0, 0, 0], [22, 64, 66], threads=8), whole)
    np.testing.assert_array_equal(v.read([3, 10, 7], [15, 50, 60], threads=4), whole[3:15, 10:50, 7:60])
    out = np.zeros((30, 70, 80), "float32")
    v.read([2, 4, 6], [12, 40, 50], out=out, at=[5, 6, 7], threads=2)
    np.testing.assert_array_equal(out[5:15, 6:42, 7:51], whole[2:12, 4:40, 6:50])


def test_a_spatial_transformation_on_a_5d_array(tmp_path):
    rng = np.random.default_rng(9)
    data = _data((2, 3, 12, 30, 34), "uint16")
    a = _store(tmp_path, data, (1, 1, 6, 16, 16))
    zyx = _affine(rng, 3, data.shape[2:])
    full = core.transforms.embed(zyx[:3].tolist(), ["z", "y", "x"], ["t", "c", "z", "y", "x"])
    out_shape = [2, 3, 12, 28, 36]
    got = a.resample(full, out_shape, "linear").read([0] * 5, out_shape, threads=4)
    for t in range(2):
        for c in range(3):
            _compare(got[t, c], _scipy(data[t, c], zyx, out_shape[2:], "linear", "constant", 0), "linear")


def test_a_region_wholly_outside_is_fill(tmp_path):
    data = _data((10, 10), "float32")
    a = _store(tmp_path, data, (5, 5))
    far = [[1, 0, 1000], [0, 1, 0]]
    got = a.resample(far, [4, 4], "linear", fill=-3).read([0, 0], [4, 4])
    assert (got == -3).all()


def test_nearest_on_bool_and_complex_linear_refused(tmp_path):
    eye = np.eye(6, dtype=bool)
    b = _store(tmp_path, eye, (3, 3), "b")
    turn = [[0, 1, 0], [-1, 0, 5]]                       # a quarter turn
    want = _scipy(eye, np.array(turn + [[0, 0, 1]], float), [6, 6], "nearest", "constant", 0)
    np.testing.assert_array_equal(b.resample(turn, [6, 6], "nearest").read([0, 0], [6, 6]), want)
    with pytest.raises(core.ReadError, match="nearest"):
        b.resample(turn, [6, 6], "linear")
    with pytest.raises(core.ReadError, match="mode"):
        b.resample(turn, [6, 6], "nearest", mode="reflect")
    with pytest.raises(core.ReadError):
        b.resample([[1, 0, 0]], [6, 6], "nearest")
