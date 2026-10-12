"""Saturating casts: ``operations.saturate_cast`` and io.write's ``dtype=``.

A narrowing write used to WRAP (uint16 4000 -> uint8 160), silently corrupting the
image. Every write path now saturates: floats rounded to nearest, NaN -> 0, values
clipped to the target range. References are computed with Python ints, so they cannot
share a NumPy promotion bug with the code under test.
"""
import itertools

import numpy as np
import pytest
import zarr

from dyna_zarr import operations as ops
from dyna_zarr.io import io
from dyna_zarr.operations.pointwise import _saturate_block

INTS = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]


def _ref_int(values, dtype):
    info = np.iinfo(dtype)
    return np.array([min(max(int(v), int(info.min)), int(info.max)) for v in values],
                    dtype=dtype)


def _ref_float(values, dtype):
    info = np.iinfo(dtype)
    out = []
    for v in values:
        if np.isnan(v):
            out.append(0)
        elif np.isinf(v):
            out.append(int(info.max) if v > 0 else int(info.min))
        else:
            out.append(min(max(round(float(v)), int(info.min)), int(info.max)))  # half-even
    return np.array(out, dtype=dtype)


def _extremes(dtype):
    info = np.iinfo(dtype)
    vals = [info.min, info.max, 0, 1, info.max // 2 + 3]
    if info.min < 0:
        vals += [-1, info.min // 2 - 3]
    return np.array(vals, dtype=dtype)


@pytest.mark.parametrize("src,dst", list(itertools.product(INTS, INTS)))
def test_every_integer_pair(src, dst):
    x = _extremes(src)
    out = _saturate_block(x, dst)
    assert out.dtype == np.dtype(dst)
    np.testing.assert_array_equal(out, _ref_int(x, dst))


@pytest.mark.parametrize("src", ["float16", "float32", "float64"])
@pytest.mark.parametrize("dst", INTS)
def test_float_to_every_integer(src, dst):
    info = np.iinfo(dst)
    vals = [254.7, 254.4, 2.5, 3.5, -0.4, -3.6, np.nan, np.inf, -np.inf,
            float(info.max) * 2, float(info.min) * 2 - 1, float(info.max), float(info.min)]
    with np.errstate(over="ignore"):
        x = np.array(vals, dtype=src)
    out = _saturate_block(x, dst)
    assert out.dtype == np.dtype(dst)
    np.testing.assert_array_equal(out, _ref_float(x, dst))


def test_the_motivating_cases():
    assert _saturate_block(np.array([4000, 100], "uint16"), "uint8").tolist() == [255, 100]
    assert _saturate_block(np.array([254.7, np.nan], "float32"), "uint8").tolist() == [255, 0]
    assert _saturate_block(np.array([-3], "int16"), "uint8").tolist() == [0]


def test_non_integer_targets_are_plain_casts():
    x = np.array([1.5, -2.25, 1e10])
    np.testing.assert_array_equal(_saturate_block(x, "float32"), x.astype("float32"))
    b = np.array([True, False])
    np.testing.assert_array_equal(_saturate_block(b, "uint8"), [1, 0])
    np.testing.assert_array_equal(_saturate_block(np.array([0, 3]), "bool"), [False, True])


def test_same_dtype_is_untouched():
    x = np.array([1, 2], "uint8")
    assert _saturate_block(x, "uint8") is x


def test_lazy_op_and_astype_keeps_numpy_semantics(tmp_path):
    data = np.array([[4000, 100], [0, 65535]], dtype="uint16")
    z = zarr.create_array(str(tmp_path / "s.zarr"), shape=data.shape, chunks=(1, 2),
                          dtype=data.dtype)
    z[...] = data
    a = io.read(str(tmp_path / "s.zarr"))
    np.testing.assert_array_equal(np.asarray(ops.saturate_cast(a, "uint8")),
                                  [[255, 100], [0, 255]])
    np.testing.assert_array_equal(np.asarray(a.astype("uint8")), data.astype("uint8"))


# --- every write path --------------------------------------------------------------------

DATA = (np.arange(4 * 16 * 16, dtype="uint16").reshape(4, 16, 16) * 17)   # up to ~17k
EXPECTED = np.minimum(DATA, 255).astype("uint8")


@pytest.fixture
def src(tmp_path):
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=DATA.shape, chunks=(2, 8, 8),
                          dtype=DATA.dtype)
    z[...] = DATA
    return io.read(str(tmp_path / "src.zarr"))


@pytest.mark.parametrize("engine", ["fused", "pipeline"])
def test_write_dtype_saturates(tmp_path, src, engine):
    out = tmp_path / "o.zarr"
    io.write(src, str(out), dtype="uint8", engine=engine)
    np.testing.assert_array_equal(zarr.open_array(str(out))[...], EXPECTED)


def test_staged_write_saturates(tmp_path, src):
    out = tmp_path / "flat.zarr"
    io.write(ops.flatten(src), str(out), dtype="uint8")
    np.testing.assert_array_equal(zarr.open_array(str(out))[...], EXPECTED.reshape(-1))


def test_float_chain_written_as_integer_rounds(tmp_path, src):
    out = tmp_path / "o.zarr"
    io.write(src.astype("float32") / 100.0, str(out), dtype="uint8")
    ref = _ref_float((DATA / np.float32(100.0)).astype("float32").ravel(), "uint8")
    np.testing.assert_array_equal(zarr.open_array(str(out))[...].ravel(), ref)


def test_dask_write_saturates(tmp_path):
    da = pytest.importorskip("dask.array")
    out = tmp_path / "o.zarr"
    io.write(da.from_array(DATA, chunks=(2, 8, 8)), str(out), dtype="uint8")
    np.testing.assert_array_equal(zarr.open_array(str(out))[...], EXPECTED)


def test_sink_saturates(tmp_path):
    sink = io.create_sink(str(tmp_path / "o.zarr"), DATA.shape, "uint8", chunks=(2, 8, 8))
    sink[...] = DATA
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "o.zarr"))[...], EXPECTED)
