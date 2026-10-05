"""DynamicArray.dtype is always a numpy.dtype, whatever the source.

The default reader is TensorStore, whose arrays report a ``tensorstore.dtype``: it
prints as ``dtype("float32")`` and compares equal to ``np.float32``, but NumPy cannot
interpret it. It used to leak out as ``arr.dtype``, so the scans, the ``*_like`` ops
and ordinary user code such as ``np.zeros(shape, arr.dtype)`` raised
"Cannot interpret 'dtype(\"float32\")' as a data type" on a plain ``io.read`` - while
the suite, built on zarr sources, never saw it.
"""
import numpy as np
import pytest
import tensorstore as ts
import zarr

from dyna_zarr import DynamicArray, io, operations as ops

DTYPES = ["float32", "float64", "uint8", "uint16", "int32", "bool"]


def _write(path, dtype, zarr_format=3):
    data = (np.random.default_rng(0).random((6, 8, 10)) * 50).astype(dtype)
    z = zarr.create_array(str(path), shape=data.shape, chunks=(3, 4, 5), dtype=dtype,
                          zarr_format=zarr_format)
    z[...] = data
    return data


def _ts_open(path, zarr_format=3):
    driver = "zarr3" if zarr_format == 3 else "zarr"
    return ts.open({"driver": driver, "kvstore": {"driver": "file", "path": str(path)}},
                   read=True).result()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("zarr_format", [2, 3])
def test_default_reader_reports_a_numpy_dtype(tmp_path, dtype, zarr_format):
    _write(tmp_path / "a.zarr", dtype, zarr_format)
    a = io.read(str(tmp_path / "a.zarr"))
    assert a.is_tensorstore                             # the path that used to leak
    assert type(a.dtype) is type(np.dtype(dtype)) and a.dtype == np.dtype(dtype)
    assert a.dtype.itemsize == np.dtype(dtype).itemsize
    np.zeros((2, 2), a.dtype)                           # ordinary user code
    np.dtype(a.dtype)


def test_wrapping_a_tensorstore_array_directly(tmp_path):
    _write(tmp_path / "a.zarr", "uint16")
    handle = _ts_open(tmp_path / "a.zarr")
    assert not isinstance(handle.dtype, np.dtype)       # the raw source does leak one
    a = DynamicArray(handle)
    assert isinstance(a.dtype, np.dtype) and a.dtype == np.uint16
    assert isinstance(DynamicArray(a).dtype, np.dtype)  # copy constructor


def test_derived_arrays_keep_a_numpy_dtype(tmp_path):
    _write(tmp_path / "a.zarr", "float32")
    a = io.read(str(tmp_path / "a.zarr"))
    for derived in (a[1:3], a + 1, ops.gaussian_filter(a, 1), ops.transpose(a, (2, 1, 0)),
                    ops.max(a, axis=0), a.astype("uint8"), ops.stack([a, a])):
        assert isinstance(derived.dtype, np.dtype), derived


@pytest.mark.parametrize("name", ["cumsum", "cumprod", "cummax", "cummin"])
def test_scans_on_a_tensorstore_read_array(tmp_path, name):
    data = _write(tmp_path / "a.zarr", "float32")
    a = io.read(str(tmp_path / "a.zarr"))
    ref = {"cumsum": np.cumsum, "cumprod": np.cumprod,
           "cummax": np.maximum.accumulate, "cummin": np.minimum.accumulate}[name]
    for axis in (0, 2):
        r = getattr(ops, name)(a, axis=axis)
        np.testing.assert_allclose(r.compute(), ref(data, axis=axis), rtol=1e-5)
    io.write(getattr(ops, name)(a, axis=1), str(tmp_path / "scan.zarr"))
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "scan.zarr"))[...],
                               ref(data, axis=1), rtol=1e-5)


@pytest.mark.parametrize("name, fill", [("zeros_like", 0), ("ones_like", 1),
                                        ("full_like", 7), ("empty_like", None)])
def test_like_ops_on_a_tensorstore_read_array(tmp_path, name, fill):
    _write(tmp_path / "a.zarr", "int32")
    a = io.read(str(tmp_path / "a.zarr"))
    r = ops.full_like(a, fill) if name == "full_like" else getattr(ops, name)(a)
    assert r.shape == a.shape and r.dtype == np.int32 and isinstance(r.dtype, np.dtype)
    if fill is not None:
        np.testing.assert_array_equal(r.compute(), np.full(a.shape, fill, np.int32))
