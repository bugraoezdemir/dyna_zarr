"""io.write on a dask array: dask's scheduler pushes into an output dyna resolves.

The output must be exactly what io.write makes for a DynamicArray (format, chunks,
codecs, shards, dimension names, overwrite rule); dyna's region-loop options must be
refused, not ignored.
"""
import json

import numpy as np
import pytest
import zarr

da = pytest.importorskip("dask.array")

import dyna_zarr  # noqa: E402
from dyna_zarr.backends.zarrista_backend import ZARRISTA_AVAILABLE  # noqa: E402
from dyna_zarr.io import io, OutputExistsError  # noqa: E402

DATA = np.random.default_rng(0).integers(0, 4000, (6, 40, 48), dtype="uint16")


@pytest.fixture
def x():
    return da.from_array(DATA, chunks=(2, 16, 16)) + 1


def _read(path):
    return zarr.open_array(str(path))


def test_roundtrip_keeps_dask_chunks_by_default(tmp_path, x):
    out = tmp_path / "o.zarr"
    assert io.write(x, str(out)) == str(out)
    z = _read(out)
    np.testing.assert_array_equal(z[...], DATA + 1)
    assert z.chunks == (2, 16, 16)
    assert z.metadata.zarr_format == 3


def test_explicit_chunks_and_rechunked_graph(tmp_path, x):
    out = tmp_path / "o.zarr"
    io.write(x, str(out), chunks=(3, 8, 24), max_threads=3)
    z = _read(out)
    assert z.chunks == (3, 8, 24)
    np.testing.assert_array_equal(z[...], DATA + 1)


def test_chunk_size_mb_is_solved(tmp_path, x):
    out = tmp_path / "o.zarr"
    io.write(x, str(out), chunk_size_mb=0.001)
    np.testing.assert_array_equal(_read(out)[...], DATA + 1)


def test_dtype_cast(tmp_path, x):
    out = tmp_path / "o.zarr"
    io.write(x, str(out), dtype="float32")
    z = _read(out)
    assert z.dtype == np.float32
    np.testing.assert_array_equal(z[...], (DATA + 1).astype("float32"))


def test_v2_and_dimension_names(tmp_path, x):
    io.write(x, str(tmp_path / "v2.zarr"), zarr_format=2)
    assert _read(tmp_path / "v2.zarr").metadata.zarr_format == 2
    io.write(x, str(tmp_path / "v3.zarr"), dimension_names=("z", "y", "x"))
    meta = json.loads((tmp_path / "v3.zarr" / "zarr.json").read_text())
    assert meta["dimension_names"] == ["z", "y", "x"]


@pytest.mark.parametrize("backend", ["tensorstore"] + (["zarrista"] if ZARRISTA_AVAILABLE
                                                       else []))
def test_sharded(tmp_path, x, backend):
    out = tmp_path / "o.zarr"
    io.write(x, str(out), chunks=(2, 8, 16), shard_coefficients=(1, 2, 3), backend=backend)
    z = _read(out)
    assert z.shards == (2, 16, 48)
    np.testing.assert_array_equal(z[...], DATA + 1)


def test_same_output_as_the_dyna_path(tmp_path, x):
    """The pump and dyna's region loop over from_array(x) write the same bytes."""
    io.write(x, str(tmp_path / "pump.zarr"), chunks=(2, 8, 16))
    io.write(dyna_zarr.from_array(x), str(tmp_path / "pull.zarr"), chunks=(2, 8, 16))
    np.testing.assert_array_equal(_read(tmp_path / "pump.zarr")[...],
                                  _read(tmp_path / "pull.zarr")[...])
    assert _read(tmp_path / "pump.zarr").chunks == _read(tmp_path / "pull.zarr").chunks


def test_overwrite_rule(tmp_path, x):
    out = tmp_path / "o.zarr"
    io.write(x, str(out))
    with pytest.raises(OutputExistsError):
        io.write(x, str(out))
    io.write(x * 2, str(out), overwrite=True)
    np.testing.assert_array_equal(_read(out)[...], (DATA + 1) * 2)


@pytest.mark.parametrize("knob", [
    dict(region_shape=(2, 16, 16)), dict(region_size_mb=1), dict(memory_budget_mb=64),
    dict(num_readers=2), dict(device="cuda"), dict(early_tenth_timeout=10),
    dict(early_quarter_timeout=10), dict(engine="pipeline")])
def test_region_loop_options_are_refused(tmp_path, x, knob):
    with pytest.raises(ValueError, match="do(es)? not apply to a dask array"):
        io.write(x, str(tmp_path / "o.zarr"), **knob)
    assert not (tmp_path / "o.zarr").exists()


def test_unknown_chunk_sizes_are_refused(tmp_path):
    y = da.from_array(DATA, chunks=(2, 16, 16))
    y = y[y > -1]                                    # 1-d, unknown length
    with pytest.raises(ValueError, match="unknown chunk sizes"):
        io.write(y, str(tmp_path / "o.zarr"))
    assert not (tmp_path / "o.zarr").exists()


def test_invalid_output_args_fail_before_writing(tmp_path, x):
    with pytest.raises(ValueError):
        io.write(x, str(tmp_path / "o.zarr"), zarr_format=4)
    with pytest.raises(ValueError):
        io.write(x, str(tmp_path / "o.zarr"), dimension_names=("y", "x"))
    assert not (tmp_path / "o.zarr").exists()
