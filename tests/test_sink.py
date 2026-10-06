"""io.create_sink: the PUSH counterpart of io.write (e.g. for dask.array.store).

The output must be resolved exactly like io.write's - format, chunks, codecs, shards,
dimension names, overwrite rule, both backends - and `write_unit` must name the block
concurrent writers may not share (the shard when sharded, else the chunk).
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import Codecs, io
from dyna_zarr.io import OutputExistsError

BACKENDS = ["tensorstore", pytest.param("zarrista", marks=pytest.mark.zarrista)]
DATA = np.random.default_rng(0).integers(0, 999, size=(2, 30, 70, 50), dtype="uint16")


def _fill(sink):
    """Write DATA block by block in write units, as a concurrent producer would."""
    u = sink.write_unit
    for idx in np.ndindex(*[-(-s // c) for s, c in zip(DATA.shape, u)]):
        key = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, u, DATA.shape))
        sink[key] = DATA[key]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("kw,unit", [
    (dict(chunks=(1, 8, 16, 16)), (1, 8, 16, 16)),
    (dict(chunks=(1, 8, 16, 16), shard_coefficients=(2, 2, 2, 2)), (2, 16, 32, 32)),
    (dict(chunks=(1, 8, 16, 16), zarr_format=2), (1, 8, 16, 16)),
])
def test_sink_resolves_like_write_and_round_trips(tmp_path, backend, kw, unit):
    out = tmp_path / "out.zarr"
    sink = io.create_sink(str(out), DATA.shape, DATA.dtype, backend=backend,
                          compressor=Codecs("zstd", clevel=3), **kw)
    assert sink.write_unit == unit
    _fill(sink)
    z = zarr.open_array(str(out))
    np.testing.assert_array_equal(z[...], DATA)
    assert z.metadata.zarr_format == kw.get("zarr_format", 3)
    assert tuple(z.chunks) == (1, 8, 16, 16)


def test_sink_writes_dimension_names_and_refuses_bad_arguments(tmp_path):
    sink = io.create_sink(str(tmp_path / "a.zarr"), DATA.shape, DATA.dtype,
                          dimension_names=("c", "z", "y", "x"))
    _fill(sink)
    assert zarr.open_array(str(tmp_path / "a.zarr")).metadata.dimension_names == ("c", "z", "y", "x")
    with pytest.raises(ValueError, match="zarr_format=3"):
        io.create_sink(str(tmp_path / "b.zarr"), DATA.shape, DATA.dtype, zarr_format=2,
                       shard_coefficients=(1, 1, 1, 1))


def test_sink_follows_the_overwrite_rule(tmp_path):
    out = str(tmp_path / "out.zarr")
    _fill(io.create_sink(out, DATA.shape, DATA.dtype))
    with pytest.raises(OutputExistsError):
        io.create_sink(out, DATA.shape, DATA.dtype)
    io.create_sink(out, DATA.shape, DATA.dtype, overwrite=True)    # replaced, not refused


def test_dask_store_into_a_sink(tmp_path):
    da = pytest.importorskip("dask.array")
    dask = pytest.importorskip("dask")
    out = str(tmp_path / "out.zarr")
    sink = io.create_sink(out, DATA.shape, DATA.dtype, chunks=(1, 8, 16, 16),
                          shard_coefficients=(2, 2, 2, 2))
    with dask.config.set(scheduler="threads", num_workers=8):
        da.store(da.from_array(DATA, chunks=(1, 7, 9, 11)).rechunk(sink.write_unit) * 1,
                 sink, lock=False)
    np.testing.assert_array_equal(zarr.open_array(out)[...], DATA)
