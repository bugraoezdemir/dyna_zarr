"""Array metadata both backends must agree on (found comparing with eubi-bridge's writer).

- TensorStore left a zarr v2 fill_value null (an unwritten chunk is then undefined to
  readers) while zarrista wrote 0: both now write a typed zero (false / 0.0 / 0).
- A shard larger than the array (a small pyramid level under 3x-chunk shards) only held
  empty chunk slots: shards are clipped to the array, rounded up to whole chunks.
"""
import contextlib
import io as _io
import json

import numpy as np
import pytest

import dyna_zarr
from dyna_zarr.backends.zarrista_backend import ZARRISTA_AVAILABLE
from dyna_zarr.io import io

BACKENDS = ["tensorstore"] + (["zarrista"] if ZARRISTA_AVAILABLE else [])


def _write(tmp_path, data, **kw):
    out = tmp_path / "o.zarr"
    with contextlib.redirect_stdout(_io.StringIO()):
        io.write(dyna_zarr.from_array(data), str(out), overwrite=True, **kw)
    return out


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("zf", [2, 3])
@pytest.mark.parametrize("dtype,want", [("uint16", 0), ("float32", 0.0), ("bool", False)])
def test_fill_value_is_a_typed_zero(tmp_path, backend, zf, dtype, want):
    out = _write(tmp_path, np.zeros((8, 8), dtype), zarr_format=zf, backend=backend)
    meta = json.loads((out / (".zarray" if zf == 2 else "zarr.json")).read_text())
    assert meta["fill_value"] == want and type(meta["fill_value"]) is type(want)


@pytest.mark.parametrize("backend", BACKENDS)
def test_shards_are_clipped_to_the_array(tmp_path, backend):
    out = _write(tmp_path, np.zeros((41, 90), "uint8"), zarr_format=3, backend=backend,
                 chunks=(41, 80), shard_coefficients=(3, 3))
    meta = json.loads((out / "zarr.json").read_text())
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [41, 160]


@pytest.mark.parametrize("backend", BACKENDS)
def test_shards_within_the_array_are_untouched(tmp_path, backend):
    out = _write(tmp_path, np.zeros((64, 64), "uint8"), zarr_format=3, backend=backend,
                 chunks=(8, 8), shard_coefficients=(2, 4))
    meta = json.loads((out / "zarr.json").read_text())
    assert meta["chunk_grid"]["configuration"]["chunk_shape"] == [16, 32]
