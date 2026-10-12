"""_read_cost.read_unit: what reading part of an array costs, through an op chain.

The sources carry micro-reader's ``read_unit`` (None: exact rectangles). An eubi-bridge
aggregative input is exactly this chain: one source per file, concatenated along z,
squeezed.
"""
import numpy as np
import pytest
import zarr

import dyna_zarr
from dyna_zarr import operations as ops
from dyna_zarr._read_cost import read_unit


class Source:
    """A micro-reader-like source: any rectangle can be read; ``read_unit`` says what
    the backend decodes for it."""

    def __init__(self, shape, unit, dtype="uint8"):
        self.shape, self.dtype, self.read_unit = tuple(shape), np.dtype(dtype), unit
        self.ndim = len(shape)
        self.chunks = (1,) * (len(shape) - 2) + tuple(shape[-2:])   # advertised: a plane

    def __getitem__(self, key):
        return np.zeros(self.shape, self.dtype)[key]


def _plane(unit):
    return dyna_zarr.from_array(Source((1, 1, 1, 40, 60), unit))


@pytest.mark.parametrize("unit, expect", [
    (None, (1, 1, 1, 1, 1)),                    # uncompressed: exact rectangles
    ((1, 1, 1, 8, 60), (1, 1, 1, 8, 60)),       # strips of 8 rows
    ((1, 1, 1, 16, 16), (1, 1, 1, 16, 16)),     # tiles
    ((1, 1, 1, 40, 60), (1, 1, 1, 40, 60)),     # one strip per plane
])
def test_a_source(unit, expect):
    assert read_unit(_plane(unit)) == expect


def test_the_aggregative_chain():
    stack = ops.concatenate([_plane((1, 1, 1, 8, 60)) for _ in range(5)], axis=2)
    assert read_unit(stack) == (1, 1, 1, 8, 60)
    squeezed = ops.squeeze(stack)                # (z, y, x)
    assert squeezed.shape == (5, 40, 60)
    assert read_unit(squeezed) == (1, 8, 60)
    assert read_unit(squeezed.astype("float32")) == (1, 8, 60)
    assert read_unit(squeezed[1:4, 5:30, ::2]) == (1, 8, 60)
    assert read_unit(squeezed[2]) == (8, 60)
    assert read_unit(squeezed[:, None]) == (1, 1, 8, 60)
    assert read_unit(ops.transpose(squeezed, (2, 1, 0))) == (60, 8, 1)


def test_mixed_inputs_take_the_coarsest_unit():
    stack = ops.concatenate([_plane(None), _plane((1, 1, 1, 8, 60))], axis=2)
    assert read_unit(stack) == (1, 1, 1, 8, 60)
    assert read_unit(ops.stack([ops.squeeze(_plane(None)), ops.squeeze(_plane(None))],
                               axis=0)) == (1, 1, 1)


def test_zarr_sources_read_whole_chunks(tmp_path):
    z = zarr.create_array(str(tmp_path / "a.zarr"), shape=(10, 40, 60), chunks=(5, 8, 16),
                          dtype="uint8")
    assert read_unit(dyna_zarr.io.read(str(tmp_path / "a.zarr"))) == (5, 8, 16)


def test_unknown_where_data_is_rearranged():
    a = ops.squeeze(_plane(None))
    assert read_unit(ops.reshape(a, (2400,))) is None
    assert read_unit(ops.pad(a, 2)) is None


def test_a_source_without_read_unit_is_unknown():
    class Plain:
        shape, dtype, ndim = (4, 5), np.dtype("uint8"), 2

        def __getitem__(self, key):
            return np.zeros(self.shape, self.dtype)[key]
    assert read_unit(dyna_zarr.from_array(Plain())) is None
