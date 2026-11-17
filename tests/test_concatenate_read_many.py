"""Concatenation reads its parts into one array (no second copy), and in one call
when every part is a plain source whose type offers ``read_many``."""
import numpy as np
import pytest

from dyna_zarr import from_array, operations as ops


class Source:
    """An array-like source (shape, dtype, __getitem__) counting its reads."""

    def __init__(self, data):
        self._data = data
        self.shape, self.dtype = data.shape, data.dtype
        self.reads = 0

    def __getitem__(self, key):
        self.reads += 1
        return self._data[key]


class ManySource(Source):
    """A source whose type reads many sources' keys in one call."""
    calls = []

    @staticmethod
    def read_many(requests):
        ManySource.calls.append(len(requests))
        for source, key, out in requests:
            out[...] = source._data[key]


class OtherManySource(Source):
    @staticmethod
    def read_many(requests):
        raise AssertionError("a read_many of another type must not be used")


def _parts(kind, n=5, shape=(3, 40, 50), dtype="u2"):
    rng = np.random.default_rng(1)
    datas = [rng.integers(0, 60000, size=shape).astype(dtype) for _ in range(n)]
    return datas, [kind(d) for d in datas]


KEYS = [
    np.s_[...],
    np.s_[2:11, 5:30, 7:41],                 # across parts, a rectangle
    np.s_[4, :, 3:9],                        # inside one part, an int
    np.s_[1:14, 0, :],                       # an int on another axis
    np.s_[0:15:2, 1:39:3, ::5],              # steps (the concat axis too)
    np.s_[5:5, :, :],                        # empty
]


@pytest.mark.parametrize("key", KEYS, ids=range(len(KEYS)))
@pytest.mark.parametrize("kind", [Source, ManySource])
def test_concatenation_reads_as_numpy(kind, key):
    datas, sources = _parts(kind)
    joined = ops.concatenate([from_array(s) for s in sources], axis=0)
    want = np.concatenate(datas, axis=0)[key]
    got = np.asarray(joined[key])
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got, want)


def test_one_read_many_call_for_all_parts():
    ManySource.calls.clear()
    datas, sources = _parts(ManySource)
    joined = ops.concatenate([from_array(s) for s in sources], axis=0)
    got = np.asarray(joined[1:14, 2:30, 4:44])
    np.testing.assert_array_equal(got, np.concatenate(datas)[1:14, 2:30, 4:44])
    assert ManySource.calls == [5]           # one call, five requests
    assert all(s.reads == 0 for s in sources)


@pytest.mark.parametrize("case", ["mixed types", "other read_many", "lock", "steps"])
def test_parts_read_one_by_one_when_read_many_does_not_apply(case):
    ManySource.calls.clear()
    datas, sources = _parts(ManySource)
    if case == "mixed types":
        sources[2] = Source(datas[2])
    elif case == "other read_many":
        sources[2] = OtherManySource(datas[2])
    arrays = [from_array(s, lock=(case == "lock")) for s in sources]
    joined = ops.concatenate(arrays, axis=0)
    key = np.s_[0:15, 0:40:3, :] if case == "steps" else np.s_[0:15, 5:9, :]
    np.testing.assert_array_equal(np.asarray(joined[key]), np.concatenate(datas)[key])
    assert ManySource.calls == []


def test_concatenation_along_another_axis_and_mixed_dtypes():
    a = np.arange(2 * 3 * 4, dtype="u1").reshape(2, 3, 4)
    b = np.arange(2 * 5 * 4, dtype="u2").reshape(2, 5, 4)
    joined = ops.concatenate([from_array(Source(a)), from_array(Source(b))], axis=1)
    want = np.concatenate([a, b], axis=1)
    got = np.asarray(joined[:, 1:7, ::2])
    assert got.dtype == want.dtype           # as np.concatenate promotes
    np.testing.assert_array_equal(got, want[:, 1:7, ::2])
