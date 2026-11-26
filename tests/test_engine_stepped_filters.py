"""A filter under a stepped slice computes only the elements it keeps (2026-11-25): on
a stepped axis the last pass along it computes every step-th position only, and the
passes after it run on the decimated data. The values are the full filter's, so the
engine gives exactly what the Python path gives (scipy's filter, then the stride).

Measured on MouseBrain v2 (gaussian_filter sigma 2/3, then every 2nd pixel in y and x,
busy machine): the engine's region reads on one thread 5.4 -> 3.8 s; the write 1.3x
faster than the Python path at 1 and 4 threads.
"""
import numpy as np
import pytest
import zarr

from dyna_zarr import _engine, _lowering, io
from dyna_zarr.operations import neighborhood as nb

pytestmark = pytest.mark.engine


@pytest.fixture(autouse=True)
def engine_on(monkeypatch):
    if not _engine.available():
        pytest.skip("dyna-zarr-core is not installed")
    monkeypatch.setattr(_engine, "_choice", True)


@pytest.fixture(scope="module", params=["uint16", "float32", "int16"])
def source(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("stepped") / "s.zarr"
    rng = np.random.default_rng(4)
    data = rng.integers(0, 3000, (5, 61, 83)).astype(request.param)
    zarr.create_array(str(path), shape=data.shape, chunks=(2, 16, 32),
                      dtype=request.param)[:] = data
    return str(path)


FILTERS = {
    "gaussian": lambda a: nb.gaussian_filter(a, (0.0, 2 / 3, 2 / 3), boundary="reflect"),
    "gaussian3d": lambda a: nb.gaussian_filter(a, (1.0, 1.5, 0.8), boundary="nearest"),
    "uniform": lambda a: nb.uniform_filter(a, (1, 3, 5), boundary="mirror"),
    "maximum": lambda a: nb.maximum_filter(a, (1, 3, 3)),
    "laplace": lambda a: nb.laplace(a),
    "gaussian_laplace": lambda a: nb.gaussian_laplace(a, 1.0),
}

STEPS = [
    (slice(None), slice(None, None, 2), slice(None, None, 2)),
    (slice(1, None, 2), slice(3, 50, 3), slice(1, None, 2)),
    (slice(None), slice(5, 60, 4), slice(None)),            # one axis stepped
]


@pytest.mark.parametrize("name", sorted(FILTERS))
@pytest.mark.parametrize("key", range(len(STEPS)))
def test_a_stepped_filter_is_the_filter_then_the_stride(source, name, key):
    v = FILTERS[name](io.read(source))
    k = STEPS[key]
    if _lowering.lowered(v) is None:
        pytest.skip(f"{name} is not lowered for this dtype")
    got = np.asarray(v[k].compute())
    _engine.set_engine(False)
    try:
        want = np.asarray(v.compute())[k]
    finally:
        _engine.set_engine(True)
    np.testing.assert_array_equal(got, want)


def test_regions_of_a_stepped_filter_are_its_parts(source):
    """Any region of the stepped view is the same part of the whole stepped read."""
    v = FILTERS["gaussian"](io.read(source))[:, ::2, ::2]
    whole = np.asarray(v.compute())
    for sl in [(slice(0, 2), slice(3, 20), slice(0, 9)),
               (slice(4, 5), slice(25, 31), slice(30, 42))]:
        np.testing.assert_array_equal(np.asarray(v[sl].compute()), whole[sl])


def test_the_lowered_view_computes_only_the_kept_elements(source):
    """The engine's view under the stride costs less than the full filter's read."""
    a = io.read(source)
    g = _lowering.lowered(FILTERS["gaussian"](a))
    s = _lowering.lowered(FILTERS["gaussian"](a)[:, ::2, ::2])
    full = g.cost([0, 0, 0], list(g.shape), 1)
    kept = s.cost([0, 0, 0], list(s.shape), 1)
    assert kept < full, (kept, full)
