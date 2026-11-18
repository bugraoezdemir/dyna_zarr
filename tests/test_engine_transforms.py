"""The engine's coordinate transformations (dyna-zarr-core, Workplan WP15 / DR2)
against ome_zarr_pyramid's spatial model and the NGFF 0.6 text.

tests/transform_fixtures.json holds what ozp's own code gives (core/scripts/
make_transform_fixtures.py writes it; ozp itself cannot be installed here, it depends
on dyna). rotation and mapAxis, which ozp does not model yet, are checked against the
spec. Input regions are checked against mapping every output pixel one by one."""
import json
import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.engine
core = pytest.importorskip("dyna_zarr_core")
tf = core.transforms

FIXTURES = json.load(open(os.path.join(os.path.dirname(__file__), "transform_fixtures.json"),
                          encoding="utf-8"))
CASES = FIXTURES["cases"]
EPS = 1e-12


def _id(case):
    return f"{case['json']['type']}-{case['ndim']}d-{CASES.index(case)}"


def _same(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0, atol=EPS)


def _ozp_end(end):
    """ozp's default reading of an end with a path but no name (a level's
    ``{"path": "0"}``) writes the name back as the string 'None' (its readers pass their
    own end function, so files are not affected); the engine keeps the end as read."""
    if isinstance(end, dict) and end.get("name") == "None":
        return {k: v for k, v in end.items() if k != "name"}
    return end


def _ozp_json(obj):
    out = {k: _ozp_end(v) if k in ("input", "output") else v for k, v in obj.items()}
    if "transformations" in out:
        out["transformations"] = [_ozp_json(s) for s in out["transformations"]]
    return out


def _same_type(got, want, matrix):
    """The simplest type of a matrix is decided with exact comparisons, in ozp and in the
    engine. A product that lands within rounding of a simpler form (a rotation times its
    inverse) can come out simpler in one and as an affine in the other; the matrices
    still agree to EPS (checked by the caller)."""
    if got == want:
        return True
    m = np.asarray(matrix, float)
    n = m.shape[0] - 1
    linear = m[:n, :n]
    return {got, want} <= {"affine", "identity", "translation", "scale", "sequence"} and \
        np.allclose(linear, np.diag(np.diag(linear)), rtol=0, atol=EPS)


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_matrix_as_ozp_or_the_spec(case):
    text = json.dumps(case["json"])
    if "error" in case:
        with pytest.raises(tf.TransformError):
            tf.matrix(text, case["ndim"])
        return
    assert _same(tf.matrix(text, case["ndim"]), case["matrix"])


@pytest.mark.parametrize("case", [c for c in CASES if "round_trip" in c], ids=_id)
def test_json_round_trip_and_modelled_as_ozp(case):
    text = json.dumps(case["json"])
    assert json.loads(tf.round_trip(text)) == _ozp_json(case["round_trip"])
    assert tf.modelled(text) == case["modelled"]


@pytest.mark.parametrize("case", [c for c in CASES if "matrix" in c], ids=_id)
def test_scale_translation_as_ozp(case):
    got = tf.scale_translation(json.dumps(case["json"]), case["ndim"])
    want = case["scale_translation"]
    assert (got is None) == (want is None)
    if got is not None:
        assert _same(got[0], want[0]) and _same(got[1], want[1])


@pytest.mark.parametrize("case", [c for c in CASES if "matrix" in c], ids=_id)
def test_inverse_as_ozp(case):
    text, n = json.dumps(case["json"]), case["ndim"]
    if "inverse_error" in case:
        with pytest.raises(tf.TransformError, match="singular"):
            tf.invert(text, n)
        return
    got = json.loads(tf.invert(text, n))
    if "inverse" in case:                                   # ozp's own inverse
        want = _ozp_json(case["inverse"]["json"])
        assert _same_type(got["type"], want["type"], case["inverse"]["matrix"])
        assert got.get("input") == want.get("input") and got.get("output") == want.get("output")
        assert got.get("name") == want.get("name")
        assert _same(tf.matrix(json.dumps(got), n), case["inverse"]["matrix"])
    else:                                                   # spec cases: the matrix inverse
        assert _same(tf.matrix(json.dumps(got), n), case["inverse_matrix"])


@pytest.mark.parametrize("item", FIXTURES["compose"], ids=lambda c: str(len(c["json"])))
def test_compose_as_ozp(item):
    got = json.loads(tf.compose([json.dumps(o) for o in item["json"]], item["ndim"]))
    want = item["expected"]
    assert _same(tf.matrix(json.dumps(got), item["ndim"]), want["matrix"])
    assert _same_type(got["type"], want["json"]["type"], want["matrix"])


def test_from_matrix_gives_the_simplest_type():
    kinds = {
        "identity": [[1, 0, 0], [0, 1, 0]],
        "translation": [[1, 0, 2], [0, 1, 0]],
        "scale": [[2, 0, 0], [0, 1, 0]],
        "sequence": [[2, 0, 1], [0, 1, 0]],
        "affine": [[0, 1, 0], [1, 0, 0]],
    }
    for kind, rows in kinds.items():
        assert json.loads(tf.from_matrix(rows))["type"] == kind
    # the full homogeneous form is accepted too
    assert json.loads(tf.from_matrix([[2, 0, 0], [0, 1, 0], [0, 0, 1]]))["type"] == "scale"


def test_embed_acts_on_the_named_axes():
    yx = [[2.0, 0.0, 1.0], [0.0, 3.0, 4.0]]
    full = np.asarray(tf.embed(yx, ["y", "x"], ["t", "c", "z", "y", "x"]))
    assert full.shape == (6, 6)
    assert np.allclose(full @ [1, 2, 3, 1, 1, 1], [1, 2, 3, 3, 7, 1])
    with pytest.raises(tf.TransformError, match="not one of"):
        tf.embed(yx, ["y", "q"], ["y", "x"])


def test_bounding_box_of_a_rotated_box():
    lo, hi = tf.bounding_box([[0, -1, 0], [1, 0, 0]], [0, 0], [2, 4])
    assert (lo, hi) == ([-4.0, 0.0], [0.0, 2.0])


@pytest.mark.parametrize("unit, into, factor", [
    ("micrometer", "nanometer", 1e-6 / 1e-9), ("minute", "second", 60.0),
    (None, None, 1.0), ("micrometer", "second", None), ("micrometer", None, None)])
def test_unit_factor_as_ozp(unit, into, factor):
    assert tf.unit_factor(unit, into) == factor


# ------------------------------------------------------------- input regions


def _snap(x):
    r = round(x)
    return float(r) if abs(x - r) <= 1e-9 else x


def _support(x, interpolation):
    if interpolation == "nearest":
        i = math.floor(_snap(x + 0.5))
        return i, i
    s = _snap(x)
    return math.floor(s), math.ceil(s)


def _brute(rows, start, stop, interpolation, shape):
    """Map every output pixel centre and collect what the interpolation reads."""
    m = np.asarray(rows, float)
    grids = np.meshgrid(*[np.arange(a, b) for a, b in zip(start, stop)], indexing="ij")
    pts = np.stack([g.ravel() for g in grids] + [np.ones(grids[0].size)])
    mapped = (m @ pts)[: len(start)]
    lo, hi = [], []
    for axis, values in enumerate(mapped):
        sup = [_support(v, interpolation) for v in values]
        first, last = min(s[0] for s in sup), max(s[1] for s in sup)
        if last < 0 or first > shape[axis] - 1:
            return None
        lo.append(max(first, 0))
        hi.append(min(last, shape[axis] - 1) + 1)
    return lo, hi


def _random_affine(rng, n):
    """A rotation, scale, shear and shift in n dimensions (pixel units)."""
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    a = q @ np.diag(rng.uniform(0.3, 3.0, n)) @ (np.eye(n) + np.triu(rng.uniform(-0.4, 0.4, (n, n)), 1))
    return np.hstack([a, rng.uniform(-20, 60, (n, 1))]).tolist()


@pytest.mark.parametrize("interpolation", ["nearest", "linear"])
@pytest.mark.parametrize("ndim", [2, 3])
def test_input_region_is_what_mapping_every_pixel_reads(interpolation, ndim):
    rng = np.random.default_rng(7 + ndim)
    shape = [40, 50, 30][:ndim]
    for _ in range(60):
        rows = _random_affine(rng, ndim)
        start = rng.integers(0, 30, ndim).tolist()
        stop = [s + int(rng.integers(1, 12)) for s in start]
        got = tf.input_region(rows, start, stop, interpolation, shape)
        want = _brute(rows, start, stop, interpolation, shape)
        assert (None if got is None else (list(got[0]), list(got[1]))) == want, (rows, start, stop)


def test_input_region_on_pixel_centres_and_half_way():
    ident = [[1, 0]]
    assert tf.input_region(ident, [2], [5], "linear", [10]) == ([2], [5])
    half = [[1, 0.5]]
    assert tf.input_region(half, [2], [5], "linear", [10]) == ([2], [6])
    assert tf.input_region(half, [2], [5], "nearest", [10]) == ([3], [6])
    assert tf.input_region([[1, 1000]], [0], [4], "linear", [10]) is None
    with pytest.raises(tf.TransformError, match="interpolation"):
        tf.input_region(ident, [0], [1], "cubic", [10])
