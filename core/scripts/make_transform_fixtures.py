"""Write tests/transform_fixtures.json: what ome_zarr_pyramid's spatial model gives for a
set of NGFF transformations, so dyna's tests can check the engine's transformations
crate (Workplan DR2) against it without installing ozp (ozp depends on dyna).

    python core/scripts/make_transform_fixtures.py      (from the repository root, in a
                                                        venv with ome_zarr_pyramid)

For the five types ozp models (identity, translation, scale, affine, sequence), the
expected values are ozp's own: spatial.matrix, invert, compose, scale_translation and
the JSON round trip. For rotation and mapAxis, which ozp does not model yet, they are
computed here from the NGFF 0.6 text (rotation: the matrix as given; mapAxis: output
axis i is input axis mapAxis[i]); those cases say ``"source": "spec"``. Run it again
when ozp's spatial model changes.
"""
from __future__ import annotations

import json
import os

import numpy as np
from ome_zarr_pyramid.core import spatial
from ome_zarr_pyramid import __version__ as ozp_version

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "tests", "transform_fixtures.json")

c, s = np.cos(0.3), np.sin(0.3)
ROT3 = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]          # about the last axis

#: (transformation, ndim); every one is also inverted and read as scale + translation
CASES = [
    ({"type": "identity"}, 3),
    ({"type": "translation", "translation": [1.5, -2.0, 30.0]}, 3),
    ({"type": "scale", "scale": [2.0, 0.65, 0.65]}, 3),
    ({"type": "scale", "scale": [0.0, 1.0]}, 2),                                 # singular
    ({"type": "affine", "affine": [[0.8, -0.6, 5.0], [0.6, 0.8, -3.0]]}, 2),
    ({"type": "affine", "affine": [0.8, -0.6, 5.0, 0.6, 0.8, -3.0]}, 2),         # flat
    ({"type": "affine", "affine": [[2.0, 0.0, 1.0], [0.0, 3.0, 4.0], [0.0, 0.0, 1.0]]}, 2),
    ({"type": "affine", "affine": [[1.0, 0.3, 0.0, -7.2], [0.0, 1.0, 0.1, 11.6],
                                   [0.2, 0.0, 1.0, 0.5]]}, 3),
    ({"type": "affine", "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]}, 3),
    ({"type": "affine", "path": "coefficients"}, 2),                             # stored
    ({"type": "sequence", "input": {"path": "0"}, "output": {"name": "physical"},
      "transformations": [{"type": "scale", "scale": [1.0, 0.5, 0.5]},
                          {"type": "translation", "translation": [0.0, 10.0, -4.0]}]}, 3),
    ({"type": "sequence", "name": "reg", "input": {"name": "a"}, "output": {"name": "b"},
      "transformations": [{"type": "translation", "translation": [3.0, 4.0]},
                          {"type": "affine", "affine": [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]},
                          {"type": "scale", "scale": [2.0, 2.0]}]}, 2),
    ({"type": "displacements", "path": "field", "interpolation": "linear"}, 2),  # other
    ({"type": "rotation", "rotation": [[0.0, -1.0], [1.0, 0.0]]}, 2),             # spec
    ({"type": "rotation", "rotation": ROT3}, 3),
    ({"type": "rotation", "rotation": [[1.0, 1.0], [0.0, 1.0]]}, 2),             # not one
    ({"type": "mapAxis", "mapAxis": [1, 0]}, 2),
    ({"type": "mapAxis", "mapAxis": [2, 0, 1]}, 3),
    ({"type": "mapAxis", "mapAxis": [0, 0, 1]}, 3),                              # not one
    ({"type": "sequence", "transformations": [
        {"type": "mapAxis", "mapAxis": [1, 0]},
        {"type": "scale", "scale": [2.0, 3.0]}]}, 2),
]

#: lists of transformations to compose (the first applied first), and their ndim
COMPOSE = [
    ([{"type": "scale", "scale": [2.0, 2.0]}, {"type": "scale", "scale": [0.5, 0.5]}], 2),
    ([{"type": "scale", "scale": [2.0, 3.0]}, {"type": "translation", "translation": [1.0, 1.0]}], 2),
    ([{"type": "translation", "translation": [1.0, 1.0]}, {"type": "scale", "scale": [2.0, 3.0]}], 2),
    ([{"type": "affine", "affine": [[0.8, -0.6, 5.0], [0.6, 0.8, -3.0]]},
      {"type": "affine", "affine": [[0.8, 0.6, 0.0], [-0.6, 0.8, 0.0]]}], 2),
    ([{"type": "translation", "translation": [0.0, 1.0, 2.0]},
      {"type": "scale", "scale": [4.0, 1.0, 1.0]},
      {"type": "translation", "translation": [0.0, -1.0, -2.0]}], 3),
]


def spec_matrix(obj: dict, ndim: int) -> np.ndarray:
    """rotation / mapAxis (and sequences of them with ozp's types) from the NGFF text."""
    m = np.eye(ndim + 1)
    if obj["type"] == "rotation":
        r = np.asarray(obj["rotation"], dtype=float)
        if r.shape != (ndim, ndim) or not np.allclose(r @ r.T, np.eye(ndim), atol=1e-6) \
                or abs(np.linalg.det(r) - 1) > 1e-6:
            raise ValueError("not a rotation")
        m[:ndim, :ndim] = r
        return m
    if obj["type"] == "mapAxis":
        order = obj["mapAxis"]
        if sorted(order) != list(range(ndim)):
            raise ValueError("not a permutation")
        m[:ndim, :ndim] = 0.0
        for i, src in enumerate(order):
            m[i, src] = 1.0
        return m
    if obj["type"] == "sequence":
        for step in obj["transformations"]:
            m = spec_matrix(step, ndim) @ m
        return m
    return spatial.matrix(spatial.transformation_from_json(obj), ndim)


def uses_spec(obj: dict) -> bool:
    if obj["type"] in ("rotation", "mapAxis"):
        return True
    return any(uses_spec(s) for s in obj.get("transformations") or [])


def describe(t: spatial.Transformation, ndim: int) -> dict:
    return {"json": spatial.transformation_to_json(t), "matrix": spatial.matrix(t, ndim).tolist()}


def case(obj: dict, ndim: int) -> dict:
    out = {"json": obj, "ndim": ndim}
    if uses_spec(obj):
        out["source"] = "spec"
        try:
            m = spec_matrix(obj, ndim)
        except ValueError as e:
            out["error"] = str(e)
            return out
        out["matrix"] = m.tolist()
        lin = m[:ndim, :ndim]
        out["scale_translation"] = ([np.diag(lin).tolist(), m[:ndim, ndim].tolist()]
                                    if np.array_equal(lin, np.diag(np.diag(lin))) else None)
        out["inverse_matrix"] = np.linalg.inv(m).tolist()
        return out
    out["source"] = f"ome_zarr_pyramid {ozp_version}"
    t = spatial.transformation_from_json(obj)
    out["modelled"] = t.modelled
    out["round_trip"] = spatial.transformation_to_json(t)
    try:
        m = spatial.matrix(t, ndim)
    except spatial.TransformError as e:
        out["error"] = str(e)
        return out
    out["matrix"] = m.tolist()
    st = spatial.scale_translation(t, ndim)
    out["scale_translation"] = None if st is None else [st[0].tolist(), st[1].tolist()]
    try:
        out["inverse"] = describe(spatial.invert(t, ndim), ndim)
    except spatial.TransformError as e:
        out["inverse_error"] = str(e)
    return out


def main():
    fixtures = {
        "about": "Expected values from ome_zarr_pyramid's spatial model (and the NGFF 0.6 "
                 "text for rotation / mapAxis), written by core/scripts/make_transform_fixtures.py",
        "ozp_version": ozp_version,
        "cases": [case(obj, n) for obj, n in CASES],
        "compose": [],
    }
    for objs, n in COMPOSE:
        ts = [spatial.transformation_from_json(o) for o in objs]
        fixtures["compose"].append({"json": objs, "ndim": n,
                                    "expected": describe(spatial.compose(*ts, ndim=n), n)})
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(fixtures, fh, indent=1)
        fh.write("\n")
    print(f"{OUT}: {len(fixtures['cases'])} cases, {len(fixtures['compose'])} compositions "
          f"(ozp {ozp_version})")


if __name__ == "__main__":
    main()
