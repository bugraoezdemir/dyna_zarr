"""The chunk-grid invariant, enforced for EVERY op.

Every lazy op's grid passes through DynamicArray._with_transform / _from_transform, which
now require it to be None or one positive int per output axis (dynamic_array.
_checked_chunks). This sweep BUILDS every public op on gridded, grid-less, 1-D and
broadcast inputs - construction is where the check runs - so an op that reports a bad
grid fails here, including ops added later (the sweep enumerates the modules' __all__).
"""
import inspect

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, operations as ops
from dyna_zarr.operations import neighborhood, pointwise, reductions
from dyna_zarr.operations._base import Transform


def _inputs():
    f = np.random.default_rng(0).random((6, 8, 10)).astype("f4") + 0.5
    i = (f * 10).astype("i4")
    gf = DynamicArray(zarr.array(f, chunks=(3, 4, 5)))
    gi = DynamicArray(zarr.array(i, chunks=(3, 4, 5)))
    return {
        "gridded": (gf, gi),
        "gridless": (ops.reshape(ops.reshape(gf, (6, 80)), (6, 8, 10)),
                     ops.reshape(ops.reshape(gi, (6, 80)), (6, 8, 10))),
        "1d": (DynamicArray(zarr.array(f.ravel(), chunks=(64,))),
               DynamicArray(zarr.array(i.ravel(), chunks=(64,)))),
    }


INPUTS = _inputs()
KINDS = list(INPUTS)


def _valid(r, label):
    if isinstance(r, tuple):                 # histogram / gradient-style results
        for x in r:
            _valid(x, label)
        return
    if not isinstance(r, DynamicArray):
        return
    c = r.chunks
    assert c is None or (len(c) == r.ndim and all(int(v) >= 1 for v in c)), \
        f"{label}: chunks {c} for shape {r.shape}"


def _pointwise_names(n_params):
    names = []
    for name in pointwise.__all__:
        fn = getattr(pointwise, name)
        if not inspect.isfunction(fn) or name in ("map_blocks", "where", "isin", "digitize",
                                                   "clip", "round", "astype"):
            continue
        params = [p for p in inspect.signature(fn).parameters.values()
                  if p.default is inspect.Parameter.empty]
        if len(params) == n_params:
            names.append(name)
    return names


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", _pointwise_names(1))
def test_unary_pointwise(kind, name):
    f, i = INPUTS[kind]
    fn = getattr(ops, name)
    # integer-only ufuncs (invert, ...) reject the float sample while inferring dtype
    try:
        r = fn(f)
    except TypeError:
        r = fn(i)
    _valid(r, f"{name}({kind})")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", _pointwise_names(2))
def test_binary_pointwise_including_broadcast(kind, name):
    f, i = INPUTS[kind]
    fn = getattr(ops, name)
    for x in (f, i):
        try:
            for other in (x, 2, ops.max(x), ops.max(x, axis=0, keepdims=True)):
                _valid(fn(x, other), f"{name}({kind}, {type(other).__name__})")
                _valid(fn(other, x), f"{name}({type(other).__name__}, {kind})")
            break
        except TypeError:
            continue


@pytest.mark.parametrize("kind", KINDS)
def test_other_pointwise(kind):
    f, _ = INPUTS[kind]
    for label, r in [("where", ops.where(f > 1, f, 0)), ("where_bcast", ops.where(f > ops.mean(f), f, ops.min(f))),
                     ("isin", ops.isin(f, [1.0])), ("digitize", ops.digitize(f, [0.7, 1.0])),
                     ("clip", ops.clip(f, 0.6, 1.2)), ("round", ops.round(f, 1)),
                     ("astype", ops.astype(f, "u1")),
                     ("map_blocks", ops.map_blocks(lambda b: b * 2, f))]:
        _valid(r, f"{label}({kind})")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", [n for n in reductions.__all__
                                  if n not in ("ReduceTransform", "reduce", "histogram", "unique")])
def test_reductions(kind, name):
    f, _ = INPUTS[kind]
    fn = getattr(ops, name)
    axes = [None, 0, -1] if name not in ("argmin", "argmax") else [None, 0]
    for axis in axes:
        for keepdims in (False, True):
            _valid(fn(f, axis=axis, keepdims=keepdims), f"{name}({kind}, {axis}, {keepdims})")


@pytest.mark.parametrize("kind", KINDS)
def test_neighborhood(kind):
    f, _ = INPUTS[kind]
    nd = f.ndim
    for name in neighborhood.__all__:
        fn = getattr(neighborhood, name)
        if not inspect.isfunction(fn) or name == "map_overlap":
            continue
        params = inspect.signature(fn).parameters
        if "weights" in params:
            r = fn(f, np.ones((3,) * nd, "f4") / 3 ** nd)
        elif "sigma" in params:
            r = fn(f, 1)
        elif "size" in params:
            r = fn(f, size=3)
        else:
            r = fn(f)
        _valid(r, f"{name}({kind})")
    _valid(ops.map_overlap(f, lambda b: b, 1), f"map_overlap({kind})")


@pytest.mark.parametrize("kind", KINDS)
def test_structural_differences_scans_creation(kind):
    f, _ = INPUTS[kind]
    nd = f.ndim
    cases = [
        ("slice", lambda: f[1:4]), ("int_index", lambda: f[2]), ("reversed", lambda: f[::-2]),
        ("newaxis", lambda: f[None]), ("expand_dims", lambda: ops.expand_dims(f, 0)),
        ("squeeze", lambda: ops.squeeze(f[0:1])),
        ("squeeze_0d", lambda: ops.squeeze(f[tuple(slice(0, 1) for _ in range(nd))])),
        ("stack", lambda: ops.stack([f, f])), ("stack_last", lambda: ops.stack([f, f], axis=-1)),
        ("concatenate", lambda: ops.concatenate([f, f])),
        ("transpose", lambda: ops.transpose(f, tuple(reversed(range(nd))))),
        ("swap_axes", lambda: ops.swap_axes(f, 0, nd - 1)),
        ("reshape", lambda: ops.reshape(f, (int(np.prod(f.shape)),))),
        ("flatten", lambda: ops.flatten(f)), ("pad", lambda: ops.pad(f, 1)),
        ("tile", lambda: ops.tile(f, (2,) * nd)), ("roll", lambda: ops.roll(f, 1, 0)),
        ("flip", lambda: ops.flip(f, 0)),
        ("diff", lambda: ops.diff(f, 1, -1)), ("gradient", lambda: ops.gradient(f, axis=-1)),
        ("cumsum", lambda: ops.cumsum(f, axis=0)), ("cummax", lambda: ops.cummax(f, axis=-1)),
        ("zeros_like", lambda: ops.zeros_like(f)), ("random_like_shape", lambda: ops.random(f.shape, seed=0)),
    ]
    if nd >= 2:
        cases.append(("rot90", lambda: ops.rot90(f, 1, (0, 1))))
    for label, build in cases:
        _valid(build(), f"{label}({kind})")


def test_a_transform_reporting_a_bad_grid_is_refused():
    base = DynamicArray(zarr.array(np.zeros((4, 6), "f4"), chunks=(2, 3)))

    class Broken(Transform):
        def __init__(self, chunks):
            super().__init__()
            self.shape, self.dtype, self.chunks = (4, 6), np.dtype("f4"), chunks

        def read(self, key):
            return np.zeros((4, 6), "f4")[key]

    for bad in [(1,), (2, 3, 1), (0, 3), (2, -1), ("a", 3)]:
        with pytest.raises(ValueError, match="internal error: Broken reported chunks"):
            base._with_transform(Broken(bad))
        with pytest.raises(ValueError, match="internal error"):
            DynamicArray._from_transform(Broken(bad))
    assert base._with_transform(Broken(None)).chunks is None
    assert base._with_transform(Broken((np.int64(2), 3))).chunks == (2, 3)
