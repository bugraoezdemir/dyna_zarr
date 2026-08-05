"""
Correctness fuzz harness for the lazy Transform layer.

The risky part of dyna_zarr is the pull-based ``read(key)`` slice-translation math
inside every Transform: given an *output-space* slice, each transform must map it
back to the correct *input-space* read and return exactly what NumPy would for the
same operation followed by the same slice. This module fuzzes that contract.

For every operation we check two things against a NumPy reference:

1. **Full compute**  ``op(da).compute()``            == ``op_np(arr)``
2. **Random sub-slice**  ``op(da)[k]`` for many random ``k`` == ``op_np(arr)[k]``

(2) is where the ``read(key)`` composition math lives, so most ops are fuzzed there;
ops whose ``read`` deliberately materializes the whole input (pad/tile/roll/flip/
flatten) are still checked for correctness under sub-slicing even though they are not
memory-bound -- their memory behaviour is a *performance* concern tracked in the
non-shipped benchmark report, not a correctness bug.
"""

import numpy as np
import pytest
import zarr

from dyna_zarr import DynamicArray, operations


def _random_key_signed(rng, shape):
    """Random basic-index key that INCLUDES negative starts/stops/ints (regression guard
    for SliceTransform, which previously mis-sized negative slices like a[:-1])."""
    key = []
    for size in shape:
        if size == 0:
            key.append(slice(None))
            continue
        c = rng.integers(0, 5)
        if c == 0:
            i = int(rng.integers(-size, size))
            key.append(i)
        elif c == 1:
            key.append(slice(None))
        else:
            lo = int(rng.integers(-size - 1, size + 1))
            hi = int(rng.integers(-size - 1, size + 1))
            step = int(rng.integers(1, 3))
            key.append(slice(lo, hi, step))
    return tuple(key)


@pytest.mark.parametrize("chunks", [(4, 5, 6), (2, 2, 2), (1, 5, 6)])
def test_slice_negative_indices(chunks):
    """da[k] == arr[k] for keys with negative indices, single and composed (slice-of-slice)."""
    rng = np.random.default_rng(2024)
    arr = rng.random((4, 5, 6))
    da = DynamicArray(zarr.array(arr, chunks=chunks))
    for _ in range(200):
        k = _random_key_signed(rng, arr.shape)
        got = np.asarray(da[k].compute())
        ref = arr[k]
        assert got.shape == ref.shape, f"key {k}: {got.shape} != {ref.shape}"
        np.testing.assert_array_equal(got, ref, err_msg=f"key {k}")
        # compose a second (also-signed) slice on top
        if got.ndim:
            k2 = _random_key_signed(rng, got.shape)
            np.testing.assert_array_equal(np.asarray(da[k][k2].compute()), ref[k2],
                                          err_msg=f"composed {k} then {k2}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def da_from_numpy(arr: np.ndarray) -> DynamicArray:
    """Wrap a NumPy array as a DynamicArray via an in-memory zarr array."""
    z = zarr.array(arr, chunks=arr.shape)
    return DynamicArray(z)


def random_key(rng: np.random.Generator, shape: tuple) -> tuple:
    """Generate a random valid basic-indexing key for an array of ``shape``.

    Mixes integers, full slices, bounded slices, and stepped slices per axis.
    """
    key = []
    for size in shape:
        choice = rng.integers(0, 5)
        if choice == 0:                       # integer index
            key.append(int(rng.integers(0, size)))
        elif choice == 1:                     # full slice
            key.append(slice(None))
        elif choice == 2:                     # start:stop
            a, b = sorted(rng.integers(0, size + 1, size=2))
            key.append(slice(int(a), int(b)))
        elif choice == 3:                     # start:stop:step
            a, b = sorted(rng.integers(0, size + 1, size=2))
            step = int(rng.integers(1, 3))
            key.append(slice(int(a), int(b), step))
        else:                                 # open-ended
            a = int(rng.integers(0, size + 1))
            key.append(slice(a, None))
    return tuple(key)


def assert_close(got, ref, label):
    got = np.asarray(got)
    ref = np.asarray(ref)
    assert got.shape == ref.shape, f"{label}: shape {got.shape} != {ref.shape}"
    if np.issubdtype(ref.dtype, np.floating):
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-6,
                                   err_msg=f"{label}: value mismatch")
    else:
        np.testing.assert_array_equal(got, ref, err_msg=f"{label}: value mismatch")


# --------------------------------------------------------------------------- #
# Operation registry:  (name, dyna_fn, numpy_fn, supports_subslice)
#
# dyna_fn / numpy_fn take the *primary* operand (DynamicArray / ndarray) and,
# for binary ops, close over a second operand built the same way.
# --------------------------------------------------------------------------- #

BASE_SHAPE = (4, 5, 6)


def build_registry(arr, da, arr2, da2):
    """Return list of op specs bound to concrete operands."""
    perm = (2, 0, 1)
    return [
        # name,               dyna_fn,                                   numpy_fn,                                    subslice
        ("transpose",         lambda: operations.transpose(da, perm),    lambda: np.transpose(arr, perm),             True),
        ("swap_axes",         lambda: operations.swap_axes(da, 0, 2),    lambda: np.swapaxes(arr, 0, 2),              True),
        ("expand_dims",       lambda: operations.expand_dims(da, 1),     lambda: np.expand_dims(arr, 1),              True),
        ("squeeze",           lambda: operations.squeeze(
                                  operations.expand_dims(da, 1), 1),     lambda: np.squeeze(np.expand_dims(arr, 1), 1), True),
        ("flip",              lambda: operations.flip(da, 1),            lambda: np.flip(arr, 1),                     True),
        ("roll",              lambda: operations.roll(da, 2, 0),         lambda: np.roll(arr, 2, 0),                  True),
        ("pad",               lambda: operations.pad(da, 1),             lambda: np.pad(arr, 1),                      True),
        ("tile",              lambda: operations.tile(da, (1, 2, 1)),    lambda: np.tile(arr, (1, 2, 1)),             True),
        ("concatenate",       lambda: operations.concatenate([da, da2], 0), lambda: np.concatenate([arr, arr2], 0),  True),
        ("stack",             lambda: operations.stack([da, da2], 0),    lambda: np.stack([arr, arr2], 0),            True),
        ("clip",              lambda: operations.clip(da, 0.2, 0.8),     lambda: np.clip(arr, 0.2, 0.8),              True),
        ("abs",               lambda: operations.abs(da),                lambda: np.abs(arr),                         True),
        ("sign",              lambda: operations.sign(da),               lambda: np.sign(arr),                        True),
        ("round",             lambda: operations.round(da, 2),           lambda: np.round(arr, 2),                    True),
        ("sqrt",              lambda: operations.sqrt(da),               lambda: np.sqrt(np.abs(arr)),                True),
        ("multiply",          lambda: operations.multiply(da, da2),      lambda: arr * arr2,                          True),
        ("add",               lambda: operations.add(da, da2),           lambda: arr + arr2,                          True),
        ("multiply_scalar",   lambda: operations.multiply(da, 3.0),      lambda: arr * 3.0,                           True),
        ("where",             lambda: operations.where(da, da, da2),     lambda: np.where(arr, arr, arr2),            True),
        # --- map_blocks ufunc surface: unary (da2 is strictly positive, safe for log/recip) ---
        ("negative",          lambda: operations.negative(da),           lambda: np.negative(arr),                    True),
        ("square",            lambda: operations.square(da),             lambda: np.square(arr),                      True),
        ("exp",               lambda: operations.exp(da),                lambda: np.exp(arr),                         True),
        ("log",               lambda: operations.log(da2),               lambda: np.log(arr2),                        True),
        ("log2",              lambda: operations.log2(da2),              lambda: np.log2(arr2),                       True),
        ("log10",             lambda: operations.log10(da2),             lambda: np.log10(arr2),                      True),
        ("floor",             lambda: operations.floor(da2),             lambda: np.floor(arr2),                      True),
        ("ceil",              lambda: operations.ceil(da2),              lambda: np.ceil(arr2),                       True),
        ("reciprocal",        lambda: operations.reciprocal(da2),        lambda: np.reciprocal(arr2),                 True),
        ("astype_i16",        lambda: operations.astype(da2, np.int16),  lambda: arr2.astype(np.int16),               True),
        # --- binary ---
        ("subtract",          lambda: operations.subtract(da, da2),      lambda: np.subtract(arr, arr2),              True),
        ("divide",            lambda: operations.divide(da, da2),        lambda: np.divide(arr, arr2),                True),
        ("floor_divide",      lambda: operations.floor_divide(da, da2),  lambda: np.floor_divide(arr, arr2),          True),
        ("mod",               lambda: operations.mod(da, da2),           lambda: np.mod(arr, arr2),                   True),
        ("power",             lambda: operations.power(da2, da2),        lambda: np.power(arr2, arr2),                True),
        ("maximum",           lambda: operations.maximum(da, da2),       lambda: np.maximum(arr, arr2),               True),
        ("minimum",           lambda: operations.minimum(da, da2),       lambda: np.minimum(arr, arr2),               True),
        ("add_scalar",        lambda: operations.add(da, 2.5),           lambda: arr + 2.5,                           True),
        # --- comparisons & logical (bool out) ---
        ("greater",           lambda: operations.greater(da, da2),       lambda: np.greater(arr, arr2),               True),
        ("less_equal",        lambda: operations.less_equal(da, da2),    lambda: np.less_equal(arr, arr2),            True),
        ("equal",             lambda: operations.equal(da, da),          lambda: np.equal(arr, arr),                  True),
        ("logical_and",       lambda: operations.logical_and(da, da2),   lambda: np.logical_and(arr, arr2),           True),
        ("logical_not",       lambda: operations.logical_not(da),        lambda: np.logical_not(arr),                 True),
        # reductions: full-compute only (sub-slicing a reduced axis is a separate contract)
        ("min_axis0",         lambda: operations.min(da, 0),             lambda: np.min(arr, 0),                      False),
        ("max_axis0",         lambda: operations.max(da, 0),             lambda: np.max(arr, 0),                      False),
    ]


@pytest.fixture
def operands():
    rng = np.random.default_rng(0)
    arr = rng.random(BASE_SHAPE)               # float64 in [0, 1)
    arr[arr < 0.05] = 0.0                       # a few exact zeros for sign/where
    arr2 = rng.random(BASE_SHAPE) + 0.1
    return arr, da_from_numpy(arr), arr2, da_from_numpy(arr2)


def op_ids(reg):
    return [spec[0] for spec in reg]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def _registry(operands):
    return build_registry(*operands)


def test_full_compute_matches_numpy(operands):
    """op(da).compute() == op_np(arr) for every operation."""
    reg = _registry(operands)
    failures = []
    for name, dyna_fn, np_fn, _ in reg:
        try:
            assert_close(dyna_fn().compute(), np_fn(), f"{name}[full]")
        except Exception as e:
            failures.append(f"{name}[full]: {type(e).__name__}: {str(e).splitlines()[0]}")
    assert not failures, "Full-compute mismatches:\n" + "\n".join(failures)


@pytest.mark.parametrize("seed", [0, 1, 7, 1234, 99999])
def test_random_subslice_matches_numpy(operands, seed):
    """op(da)[k] == op_np(arr)[k] over many random keys (the read(key) math)."""
    rng = np.random.default_rng(seed)
    reg = _registry(operands)
    failures = []
    for name, dyna_fn, np_fn, subslice in reg:
        if not subslice:
            continue
        ref_full = np_fn()
        result = dyna_fn()
        for _ in range(40):
            k = random_key(rng, result.shape)
            try:
                got = result[k].compute()
                assert_close(got, ref_full[k], f"{name}[{k}]")
            except Exception as e:
                failures.append(f"{name}[{k}]: {type(e).__name__}: {str(e).splitlines()[0]}")
                break  # one failure per op is enough signal
    assert not failures, "Sub-slice mismatches:\n" + "\n".join(failures)
