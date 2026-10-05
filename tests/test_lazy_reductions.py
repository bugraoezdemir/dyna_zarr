"""Lazy reductions, broadcasting, the small-result cache, and persist().

The contract under test:
- map_blocks operands broadcast like NumPy, so `x > x.mean()` and
  `x - x.mean(axis=0, keepdims=True)` are ordinary lazy chains, exact at every key.
- Reduction METHODS are lazy (like operations.<reducer>); a one-element result
  converts on demand (float/int/bool/index).
- Tier 1: a reduction with output <= 1 MiB is computed once, fused with its siblings
  on the same input, and cached until io.clear_cache() / arr.clear_cache(). The cache
  lives on the input, and a cached min/max/sum/mean brings the other three along
  (var/std bring all of them) when the bundle fits the same 1 MiB.
- Tier 2: a larger reduction stays blockwise. Broadcast back over the axes it reduced,
  io.write spans those axes when the region budget allows, else warns with the repeat
  count and recomputes; persist() computes it once.

Pass counts are measured by counting elements read from the SOURCE array.
"""

import contextlib
import gc
import io as _stdio
import os
import threading
import warnings

import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from dyna_zarr import DynamicArray, io
from dyna_zarr import operations as ops
from dyna_zarr.io import RepeatedReductionWarning


@pytest.fixture
def source_reads(monkeypatch):
    """Counter of elements read from untransformed (source) arrays."""
    lock = threading.Lock()
    count = {"n": 0}
    orig = DynamicArray._read_direct

    def counting(self, key):
        out = orig(self, key)
        if self._transform is None:
            with lock:
                count["n"] += int(np.asarray(out).size)
        return out

    monkeypatch.setattr(DynamicArray, "_read_direct", counting)
    return count


@pytest.fixture(autouse=True)
def _fresh_cache():
    io.clear_cache()
    yield
    io.clear_cache()


def _write(*args, **kwargs):
    """io.write with its progress prints silenced."""
    with contextlib.redirect_stdout(_stdio.StringIO()):
        io.write(*args, **kwargs)


def _vol(tmp_path, shape=(32, 128, 128), chunks=(8, 32, 32), dtype="float32", seed=0):
    data = np.random.default_rng(seed).random(shape).astype(dtype)
    z = zarr.create_array(str(tmp_path / "src.zarr"), shape=shape, chunks=chunks, dtype=dtype)
    z[...] = data
    return data, io.read(str(tmp_path / "src.zarr"))


# --------------------------------------------------------------------------- #
# Broadcasting
# --------------------------------------------------------------------------- #

_X = np.random.default_rng(1).random((4, 6, 8)).astype("f4")

_BROADCAST_CASES = {
    "nd > 0-d": (lambda a: a > ops.mean(a), lambda x: x > x.mean(dtype="f4")),
    "0-d < nd": (lambda a: ops.less(ops.mean(a), a), lambda x: x.mean(dtype="f4") < x),
    "nd - keepdims": (lambda a: a - ops.mean(a, axis=0, keepdims=True),
                      lambda x: x - x.mean(0, keepdims=True, dtype="f4")),
    "nd - trailing (6,8)": (lambda a: a - ops.max(a, axis=0), lambda x: x - x.max(0)),
    "nd * ndarray (8,)": (lambda a: a * np.arange(8, dtype="f4"),
                          lambda x: x * np.arange(8, dtype="f4")),
    "(1,6,1) + (4,1,8)": (
        lambda a: ops.max(a, axis=(0, 2), keepdims=True) + ops.min(a, axis=1, keepdims=True),
        lambda x: x.max((0, 2), keepdims=True) + x.min(1, keepdims=True)),
    "where(nd > 0-d, nd, 0-d)": (lambda a: ops.where(a > ops.mean(a), a, ops.min(a)),
                                 lambda x: np.where(x > x.mean(dtype="f4"), x, x.min())),
}

_KEYS = [(), (1,), (slice(1, 3), 2), (slice(None, None, 2), slice(1, 5), -1),
         (-1, slice(5, 1)), (slice(2, 2),)]


@pytest.mark.parametrize("name", list(_BROADCAST_CASES))
def test_broadcast_matches_numpy_at_every_key(name):
    lazy, ref = _BROADCAST_CASES[name]
    a = DynamicArray(zarr.array(_X, chunks=(2, 3, 4)))
    r, e = lazy(a), ref(_X)
    assert r.shape == e.shape
    np.testing.assert_allclose(r.compute(), e, rtol=1e-6)
    for k in _KEYS:
        assert r[k].shape == e[k].shape, k
        np.testing.assert_allclose(np.asarray(r[k].compute()), e[k], rtol=1e-6)


def test_incompatible_shapes_still_raise():
    a = DynamicArray(zarr.array(_X, chunks=(2, 3, 4)))
    with pytest.raises(ValueError, match="broadcast"):
        ops.add(a, ops.max(a, axis=1))                  # (4,6,8) vs (4,8)


def test_ndarray_operand_is_read_blockwise(tmp_path):
    # A raw ndarray used to be handed to func WHOLE next to operand blocks, so any
    # region smaller than the array was misaligned. It is now read per region.
    data, a = _vol(tmp_path)
    w = np.linspace(0, 1, 128, dtype="f4")
    _write(a * w, str(tmp_path / "out.zarr"), region_shape=(8, 32, 32))
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "out.zarr"))[...], data * w)


# --------------------------------------------------------------------------- #
# Lazy methods and scalar conversion
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["min", "max", "sum", "mean", "prod", "std", "var",
                                    "any", "all", "argmin", "argmax", "median"])
def test_reduction_methods_are_lazy(method):
    a = DynamicArray(zarr.array(_X, chunks=(2, 3, 4)))
    r = getattr(a, method)()
    assert isinstance(r, DynamicArray)
    np.testing.assert_allclose(np.asarray(r.compute()), getattr(np, method)(_X), rtol=1e-5)


@pytest.mark.parametrize("fn", ["argmin", "argmax"])
def test_arg_reductions_axis_none_is_flat_index(fn):
    # axis=None used to raise for ndim > 1 ("take a single axis or axis=None").
    a = DynamicArray(zarr.array(_X, chunks=(2, 3, 4)))
    assert int(getattr(ops, fn)(a)) == int(getattr(np, fn)(_X))
    kd = getattr(ops, fn)(a, keepdims=True)
    assert kd.shape == (1, 1, 1)
    assert int(np.asarray(kd.compute()).ravel()[0]) == int(getattr(np, fn)(_X))
    with pytest.raises(ValueError, match="single axis"):
        getattr(ops, fn)(a, axis=(0, 1))


def test_scalar_conversion():
    a = DynamicArray(zarr.array(_X, chunks=(2, 3, 4)))
    ai = DynamicArray(zarr.array(np.arange(24, dtype="i4").reshape(2, 3, 4), chunks=(1, 3, 4)))
    assert float(a.mean()) == pytest.approx(float(_X.mean(dtype="f4")), rel=1e-6)
    assert int(ai.max()) == 23
    assert [10, 20, 30][ai.min() + 1] == 20             # __index__
    assert bool(a.min() >= 0) is True
    if a.max() > 2:                                     # __bool__ on a 1-element array
        pytest.fail("max of values in [0, 1) is not > 2")
    assert complex(a.sum()) == pytest.approx(complex(_X.sum(dtype="f4")), rel=1e-5)


def test_ambiguous_conversions_raise_without_reading(source_reads):
    a = DynamicArray(zarr.array(_X, chunks=(2, 3, 4)))
    with pytest.raises(ValueError, match="ambiguous"):
        bool(a > 0)
    with pytest.raises(TypeError, match="exactly one element"):
        float(a)
    with pytest.raises(TypeError, match="integer"):
        [1, 2][a.mean()]
    assert source_reads["n"] == 0


# --------------------------------------------------------------------------- #
# Tier 1: small results computed once, fused, cached
# --------------------------------------------------------------------------- #

def test_headline_chain_is_two_passes_and_exact(tmp_path, source_reads):
    data, vol = _vol(tmp_path)
    smoothed = ops.gaussian_filter(vol, sigma=(1, 2, 2))
    level = smoothed.mean() + 3 * smoothed.std()
    mask = (smoothed > level).astype("uint8")
    assert source_reads["n"] == 0                       # building the chain reads nothing

    _write(mask, str(tmp_path / "mask.zarr"), region_shape=(8, 64, 64))

    ref_s = ndi.gaussian_filter(data, sigma=(1, 2, 2))
    ref = (ref_s > ref_s.mean(dtype="f4") + 3 * ref_s.std(dtype="f4")).astype("uint8")
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path / "mask.zarr"))[...], ref)
    total = source_reads["n"]

    # The same write with a constant threshold costs exactly the write pass (incl. its
    # halo). The statistics must add ONE fused pass on top: not two (mean and std
    # separately, as when the methods were eager), and not one per region.
    source_reads["n"] = 0
    _write((smoothed > 0.5).astype("uint8"), str(tmp_path / "plain.zarr"),
           region_shape=(8, 64, 64))
    assert total - source_reads["n"] == data.size


def test_sibling_reductions_share_one_pass(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    stats = a.mean() + a.std() + a.min() + a.max()
    value = float(stats)
    assert source_reads["n"] == data.size               # four reductions, one pass
    assert value == pytest.approx(float(data.mean() + data.std() + data.min() + data.max()),
                                  rel=1e-5)


def test_cached_result_is_reused_until_cleared(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    m = a.mean()
    first = float(m)
    reads_once = source_reads["n"]
    assert float(m) == first and source_reads["n"] == reads_once      # cached

    # rewrite the source in place: the cache is stale by design...
    zarr.open_array(str(tmp_path / "src.zarr"))[...] = data + 1
    assert float(m) == first
    # ...until cleared, per array or globally
    m.clear_cache()
    assert float(m) == pytest.approx(first + 1, rel=1e-5)
    zarr.open_array(str(tmp_path / "src.zarr"))[...] = data + 2
    io.clear_cache()
    assert float(m) == pytest.approx(first + 2, rel=1e-5)


def test_cache_is_computed_once_under_concurrency(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    m = ops.mean(a)
    barrier = threading.Barrier(8)
    results = []

    def worker():
        barrier.wait()
        results.append(float(np.asarray(m._read_direct(()))))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 1
    assert source_reads["n"] == data.size               # one computation, not eight


def test_cache_cap_is_one_mib():
    small = ops.mean(DynamicArray(np.zeros((4, 512, 512), "f4")), axis=0)   # 1 MiB
    large = ops.mean(DynamicArray(np.zeros((4, 512, 512), "f8")), axis=0)   # 2 MiB
    assert small._transform.cacheable
    assert not large._transform.cacheable


def test_partial_read_of_uncached_small_reduction_stays_partial(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    mip = ops.max(a, axis=0)                            # 64 KiB: cacheable
    np.testing.assert_array_equal(mip[0:4, 0:4].compute(), data.max(0)[0:4, 0:4])
    assert source_reads["n"] == data.shape[0] * 4 * 4   # only the columns asked for


def test_projection_is_blockwise_not_recomputed_per_region(tmp_path, source_reads):
    data, a = _vol(tmp_path, shape=(32, 256, 256))
    io.clear_cache()
    _write(ops.max(a, axis=0), str(tmp_path / "mip.zarr"), chunks=(64, 64),
           region_shape=(64, 64))
    assert source_reads["n"] == data.size               # 16 regions, 1 pass
    source_reads["n"] = 0
    io.clear_cache()
    _write(ops.gaussian_filter(ops.max(a, axis=0), 2), str(tmp_path / "mipg.zarr"),
           chunks=(64, 64), region_shape=(64, 64))
    assert source_reads["n"] / data.size < 1.5          # halo only
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "mipg.zarr"))[...],
                               ndi.gaussian_filter(data.max(0), 2), atol=1e-6)


# --------------------------------------------------------------------------- #
# Tier 1: statistics cached on the INPUT, computed in bundles
# --------------------------------------------------------------------------- #

def _passes(source_reads, data):
    return source_reads["n"] / data.size


def test_separate_calls_reuse_the_cache(tmp_path, source_reads):
    # The cache lives on the input, not on the result node, so a fresh a.mean() hits it.
    data, a = _vol(tmp_path)
    assert float(a.mean()) == float(a.mean())
    assert _passes(source_reads, data) == 1


def test_cheap_group_comes_with_any_member(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    values = [float(f()) for f in (a.mean, a.min, a.max, a.sum)]
    assert _passes(source_reads, data) == 1
    np.testing.assert_allclose(values, [data.mean(), data.min(), data.max(), data.sum()],
                               rtol=1e-5)


def test_mean_does_not_pay_for_std_but_std_brings_everything(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    float(a.mean())
    float(a.std())
    assert _passes(source_reads, data) == 2         # sum of squares is not bundled with mean

    io.clear_cache()
    source_reads["n"] = 0
    float(a.std())
    values = [float(f) for f in (a.var(), a.std(ddof=1), a.mean(), a.min(), a.max())]
    assert _passes(source_reads, data) == 1         # ddof applies per reader, not per cache
    np.testing.assert_allclose(
        values, [data.var(), data.std(ddof=1), data.mean(), data.min(), data.max()], rtol=1e-5)


def test_partial_reductions_bundle_per_axes(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    a.mean(axis=0).compute()
    mx = a.max(axis=0).compute()
    mn = a.min(axis=0, keepdims=True).compute()     # keepdims applies per reader
    assert _passes(source_reads, data) == 1
    np.testing.assert_array_equal(mx, data.max(0))
    np.testing.assert_array_equal(mn, data.min(0, keepdims=True))
    a.max(axis=1).compute()                         # other axes: their own pass
    assert _passes(source_reads, data) == 2


def test_bundle_over_the_cap_falls_back_to_the_requested_statistic(source_reads):
    data = np.zeros((4, 512, 512), "f4")
    a = DynamicArray(data)
    a.mean(axis=0).compute()                        # 1 MiB sum fits; 3 MiB bundle does not
    a.max(axis=0).compute()
    assert _passes(source_reads, data) == 2


@pytest.mark.parametrize("first, then", [("std", "mean"), ("mean", "std"), ("var", "max")])
def test_results_do_not_depend_on_what_was_cached_first(tmp_path, first, then):
    data, a = _vol(tmp_path)
    alone = float(getattr(a, then)())
    io.clear_cache()
    float(getattr(a, first)())
    assert float(getattr(a, then)()) == alone       # bitwise


def test_clearing_one_statistic_drops_its_siblings(tmp_path, source_reads):
    data, a = _vol(tmp_path)
    m = a.mean()
    float(m)
    zarr.open_array(str(tmp_path / "src.zarr"))[...] = data + 1
    assert float(a.max()) == pytest.approx(float(data.max()))      # cached with the mean
    m.clear_cache()
    assert float(a.max()) == pytest.approx(float(data.max()) + 1)  # gone with it


def test_empty_input_keeps_numpy_semantics():
    # the cheap group is not added for an empty input: min would raise where sum does not
    a = DynamicArray(np.zeros((0, 4), "f4"))
    assert float(a.sum()) == 0.0
    with pytest.raises(ValueError):
        float(a.min())


# --------------------------------------------------------------------------- #
# Tier 2: large partial reductions broadcast back
# --------------------------------------------------------------------------- #

def _tier2(tmp_path):
    # (1, 512, 512) float64 = 2 MiB: above the cache cap, so it stays blockwise.
    data, a = _vol(tmp_path, shape=(16, 512, 512), chunks=(4, 128, 128), dtype="float64")
    m = ops.mean(a, axis=0, keepdims=True)
    assert not m._transform.cacheable
    return data, a, m, data - data.mean(0, keepdims=True)


def test_broadcast_reduction_regions_span_axis_when_budget_allows(tmp_path, source_reads):
    data, a, m, ref = _tier2(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RepeatedReductionWarning)
        _write(a - m, str(tmp_path / "out.zarr"), chunks=(4, 128, 128))
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "out.zarr"))[...], ref)
    assert source_reads["n"] == 2 * data.size           # reduction once + x once


def test_broadcast_reduction_warns_with_repeat_count_and_is_correct(tmp_path, source_reads):
    data, a, m, ref = _tier2(tmp_path)
    with pytest.warns(RepeatedReductionWarning, match=r"re-read about 4x"):
        _write(a - m, str(tmp_path / "out.zarr"), chunks=(4, 128, 128),
               region_shape=(4, 128, 128))              # exact region: honoured, so it warns
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "out.zarr"))[...], ref)
    assert source_reads["n"] == 5 * data.size           # 4 re-reads of the mean + x


def test_persist_computes_once_and_cleans_up(tmp_path, source_reads):
    data, a, m, ref = _tier2(tmp_path)
    pm = m.persist()
    tmpdir = pm._persist_owner.tmpdir
    assert os.path.isdir(tmpdir)
    assert source_reads["n"] == data.size
    derived = a - pm                                    # holds pm through its chain
    del pm
    gc.collect()
    assert os.path.isdir(tmpdir)                        # still read by `derived`
    with warnings.catch_warnings():
        warnings.simplefilter("error", RepeatedReductionWarning)
        _write(derived, str(tmp_path / "out.zarr"), chunks=(4, 128, 128),
               region_shape=(4, 128, 128))
    np.testing.assert_allclose(zarr.open_array(str(tmp_path / "out.zarr"))[...], ref)
    del derived
    gc.collect()
    assert not os.path.exists(tmpdir)


def test_persist_to_path_keeps_store_and_small_results_stay_in_memory(tmp_path):
    data, a = _vol(tmp_path)
    kept = ops.max(a, axis=0).persist(str(tmp_path / "mip.zarr"))
    assert getattr(kept, "_persist_owner", None) is None
    np.testing.assert_array_equal(kept.compute(), data.max(0))
    del kept
    gc.collect()
    assert os.path.isdir(tmp_path / "mip.zarr")
    small = a.mean().persist()                          # 0-d: kept in memory
    assert getattr(small, "_persist_owner", None) is None
    assert float(small) == pytest.approx(float(data.mean()), rel=1e-6)
