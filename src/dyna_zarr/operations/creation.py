"""Creation ops (nullary generative sources): zeros/ones/full/empty/random (+ *_like).

These build a lazy DynamicArray from a shape/dtype with no underlying array. The
GenerativeTransform *synthesizes* whatever region a read asks for, so creation stays
lazy and memory-bound (a huge ``zeros`` never allocates the whole thing) and composes
with every other op. It also honours the execution device: under ``compute(device='cuda')``
a ``zeros`` region is created directly on the GPU.

``random`` is **position-deterministic**: each element's value is a hash of its GLOBAL
index (and a per-array seed), so it is idempotent and chunk-invariant -- reading any region
gives the same values as the whole array sliced, and ``io.write(random(...))`` equals
``random(...).compute()``. (Unlike dask.array.random, whose values depend on the chunking.)
"""

import numpy as np

from ._base import Transform, _is_int_index
from ._backend import resolve_device, xp_for_device, to_device


class GenerativeTransform(Transform):
    """A nullary source: ``read(key)`` synthesizes the requested region."""

    def __init__(self, shape, dtype, chunks, mode, fill_value=0, seed=None, device=None):
        super().__init__()
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.chunks = tuple(chunks) if chunks is not None else self.shape
        self.mode = mode                 # 'zeros' | 'ones' | 'full' | 'empty' | 'random'
        self.fill_value = fill_value
        self.seed = seed
        self._seed64 = (int(seed) & 0xFFFFFFFFFFFFFFFF) if seed is not None else 0
        self.device = device

    def _axes(self, key):
        """Per input axis: (is_int, start, step, out_size) with concrete non-negatives."""
        ndim = len(self.shape)
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (ndim - len(key))
        axes = []
        for a in range(ndim):
            k = key[a]
            size = self.shape[a]
            if _is_int_index(k):
                idx = int(k) if k >= 0 else size + int(k)
                axes.append((True, idx, 1, None))
            else:
                start, stop, step = k.indices(size)
                axes.append((False, start, step, len(range(start, stop, step))))
        return axes

    def _random_region(self, axes, out_shape):
        """Position-deterministic uniform [0,1): value = splitmix64(global_flat_index, seed).
        Built with numpy (portable, exact); moved to the device by the caller."""
        ndim = len(self.shape)
        strides = [1] * ndim
        for a in range(ndim - 2, -1, -1):
            strides[a] = strides[a + 1] * self.shape[a + 1]
        # Work IN-PLACE in a single uint64 buffer `z` (+ one scratch `t`) so we don't
        # allocate several full-region arrays -- important for large chunks.
        z = np.full(out_shape, np.uint64(0x9E3779B97F4A7C15) + np.uint64(self._seed64),
                    dtype=np.uint64)
        oi = 0
        for a, (is_int, start, step, n) in enumerate(axes):
            stride = np.uint64(strides[a])
            if is_int:
                z += np.uint64(start) * stride
            else:
                coord = (np.uint64(start) + np.arange(n, dtype=np.uint64) * np.uint64(step)) * stride
                shp = [1] * len(out_shape)
                shp[oi] = n
                z += coord.reshape(shp)            # broadcast add, in place
                oi += 1
        t = z >> np.uint64(30); z ^= t; z *= np.uint64(0xBF58476D1CE4E5B9)
        np.right_shift(z, np.uint64(27), out=t); z ^= t; z *= np.uint64(0x94D049BB133111EB)
        np.right_shift(z, np.uint64(31), out=t); z ^= t
        del t
        z >>= np.uint64(11)
        u = z.astype(np.float64); u *= (1.0 / 9007199254740992.0)   # /2**53 -> [0,1)
        return u.astype(self.dtype, copy=False)

    def read(self, key):
        axes = self._axes(key)
        out_shape = tuple(n for (is_int, _s, _st, n) in axes if not is_int)
        dev = resolve_device(self.device)
        if self.mode == "random":
            return to_device(self._random_region(axes, out_shape), dev)
        xp = xp_for_device(dev)
        if self.mode == "zeros":
            return xp.zeros(out_shape, dtype=self.dtype)
        if self.mode == "ones":
            return xp.ones(out_shape, dtype=self.dtype)
        if self.mode == "full":
            return xp.full(out_shape, self.fill_value, dtype=self.dtype)
        return xp.empty(out_shape, dtype=self.dtype)     # 'empty' (uninitialized)


def _make(shape, dtype, chunks, mode, fill_value=0, seed=None, device=None):
    from dyna_zarr.dynamic_array import DynamicArray
    if np.isscalar(shape):
        shape = (int(shape),)
    t = GenerativeTransform(shape, dtype, chunks, mode, fill_value, seed, device)
    return DynamicArray._from_transform(t)


# --------------------------------------------------------------------------- #
# Public ops
# --------------------------------------------------------------------------- #

def zeros(shape, dtype=np.float64, chunks=None, device=None):
    """Lazy array of zeros."""
    return _make(shape, dtype, chunks, "zeros", device=device)


def ones(shape, dtype=np.float64, chunks=None, device=None):
    """Lazy array of ones."""
    return _make(shape, dtype, chunks, "ones", device=device)


def full(shape, fill_value, dtype=None, chunks=None, device=None):
    """Lazy array filled with ``fill_value``."""
    if dtype is None:
        dtype = np.array(fill_value).dtype
    return _make(shape, dtype, chunks, "full", fill_value=fill_value, device=device)


def empty(shape, dtype=np.float64, chunks=None, device=None):
    """Lazy uninitialized array (values are undefined; each read is fresh garbage)."""
    return _make(shape, dtype, chunks, "empty", device=device)


def random(shape, dtype=np.float32, chunks=None, seed=None, device=None):
    """Lazy uniform-random array in [0, 1), position-deterministic (chunk-invariant, so
    reads are consistent and io.write matches compute). ``seed`` fixes the values; if None,
    a fresh random seed is drawn once so separate ``random(...)`` calls differ (but each
    array is internally consistent)."""
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
    return _make(shape, dtype, chunks, "random", seed=seed, device=device)


def zeros_like(array, dtype=None, chunks=None, device=None):
    return zeros(array.shape, dtype=dtype or array.dtype, chunks=chunks or array.chunks, device=device)


def ones_like(array, dtype=None, chunks=None, device=None):
    return ones(array.shape, dtype=dtype or array.dtype, chunks=chunks or array.chunks, device=device)


def full_like(array, fill_value, dtype=None, chunks=None, device=None):
    return full(array.shape, fill_value, dtype=dtype or array.dtype, chunks=chunks or array.chunks, device=device)


def empty_like(array, dtype=None, chunks=None, device=None):
    return empty(array.shape, dtype=dtype or array.dtype, chunks=chunks or array.chunks, device=device)


__all__ = [
    "GenerativeTransform",
    "zeros", "ones", "full", "empty", "random",
    "zeros_like", "ones_like", "full_like", "empty_like",
]
