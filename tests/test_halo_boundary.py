"""The halo is capped PER SIDE at the real data that side can supply.

Uncapped, a depth-4 halo pads a length-2 axis out to 10, and scipy's separable
passes multiply that again (17.7 GB peak measured on a 162 MB 5D array). The cap
is safe because pre-padding a block and letting the func apply the same boundary
mode itself give bit-identical results, once the halo is at least the kernel's
own reach.

It has to be per side. A single shared depth collapses to 0 on a core that spans
the whole axis, which drops the boundary border entirely and misaligns the crop;
that broke gaussian, gaussian_laplace and gradient_magnitude at every edge while
leaving the block data itself looking correct.

Two traps this file exists to catch:
  - comparing np.pad against scipy WITHOUT the mode table shows a spurious
    mismatch, because numpy's 'reflect' is scipy's 'mirror' and vice versa;
  - `arr[key]` is lazy, so a test that omits `.compute()` passes vacuously.
"""

import numpy as np
import pytest
from scipy import ndimage

from dyna_zarr import DynamicArray
from dyna_zarr.operations.neighborhood import map_overlap


# A gaussian's footprint is unbounded, so ANY finite halo truncates it slightly.
# Depth 4 against sigma 2 leaves ~3e-3; these tests use a depth wide enough that
# the truncation is below the tolerance, so a real regression is not masked.
SIGMA = 1.0
DEPTH = 8


@pytest.mark.parametrize("size", [2, 3, 5])
@pytest.mark.parametrize("mode", ["reflect", "mirror", "wrap", "nearest"])
def test_prepadding_equals_scipys_own_mode(size, mode):
    """The premise the halo cap rests on.

    Padding a block ourselves and letting the func apply the same mode itself
    give EXACTLY the same answer, once the halo is at least the kernel's reach.
    That is what makes a halo redundant on an axis the core already spans, and
    so what makes the per-side cap safe.

    It must use dyna's own scipy->numpy mode table, not the identity: numpy's
    'reflect' is scipy's 'mirror' and vice versa. Comparing with the identity
    mapping shows a spurious ~3e-3 mismatch that looks exactly like a bug.
    """
    from dyna_zarr.operations.neighborhood import _BOUNDARY_TO_NPPAD

    rng = np.random.default_rng(0)
    a = rng.random((size, 24, 24)).astype(np.float64)
    npmode, kw = _BOUNDARY_TO_NPPAD[mode]
    padded = np.pad(a, ((DEPTH, DEPTH), (0, 0), (0, 0)), mode=npmode, **kw)
    via_pad = ndimage.gaussian_filter(padded, SIGMA,
                                      mode=mode)[DEPTH:DEPTH + size]
    direct = ndimage.gaussian_filter(a, SIGMA, mode=mode)
    np.testing.assert_array_equal(via_pad, direct)


@pytest.mark.parametrize("depth", [1, 2])
def test_too_shallow_a_halo_truncates_the_kernel(depth):
    """The other half of the story, so the test above is not read as licence.

    Below the kernel's reach the two paths DO differ, and by truncation rather
    than by boundary convention. That is inherent to a finite halo and applies
    equally with or without the cap: depth is the caller's responsibility.
    """
    rng = np.random.default_rng(0)
    a = rng.random((5, 24, 24)).astype(np.float64)
    from dyna_zarr.operations.neighborhood import _BOUNDARY_TO_NPPAD
    npmode, kw = _BOUNDARY_TO_NPPAD["reflect"]
    padded = np.pad(a, ((depth, depth), (0, 0), (0, 0)), mode=npmode, **kw)
    via_pad = ndimage.gaussian_filter(padded, 2.0, mode="reflect")[depth:depth + 5]
    direct = ndimage.gaussian_filter(a, 2.0, mode="reflect")
    assert not np.allclose(via_pad, direct, atol=1e-6)


@pytest.mark.parametrize("mode", ["reflect", "nearest", "mirror", "constant"])
@pytest.mark.parametrize("shape", [(2, 24, 24), (3, 24, 24), (2, 3, 16, 16)])
def test_map_overlap_matches_scipy_on_short_axes(mode, shape):
    """The regression a cap caused: a core spanning a whole short axis lost its
    boundary padding, and every edge value came out wrong."""
    rng = np.random.default_rng(1)
    data = rng.random(shape).astype(np.float32)
    depth = tuple(DEPTH for _ in shape)

    got = map_overlap(DynamicArray(data),
                      lambda b: ndimage.gaussian_filter(b, SIGMA, mode=mode),
                      depth=depth, boundary=mode).compute()
    expected = ndimage.gaussian_filter(data, SIGMA, mode=mode)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


def test_short_axes_are_not_inflated_by_the_halo():
    """What the cap is FOR: a depth-4 halo on a length-2 axis would pad it to
    10 uncapped, and scipy's separable passes multiply that again (17.7 GB peak
    measured on a 162 MB 5D array). Capped, the axis is read whole and nothing
    is added.

    Note `arr[key]` is LAZY -- it builds another transform. Without the
    `.compute()` the func never runs and this test passes vacuously.
    """
    seen = []

    def record(block):
        seen.append(block.shape)
        return block

    shape = (2, 3, 16, 64, 64)
    data = np.zeros(shape, dtype=np.float32)
    arr = map_overlap(DynamicArray(data), record, depth=(4, 4, 4, 4, 4),
                      boundary="reflect", dtype=np.float32)
    seen.clear()
    arr[:, :, 0:8].compute()

    assert seen, "func never ran"
    for block in seen:
        assert block[0] == 2, f"length-2 axis inflated to {block[0]}"
        assert block[1] == 3, f"length-3 axis inflated to {block[1]}"
        # and the whole block stays near the core rather than exploding
        assert np.prod(block) <= 4 * (2 * 3 * 8 * 64 * 64), block


@pytest.mark.parametrize("name,fn", [
    ("gaussian", lambda b: ndimage.gaussian_filter(b, SIGMA, mode="reflect")),
    ("gaussian_laplace",
     lambda b: ndimage.gaussian_laplace(b, SIGMA, mode="reflect")),
    ("gradient_magnitude",
     lambda b: ndimage.gaussian_gradient_magnitude(b, SIGMA, mode="reflect")),
])
def test_large_sigma_ops_match_scipy_on_a_short_axis(name, fn):
    """The ops that broke under the cap: their depth is comparable to the axis."""
    rng = np.random.default_rng(2)
    data = rng.random((2, 20, 20)).astype(np.float32)
    got = map_overlap(DynamicArray(data), fn, depth=(DEPTH,) * 3,
                      boundary="reflect").compute()
    np.testing.assert_allclose(got, fn(data), rtol=1e-4, atol=1e-5,
                               err_msg=name)
