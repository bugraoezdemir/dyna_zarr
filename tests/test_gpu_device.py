"""GPU (CuPy/CUDA) device-path tests.

LOCAL ONLY. These are marked ``gpu`` and are not run by CI: GitHub's standard
runners have no CUDA device, and the project has deliberately not wired up a paid
GPU runner or a self-hosted one. Run them by hand on each platform before shipping
a release that touches the device path::

    pip install -e ".[dev,gpu-cu12]"     # match the CUDA from `nvidia-smi`
    pytest -m gpu

The whole file skips when CuPy is missing OR when a CuPy import succeeds but the
CUDA runtime is unusable (a driver older than the CuPy build is the common case,
and it raises only when a device call is made, not at import). The skip reason
says which, so a machine that is *supposed* to have a working GPU does not quietly
report "no tests ran".

What these assert is CORRECTNESS, not speed: every op must produce the same result
on the GPU as on the CPU, and every terminal call must hand back host memory.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from dyna_zarr import DynamicArray, io  # noqa: E402
from dyna_zarr import operations as ops  # noqa: E402


def _cuda_status():
    """(usable, reason) for the CUDA runtime on this machine."""
    try:
        import cupy
    except ImportError:
        return False, "CuPy is not installed (pip install 'dyna-zarr[gpu-cu12]')"
    try:
        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:                      # driver/runtime mismatch, no device
        return False, f"CuPy is installed but CUDA is unusable: {type(exc).__name__}: {exc}"
    if count < 1:
        return False, "CuPy is installed but no CUDA device is visible"
    return True, ""


_CUDA_OK, _CUDA_WHY = _cuda_status()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _CUDA_OK, reason=_CUDA_WHY),
]

SHAPE = (8, 64, 64)
CHUNKS = (4, 32, 32)


@pytest.fixture
def data():
    return np.round(
        np.random.default_rng(0).random(SHAPE, dtype=np.float32) * 100, 1
    )


@pytest.fixture
def source(tmp_path, data):
    path = tmp_path / "src.zarr"
    z = zarr.create_array(store=str(path), shape=SHAPE, chunks=CHUNKS, dtype="float32")
    z[:] = data
    return path


# --------------------------------------------------------------------------- #
# the device seam itself
# --------------------------------------------------------------------------- #

def test_to_device_roundtrip():
    from dyna_zarr.operations._backend import to_device

    host = np.arange(12, dtype="float32")
    dev = to_device(host, "cuda")
    assert type(dev).__module__.split(".")[0] == "cupy"
    back = to_device(dev, "cpu")
    assert isinstance(back, np.ndarray)
    assert np.array_equal(back, host)


def test_to_device_is_a_noop_when_already_there():
    from dyna_zarr.operations._backend import to_device

    host = np.arange(4, dtype="float32")
    assert to_device(host, "cpu") is host
    dev = to_device(host, "cuda")
    assert to_device(dev, "cuda") is dev      # no redundant copy


def test_device_context_is_thread_local():
    """io.write runs its op chain on reader THREADS, so the context must not leak."""
    import threading

    from dyna_zarr.operations._backend import current_device, device_context

    seen = {}

    def worker():
        seen["thread"] = current_device()

    with device_context("cuda"):
        assert current_device() == "cuda"
        t = threading.Thread(target=worker)
        t.start()
        t.join()

    assert seen["thread"] == "cpu"            # a fresh thread inherits nothing
    assert current_device() == "cpu"          # and the context is restored


# --------------------------------------------------------------------------- #
# compute(device=...)
# --------------------------------------------------------------------------- #

def test_compute_on_cuda_returns_host_array(source, data):
    arr = io.read(source)
    out = arr.compute(device="cuda")
    assert isinstance(out, np.ndarray), "compute must return host memory, not a CuPy array"
    assert np.array_equal(out, data)


@pytest.mark.parametrize(
    "build,expected",
    [
        (lambda a: a + 1.0, lambda d: d + 1.0),
        (lambda a: a.astype("float64"), lambda d: d.astype("float64")),
        (lambda a: a.clip(10, 50), lambda d: np.clip(d, 10, 50)),
        (lambda a: ops.sqrt(ops.abs(a)), lambda d: np.sqrt(np.abs(d))),
    ],
    ids=["add", "astype", "clip", "sqrt-abs"],
)
def test_pointwise_matches_cpu(source, data, build, expected):
    arr = io.read(source)
    gpu = build(arr).compute(device="cuda")
    cpu = build(io.read(source)).compute()
    assert np.allclose(gpu, expected(data), rtol=1e-6, atol=1e-6)
    assert np.allclose(gpu, cpu, rtol=1e-6, atol=1e-6)


def test_neighborhood_filter_matches_cpu(source, data):
    """A halo op exercises more of the seam than pointwise: the block function has
    to dispatch on the array namespace rather than calling scipy directly."""
    arr = io.read(source)
    gpu = ops.gaussian_filter(arr, sigma=1.5).compute(device="cuda")
    cpu = ops.gaussian_filter(io.read(source), sigma=1.5).compute()
    assert isinstance(gpu, np.ndarray)
    assert np.allclose(gpu, cpu, rtol=1e-4, atol=1e-4)


def test_slicing_before_and_after_a_cuda_op(source, data):
    arr = io.read(source)
    got = arr[2:6, 0:32, 0:32].clip(0, 50).compute(device="cuda")
    assert np.allclose(got, np.clip(data[2:6, 0:32, 0:32], 0, 50))


# --------------------------------------------------------------------------- #
# io.write(device=...)
# --------------------------------------------------------------------------- #

def test_write_with_cuda_matches_cpu(tmp_path, source, data):
    out = tmp_path / "gpu_out.zarr"
    io.write((io.read(source) + 1.0), out, zarr_format=3, device="cuda")
    assert np.allclose(zarr.open(str(out), mode="r")[:], data + 1.0)


def test_write_with_cuda_multiple_regions(tmp_path, source, data):
    """Small regions force many H2D/D2H hops and several reader threads, which is
    where a device-context or stream bug shows up rather than in a single region."""
    out = tmp_path / "gpu_regions.zarr"
    io.write(
        (io.read(source) * 2.0), out, zarr_format=3, chunks=CHUNKS,
        region_size_mb=0.01, max_workers=4, device="cuda",
    )
    assert np.allclose(zarr.open(str(out), mode="r")[:], data * 2.0)


def test_cpu_and_cuda_writes_agree(tmp_path, source):
    a, b = tmp_path / "a.zarr", tmp_path / "b.zarr"
    io.write(ops.gaussian_filter(io.read(source), sigma=1.0), a, zarr_format=3)
    io.write(ops.gaussian_filter(io.read(source), sigma=1.0), b,
             zarr_format=3, device="cuda")
    assert np.allclose(
        zarr.open(str(a), mode="r")[:], zarr.open(str(b), mode="r")[:],
        rtol=1e-4, atol=1e-4,
    )


# --------------------------------------------------------------------------- #
# interaction with the optional zarrista backend
# --------------------------------------------------------------------------- #

def test_cuda_compute_over_a_zarrista_source(tmp_path, data):
    """The two optional paths must compose: zarrista reads host memory, the chain
    runs on the GPU, and the result comes back to the host."""
    from dyna_zarr.backends.zarrista_backend import ZARRISTA_AVAILABLE

    if not ZARRISTA_AVAILABLE:
        pytest.skip("zarrista is not installed (optional backend)")

    from dyna_zarr.backends.zarrista_backend import create_v3

    src = tmp_path / "zst.zarr"
    arr = create_v3(src, SHAPE, CHUNKS, "float32")
    arr[:] = data

    got = io.read(src, backend="zarrista").clip(0, 50).compute(device="cuda")
    assert isinstance(got, np.ndarray)
    assert np.allclose(got, np.clip(data, 0, 50))
