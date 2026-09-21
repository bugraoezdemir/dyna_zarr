"""Tests for the optional zarrista backend.

Three things are load-bearing here and are asserted rather than assumed:

* the **undocumented v2 path** (``Array.from_metadata``) still works and still
  round-trips through zarr-python -- this is the test that catches a future
  zarrista release breaking v2 support;
* the **alignment guard** refuses exactly the misaligned+concurrent combination
  that silently loses data upstream, and allows everything else;
* **numpy basic-indexing semantics** hold through the wrapper, since zarrista
  itself keeps integer-indexed axes and rejects steps.
"""

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from dyna_zarr.backends.zarrista_backend import ZARRISTA_AVAILABLE  # noqa: E402

# The marker makes this file selectable (`-m zarrista`) and, just as importantly,
# de-selectable, so a run without the optional dependency can say so explicitly
# rather than quietly reporting green over ~12% fewer tests. The skipif stays as
# the safety net for anyone who runs the file directly without the extra.
pytestmark = [
    pytest.mark.zarrista,
    pytest.mark.skipif(
        not ZARRISTA_AVAILABLE, reason="zarrista is not installed (optional backend)"
    ),
]

if ZARRISTA_AVAILABLE:
    from dyna_zarr.backends import zarrista_backend as zb
    from dyna_zarr.backends.zarrista_backend import (
        TESTED_VERSION,
        AlignmentError,
        check_write_alignment,
        create_v2,
        create_v3,
        installed_version,
        open_array,
        require_zarrista,
    )

from dyna_zarr.codecs import Codecs  # noqa: E402

SHAPE = (16, 64, 64)
CHUNKS = (4, 32, 32)


@pytest.fixture
def data():
    return np.arange(int(np.prod(SHAPE)), dtype="float32").reshape(SHAPE)


# --------------------------------------------------------------------------- #
# round trips
# --------------------------------------------------------------------------- #

def test_v3_roundtrip(tmp_path, data):
    arr = create_v3(tmp_path / "v3", SHAPE, CHUNKS, "float32")
    arr[:] = data
    assert np.array_equal(open_array(tmp_path / "v3")[:], data)


def test_v3_sharded_roundtrip(tmp_path, data):
    arr = create_v3(tmp_path / "s", SHAPE, CHUNKS, "float32", shard=(8, 64, 64))
    arr[:] = data
    assert arr.is_sharded
    assert arr.write_unit == (8, 64, 64)      # the shard
    assert arr.chunks == CHUNKS               # the inner subchunk
    assert np.array_equal(open_array(tmp_path / "s")[:], data)


def test_v2_roundtrip_through_zarr_python(tmp_path, data):
    """Guards the undocumented from_metadata path: zarr-python must agree."""
    arr = create_v2(tmp_path / "v2", SHAPE, CHUNKS, "float32")
    arr[:] = data

    stored = zarr.open(str(tmp_path / "v2"), mode="r")
    assert stored.metadata.zarr_format == 2
    assert np.array_equal(stored[:], data)
    assert np.array_equal(open_array(tmp_path / "v2")[:], data)


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint8", "uint16", "int32"])
@pytest.mark.parametrize(
    "codecs",
    [Codecs("blosc", cname="lz4"), Codecs("zstd", clevel=3), Codecs(None)],
    ids=["blosc-lz4", "zstd", "none"],
)
def test_v2_dtype_codec_matrix(tmp_path, dtype, codecs):
    d = (np.arange(int(np.prod(SHAPE))) % 251).reshape(SHAPE).astype(dtype)
    path = tmp_path / f"{dtype}_{codecs.compressor}"
    arr = create_v2(path, SHAPE, CHUNKS, dtype, codecs=codecs)
    arr[:] = d
    assert np.array_equal(zarr.open(str(path), mode="r")[:], d)


@pytest.mark.parametrize(
    "codecs",
    [
        Codecs("blosc", cname="lz4"),
        Codecs("blosc", cname="zstd", shuffle=2),
        Codecs("zstd", clevel=3),
        Codecs("gzip", clevel=5),
        Codecs(None),
    ],
    ids=["blosc-lz4", "blosc-zstd-bitshuffle", "zstd", "gzip", "none"],
)
def test_v3_codec_matrix_matches_the_default_backend(tmp_path, data, codecs):
    """Every compressor dyna can emit must survive the zarrista write path.

    Regression test: zstd used to raise TypeError here because zarrista's
    codec.zstd takes (level, checksum) and only level was being passed - a gap the
    blosc-only tests could not see.
    """
    from dyna_zarr import io

    src = tmp_path / "csrc"
    seed = create_v3(src, SHAPE, CHUNKS, "float32")
    seed[:] = data

    out = tmp_path / f"cout_{codecs.compressor}"
    io.write(io.read(src), out, chunks=CHUNKS, zarr_format=3,
             compressor=codecs, backend="zarrista")
    assert np.array_equal(zarr.open(str(out), mode="r")[:], data)


def test_reads_array_written_by_zarr_python(tmp_path, data):
    """Interop the other way: dyna must be able to adopt existing v2 data."""
    z = zarr.open(str(tmp_path / "ext"), mode="w", zarr_format=2, shape=SHAPE,
                  chunks=CHUNKS, dtype="float32")
    z[:] = data
    assert np.array_equal(open_array(tmp_path / "ext")[:], data)


# --------------------------------------------------------------------------- #
# bug 2 containment: numpy indexing semantics
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "key",
    [
        np.s_[:],
        np.s_[3],
        np.s_[-1],
        np.s_[3, 4, 5],
        np.s_[2:10, 0:32, 0:32],
        np.s_[-4:],
        np.s_[::2],
        np.s_[:, ::2, :],
        np.s_[..., 0:8],
        np.s_[0:4],
        np.s_[:8],
        np.s_[0:999],          # out-of-bounds stop clamps, as numpy does
        np.s_[4:4],            # empty span
    ],
)
def test_getitem_matches_numpy(tmp_path, data, key):
    arr = create_v3(tmp_path / "idx", SHAPE, CHUNKS, "float32")
    arr[:] = data
    got, expected = arr[key], data[key]
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_integer_index_drops_axis(tmp_path, data):
    """The specific upstream bug: zarrista alone returns (1, 64, 64) here."""
    arr = create_v3(tmp_path / "sq", SHAPE, CHUNKS, "float32")
    arr[:] = data
    assert arr[3].shape == (64, 64)
    assert arr[3, 4, 5].shape == ()


def test_unsupported_indexing_raises(tmp_path, data):
    arr = create_v3(tmp_path / "bad", SHAPE, CHUNKS, "float32")
    arr[:] = data
    with pytest.raises(NotImplementedError):
        arr[None]
    with pytest.raises(NotImplementedError):
        arr[::-1]
    with pytest.raises(NotImplementedError):
        arr[[0, 2, 4]]
    with pytest.raises(IndexError):
        arr[99]


def test_setitem_partial_and_broadcast(tmp_path, data):
    arr = create_v3(tmp_path / "set", SHAPE, CHUNKS, "float32")
    arr[:] = data
    arr[3] = 7.0                                   # scalar broadcast into a squeezed axis
    assert np.array_equal(arr[3], np.full((64, 64), 7.0, dtype="float32"))
    block = np.full((4, 32, 32), 5.0, dtype="float32")
    arr[0:4, 0:32, 0:32] = block
    assert np.array_equal(arr[0:4, 0:32, 0:32], block)


# --------------------------------------------------------------------------- #
# bug 1 containment: the alignment guard
# --------------------------------------------------------------------------- #

def test_guard_rejects_misaligned_concurrent():
    with pytest.raises(AlignmentError, match="write unit"):
        check_write_alignment((4, 256, 256), (8, 128, 128), threads=8)


def test_guard_allows_aligned_and_serial():
    check_write_alignment((8, 128, 128), (8, 128, 128), threads=8)
    check_write_alignment((16, 256, 256), (8, 128, 128), threads=8)
    check_write_alignment((4, 256, 256), (8, 128, 128), threads=1)  # serial is safe


def test_write_regions_aligned_is_correct(tmp_path):
    shape, chunks = (32, 128, 128), (8, 64, 64)
    d = np.round(np.random.default_rng(0).random(shape, dtype=np.float32) * 100, 1)
    arr = create_v3(tmp_path / "wr", shape, chunks, "float32")
    arr.write_regions(chunks, d, threads=8)
    assert np.array_equal(open_array(tmp_path / "wr")[:], d)


def test_write_regions_misaligned_raises_before_writing(tmp_path):
    shape, chunks = (32, 128, 128), (8, 64, 64)
    d = np.zeros(shape, dtype="float32")
    arr = create_v3(tmp_path / "wr2", shape, chunks, "float32")
    with pytest.raises(AlignmentError):
        arr.write_regions((4, 128, 128), d, threads=8)


def test_write_regions_misaligned_serial_allowed(tmp_path):
    shape, chunks = (32, 128, 128), (8, 64, 64)
    d = np.round(np.random.default_rng(1).random(shape, dtype=np.float32) * 100, 1)
    arr = create_v3(tmp_path / "wr3", shape, chunks, "float32")
    arr.write_regions((4, 128, 128), d, threads=1)
    assert np.array_equal(open_array(tmp_path / "wr3")[:], d)


def test_write_unit_is_shard_when_sharded(tmp_path):
    """The guard must use the SHARD, not the inner chunk, or it under-protects."""
    arr = create_v3(tmp_path / "u", (32, 128, 128), (8, 64, 64), "float32",
                    shard=(16, 128, 128))
    assert arr.write_unit == (16, 128, 128)
    with pytest.raises(AlignmentError):
        check_write_alignment((8, 64, 64), arr.write_unit, threads=8)


# --------------------------------------------------------------------------- #
# public API: io.read/io.write backend= selection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("zarr_format", [2, 3])
def test_io_read_backend_matches_default(tmp_path, data, zarr_format):
    """backend='zarrista' must return exactly what the default backend returns."""
    from dyna_zarr import io

    src = tmp_path / f"src_v{zarr_format}"
    z = zarr.open(str(src), mode="w", zarr_format=zarr_format, shape=SHAPE,
                  chunks=CHUNKS, dtype="float32")
    z[:] = data

    default = io.read(src)
    zst = io.read(src, backend="zarrista")
    assert zst.shape == default.shape
    assert zst.chunks == default.chunks
    assert np.array_equal(zst.compute(), data)
    assert np.array_equal(zst.compute(), default.compute())


def test_io_read_backend_supports_lazy_chain(tmp_path, data):
    """A zarrista-backed array must behave like any other DynamicArray source."""
    from dyna_zarr import io

    src = tmp_path / "chain"
    arr = create_v3(src, SHAPE, CHUNKS, "float32")
    arr[:] = data

    lazy = io.read(src, backend="zarrista")[2:6].astype("float32").clip(0, 100)
    assert np.array_equal(lazy.compute(), np.clip(data[2:6], 0, 100))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"zarr_format": 3},
        {"zarr_format": 2},
        {"zarr_format": 3, "shard_coefficients": (2, 2, 2)},
        {"zarr_format": 3, "region_size_mb": 0.05},   # forces many small regions
    ],
    ids=["v3", "v2", "v3-sharded", "v3-small-regions"],
)
def test_io_write_backend_roundtrip(tmp_path, data, kwargs):
    from dyna_zarr import io

    src = tmp_path / "wsrc"
    seed = create_v3(src, SHAPE, CHUNKS, "float32")
    seed[:] = data

    out = tmp_path / "wout"
    io.write(io.read(src), out, chunks=CHUNKS, backend="zarrista", **kwargs)
    assert np.array_equal(zarr.open(str(out), mode="r")[:], data)


def test_read_and_write_backends_are_independent(tmp_path, data):
    """Mixing is supported on purpose: the write backend is not inherited."""
    from dyna_zarr import io

    src = tmp_path / "mix"
    seed = create_v3(src, SHAPE, CHUNKS, "float32")
    seed[:] = data

    io.write(io.read(src, backend="zarrista"), tmp_path / "a", zarr_format=3)
    io.write(io.read(src), tmp_path / "b", zarr_format=3, backend="zarrista")
    assert np.array_equal(zarr.open(str(tmp_path / "a"), mode="r")[:], data)
    assert np.array_equal(zarr.open(str(tmp_path / "b"), mode="r")[:], data)


def test_sharded_write_regions_never_split_a_shard(tmp_path):
    """The reason io.write grows its region: concurrent writers sharing a shard
    silently lose data upstream. A tiny region_size_mb would normally produce
    sub-shard regions, so this is the case that would corrupt if ungrown."""
    from dyna_zarr import io

    shape, chunks = (32, 128, 128), (8, 64, 64)
    d = np.round(np.random.default_rng(3).random(shape, dtype=np.float32) * 100, 1)
    src = tmp_path / "ssrc"
    seed = create_v3(src, shape, chunks, "float32")
    seed[:] = d

    out = tmp_path / "sout"
    io.write(io.read(src), out, chunks=chunks, zarr_format=3,
             shard_coefficients=(2, 2, 2), region_size_mb=0.05,
             max_workers=8, backend="zarrista")
    assert np.array_equal(zarr.open(str(out), mode="r")[:], d)


def test_unknown_backend_raises(tmp_path, data):
    from dyna_zarr import io

    src = tmp_path / "ub"
    seed = create_v3(src, SHAPE, CHUNKS, "float32")
    seed[:] = data

    with pytest.raises(ValueError, match="unknown backend"):
        io.read(src, backend="nope")
    with pytest.raises(ValueError, match="unknown backend"):
        io.write(io.read(src), tmp_path / "o", backend="nope")


def test_remote_write_goes_through_the_async_obstore_path(tmp_path, data):
    """Object stores are writable too, via AsyncArray + obstore.

    Driven over a file:// URL so the test stays offline and credential-free while
    exercising the same ObjectStore code path an s3:// write would use.
    """
    pytest.importorskip("obstore")
    from dyna_zarr.backends import zarrista_backend as zbk

    dest = tmp_path / "remote_out.zarr"
    dest.mkdir()

    # file:// is served by the SYNC path, so build the async store directly to
    # exercise the ObjectStore code an s3:// write would take.
    from obstore.store import from_url
    from zarrista import codec

    import zarrista as zst

    store = from_url(dest.as_uri())
    grid = zst.ChunkGrid.regular(SHAPE, chunk_shape=CHUNKS)
    builder = zst.ArrayBuilder(
        grid,
        zst.DataType.from_string("float32"),
        zst.FillValue(np.float32(0).tobytes()),
    ).compressors([codec.blosc("lz4", 5, "shuffle", typesize=4)])
    inner = zbk._run_coroutine_fn(
        lambda: builder.create_async(store=store, path="/")
    )
    arr = zbk.ZarristaArray(inner)
    arr[...] = data

    # zarr-python must agree the object store now holds a valid array
    assert np.array_equal(zarr.open(str(dest), mode="r")[:], data)


def test_remote_read_goes_through_the_async_obstore_path(tmp_path, data):
    """Object stores are reachable, just via a different zarrista API.

    A file:// URL is served by the sync path, so this drives _open_remote directly
    to exercise the obstore + AsyncArray code an s3:// read would use. Doing it
    over a local directory keeps the test credential-free and offline.
    """
    pytest.importorskip("obstore")
    from dyna_zarr.backends import zarrista_backend as zbk

    src = tmp_path / "remote.zarr"
    seed = create_v3(src, SHAPE, CHUNKS, "float32")
    seed[:] = data

    arr = zbk.ZarristaArray(zbk._open_remote(src.as_uri()))
    assert arr.shape == SHAPE
    assert np.array_equal(arr[...], data)
    assert np.array_equal(arr[2:6], data[2:6])
    assert arr[3].shape == data[3].shape      # numpy semantics hold here too


def test_storage_options_reach_obstore(monkeypatch, tmp_path, data):
    """storage_options must arrive at obstore.store.from_url unchanged.

    Every non-AWS S3 (MinIO, Ceph, an institutional endpoint) is reachable only
    through these, so silently dropping them would make those stores unusable
    while looking like a zarrista or credentials problem.
    """
    pytest.importorskip("obstore")
    import obstore.store as obs
    from dyna_zarr import io
    from dyna_zarr.backends import zarrista_backend as zbk

    src = tmp_path / "opt.zarr"
    seed = create_v3(src, SHAPE, CHUNKS, "float32")
    seed[:] = data

    seen = {}
    real = obs.from_url

    def spy(url, **kwargs):
        seen["url"], seen["kwargs"] = url, kwargs
        # redirect the "remote" store at the real local directory
        return real(src.as_uri())

    monkeypatch.setattr(obs, "from_url", spy)

    opts = {"endpoint": "https://s3.example.org", "skip_signature": True,
            "virtual_hosted_style_request": False}
    got = io.read("s3://bucket/opt.zarr", backend="zarrista", storage_options=opts)

    assert seen["url"] == "s3://bucket/opt.zarr"
    assert seen["kwargs"] == opts
    assert np.array_equal(got.compute(), data)


def test_bad_storage_options_raise_a_useful_error(monkeypatch, tmp_path):
    pytest.importorskip("obstore")
    from dyna_zarr import io

    with pytest.raises(ValueError, match="obstore rejected these storage_options"):
        io.read("s3://bucket/x.zarr", backend="zarrista",
                storage_options={"definitely_not_a_real_option": 1})


@pytest.mark.parametrize(
    "url,remote",
    [
        ("/local/path", False),
        ("C:/drive/path", False),
        ("file:///tmp/a.zarr", False),
        ("s3://bucket/key", True),
        ("gs://bucket/key", True),
        ("http://host/a.zarr", True),
    ],
)
def test_remote_detection(url, remote):
    from dyna_zarr.backends.zarrista_backend import _is_remote

    assert _is_remote(url) is remote


# --------------------------------------------------------------------------- #
# version pinning
# --------------------------------------------------------------------------- #

def test_installed_version_matches_the_tested_pin():
    """If this fails, the guards in this backend are running unverified.

    Re-run the whole file against the new zarrista, re-check the two upstream bugs
    in reports/zarrista_bugs/, then move TESTED_VERSION and the pyproject pin
    together -- do not just bump one.
    """
    assert installed_version() == TESTED_VERSION


def test_pyproject_pin_agrees_with_tested_version():
    """The declared extra and the runtime constant must not drift apart.

    Skipped when pyproject.toml is not reachable: the install-check job copies the
    tests out of the repo on purpose, and there the pin is verified from the
    installed distribution's metadata instead (see the test below).
    """
    import re
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.is_file():
        pytest.skip("pyproject.toml not reachable (tests running outside the repo)")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11 is unsupported anyway
        pytest.skip("tomllib unavailable")

    with pyproject.open("rb") as fh:
        extra = tomllib.load(fh)["project"]["optional-dependencies"]["zarrista"]

    pins = [r for r in extra if re.match(r"^zarrista\b", r)]
    assert pins, "the [zarrista] extra no longer declares zarrista"
    match = re.match(r"^zarrista==([0-9][^;\s]*)$", pins[0])
    assert match, f"the [zarrista] extra is no longer an exact pin: {pins[0]!r}"
    assert match.group(1) == TESTED_VERSION


def test_distribution_metadata_pin_agrees_with_tested_version():
    """Same check against the INSTALLED metadata, so it also holds for a wheel.

    This is the one that survives being copied out of the repo, and it checks what
    a real `pip install "dyna-zarr[zarrista]"` would actually resolve.
    """
    import re
    from importlib.metadata import distribution

    reqs = distribution("dyna_zarr").requires or []
    pins = [r for r in reqs if r.startswith("zarrista")]
    assert pins, "the installed distribution declares no zarrista requirement"
    match = re.search(r"zarrista==([0-9][^;\s]*)", pins[0])
    assert match, f"the zarrista requirement is not an exact pin: {pins[0]!r}"
    assert match.group(1) == TESTED_VERSION


def test_version_mismatch_warns(monkeypatch):
    monkeypatch.setattr(zb, "_version_warned", False)
    monkeypatch.setattr(zb, "installed_version", lambda: "9.9.9")
    with pytest.warns(RuntimeWarning, match="verified against"):
        require_zarrista()


def test_matching_version_does_not_warn(monkeypatch):
    monkeypatch.setattr(zb, "_version_warned", False)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        require_zarrista()


# --------------------------------------------------------------------------- #
# integration with DynamicArray
# --------------------------------------------------------------------------- #

def test_dynamic_array_accepts_zarrista_source(tmp_path, data):
    from dyna_zarr import DynamicArray

    arr = create_v3(tmp_path / "dyn", SHAPE, CHUNKS, "float32")
    arr[:] = data
    dyn = DynamicArray(arr)
    assert dyn.shape == SHAPE
    assert dyn.chunks == CHUNKS
    assert np.array_equal(dyn.compute(), data)
    assert np.array_equal(dyn[2:10].compute(), data[2:10])
