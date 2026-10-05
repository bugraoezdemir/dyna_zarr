"""Remote I/O is consistent across backends.

Before: the default TensorStore writer opened its output on the local `file` store with
the URL as a PATH (so it could not write remotely at all), and TensorStore reads silently
IGNORED storage_options (a non-AWS endpoint was unreachable). Now both backends read and
write s3:// with the SAME storage_options names, refuse what they cannot do instead of
ignoring it, and report the same format/codecs/shards/chunks for the same array whether
it is local or remote and whichever backend read it.

S3 tests run against an S3 emulator: DYNA_TEST_S3_ENDPOINT if set, else an in-process
moto server (the [dev] extra installs moto), else they skip.
"""
import contextlib
import io as _stdio
import os
import socket
import urllib.request
import uuid

import numpy as np
import pytest

from dyna_zarr import Codecs, DynamicArray, io, operations as ops
from dyna_zarr.backends.zarrista_backend import UnsupportedByBackend
from dyna_zarr.io import _ts_kvstore


def _write(*args, **kwargs):
    with contextlib.redirect_stdout(_stdio.StringIO()):
        return io.write(*args, **kwargs)


@pytest.fixture(scope="session")
def s3_endpoint():
    """An S3-compatible endpoint URL with test credentials in the environment."""
    saved = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                                            "AWS_REGION", "AWS_DEFAULT_REGION")}
    os.environ.update(AWS_ACCESS_KEY_ID="test", AWS_SECRET_ACCESS_KEY="test",
                      AWS_REGION="us-east-1", AWS_DEFAULT_REGION="us-east-1")
    server = None
    endpoint = os.environ.get("DYNA_TEST_S3_ENDPOINT")
    if not endpoint:
        server_mod = pytest.importorskip("moto.server", reason="needs moto[server] or "
                                         "DYNA_TEST_S3_ENDPOINT for S3 tests")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = server_mod.ThreadedMotoServer(ip_address="127.0.0.1", port=port)
        server.start()
        endpoint = f"http://127.0.0.1:{port}"
    yield endpoint
    if server is not None:
        server.stop()
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def bucket(s3_endpoint):
    name = f"dz-{uuid.uuid4().hex[:12]}"
    urllib.request.urlopen(urllib.request.Request(f"{s3_endpoint}/{name}", method="PUT")).read()
    opts = {"endpoint": s3_endpoint, "region": "us-east-1",
            "virtual_hosted_style_request": False, "client_options": {"allow_http": True}}
    return name, opts


X = np.arange(8 * 16 * 16, dtype="f4").reshape(8, 16, 16)
BACKENDS = ["tensorstore", pytest.param("zarrista", marks=pytest.mark.zarrista)]


# --------------------------------------------------------------------------- #
# writes and reads through S3
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("staged", [False, True], ids=["region", "staged-flatten"])
def test_write_then_read_s3(bucket, backend, staged):
    name, opts = bucket
    url = f"s3://{name}/{backend}/out.zarr"
    lazy, ref = (ops.flatten(DynamicArray(X)), X.ravel()) if staged else (DynamicArray(X) * 2, X * 2)
    _write(lazy, url, backend=backend, storage_options=opts)
    back = io.read(url, backend=backend, storage_options=opts)
    np.testing.assert_array_equal(back.compute(), ref)


@pytest.mark.zarrista
@pytest.mark.parametrize("writer, reader", [("tensorstore", "zarrista"), ("zarrista", "tensorstore")])
def test_each_backend_reads_what_the_other_wrote(bucket, writer, reader):
    name, opts = bucket
    url = f"s3://{name}/x.zarr"
    _write(DynamicArray(X), url, backend=writer, storage_options=opts)
    np.testing.assert_array_equal(io.read(url, backend=reader, storage_options=opts).compute(), X)


VARIANTS = {
    "v2-zstd4": dict(zarr_format=2, compressor=Codecs("zstd", clevel=4), chunks=(4, 8, 8)),
    "v3-blosc-bitshuffle": dict(zarr_format=3, compressor=Codecs("blosc", cname="zstd", clevel=3, shuffle=2),
                                chunks=(4, 8, 8)),
    "v3-sharded-gzip": dict(zarr_format=3, compressor=Codecs("gzip", clevel=2), chunks=(4, 8, 8),
                            shard_coefficients=(2, 2, 2)),
}


def _settings(arr):
    c = arr.codecs
    return (arr.zarr_format, None if c is None else (c.compressor, c.clevel),
            arr.shards, arr.chunks)


@pytest.mark.zarrista
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_same_settings_from_every_backend_and_location(tmp_path, bucket, variant):
    # Remote zarrista reads used to report no format; remote TensorStore reads no chunk
    # grid, codecs or shards - so the same array re-wrote differently by where it came from.
    name, opts = bucket
    kw = VARIANTS[variant]
    local, remote = str(tmp_path / "a.zarr"), f"s3://{name}/a.zarr"
    _write(DynamicArray(X), local, **kw)
    _write(DynamicArray(X), remote, storage_options=opts, **kw)
    seen = {f"{loc}/{b}": _settings(io.read(url, backend=b, storage_options=so))
            for loc, url, so in (("local", local, None), ("s3", remote, opts))
            for b in ("tensorstore", "zarrista")}
    assert len(set(seen.values())) == 1, seen
    fmt, codecs, shards, chunks = next(iter(seen.values()))
    assert fmt == kw["zarr_format"] and chunks == (4, 8, 8)
    assert codecs == (kw["compressor"].compressor, kw["compressor"].clevel)
    assert shards == ((8, 16, 16) if "shard_coefficients" in kw else None)


# --------------------------------------------------------------------------- #
# what TensorStore cannot do is refused, never ignored
# --------------------------------------------------------------------------- #

def test_storage_options_on_a_local_path_are_refused(tmp_path):
    local = str(tmp_path / "a.zarr")
    _write(DynamicArray(X), local)
    with pytest.raises(UnsupportedByBackend, match="local path"):
        io.read(local, storage_options={"region": "x"})
    with pytest.raises(UnsupportedByBackend, match="local path"):
        _write(DynamicArray(X), str(tmp_path / "b.zarr"), storage_options={"region": "x"})
    assert not (tmp_path / "b.zarr").exists()


@pytest.mark.parametrize("url, opts, match", [
    ("https://example.org/out.zarr", None, "read-only"),                 # TS http store
    ("az://container/out.zarr", None, "no store for this scheme"),
    ("s3://bucket-name/out.zarr", {"access_key_id": "k"}, "not supported"),   # inline creds
    ("s3://bucket-name/out.zarr", {"endpoint": "http://h", "virtual_hosted_style_request": True},
     "path-style"),
    ("s3://bucket-name/out.zarr", {"client_options": {"timeout": "5s"}}, "not supported"),
    ("gs://bucket-name/out.zarr", {"endpoint": "http://h"}, "not supported"),
])
def test_tensorstore_refuses_what_it_cannot_do(tmp_path, monkeypatch, url, opts, match):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(UnsupportedByBackend, match=match):
        _write(DynamicArray(X), url, storage_options=opts)
    assert os.listdir(tmp_path) == []                     # nothing written locally


def test_kvstore_mapping():
    kv = _ts_kvstore("s3://my-bucket/a/b.zarr", {"endpoint": "http://h:9", "region": "eu-1",
                                                 "virtual_hosted_style_request": False,
                                                 "client_options": {"allow_http": True}})
    assert kv == {"driver": "s3", "bucket": "my-bucket", "path": "a/b.zarr/",
                  "endpoint": "http://h:9", "aws_region": "eu-1"}
    assert _ts_kvstore("gs://my-bucket/p") == {"driver": "gcs", "bucket": "my-bucket", "path": "p/"}
    assert _ts_kvstore("https://h/x.zarr") == {"driver": "http", "base_url": "https://h/x.zarr/"}
    from dyna_zarr.io import _TS_ANONYMOUS_S3_MIN, _ts_version
    if _ts_version() >= _TS_ANONYMOUS_S3_MIN:
        assert _ts_kvstore("s3://my-bucket/p", {"skip_signature": True})["aws_credentials"] == \
            {"type": "anonymous"}
    else:
        with pytest.raises(UnsupportedByBackend, match="needs tensorstore"):
            _ts_kvstore("s3://my-bucket/p", {"skip_signature": True})
