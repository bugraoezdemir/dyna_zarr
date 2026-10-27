"""blosc 'snappy': both backends write AND read it, numcodecs cannot.

numcodecs (and so zarr-python, ome-zarr-py, napari-ome-zarr) has shipped blosc
without snappy since 0.9.0. TensorStore and zarrista bundle their own c-blosc with
snappy, so dyna writes it with a LimitedReaderSupportWarning instead of refusing it.
An unknown cname is still refused.
"""
import contextlib
import io as _io
import warnings

import numpy as np
import pytest

import dyna_zarr
from dyna_zarr import Codecs, LimitedReaderSupportWarning
from dyna_zarr.backends.zarrista_backend import ZARRISTA_AVAILABLE
from dyna_zarr.io import io

BACKENDS = ["tensorstore"] + (["zarrista"] if ZARRISTA_AVAILABLE else [])

try:
    from numcodecs.blosc import list_compressors
    NUMCODECS_HAS_SNAPPY = "snappy" in list_compressors()
except Exception:
    NUMCODECS_HAS_SNAPPY = None

DATA = np.arange(48 * 40, dtype="uint16").reshape(48, 40)


def _write(path, codecs, **kw):
    with contextlib.redirect_stdout(_io.StringIO()):
        io.write(dyna_zarr.from_array(DATA), str(path), overwrite=True,
                 compressor=codecs, chunks=(16, 16), **kw)


@pytest.mark.parametrize("writer", BACKENDS)
@pytest.mark.parametrize("zf", [2, 3])
def test_snappy_is_written_with_a_warning_and_reads_back_on_every_backend(
        tmp_path, writer, zf):
    out = tmp_path / "o.zarr"
    if NUMCODECS_HAS_SNAPPY is False:
        with pytest.warns(LimitedReaderSupportWarning, match="LIMITED in OME-Zarr readers"):
            _write(out, Codecs("blosc", cname="snappy"), zarr_format=zf, backend=writer)
    else:
        _write(out, Codecs("blosc", cname="snappy"), zarr_format=zf, backend=writer)
    for reader in BACKENDS:
        back = np.asarray(io.read(str(out), backend=reader))
        assert np.array_equal(back, DATA), (writer, reader)


@pytest.mark.parametrize("cname", ["lz4", "zstd"])
def test_numcodecs_cnames_write_without_the_warning(tmp_path, cname):
    with warnings.catch_warnings():
        warnings.simplefilter("error", LimitedReaderSupportWarning)
        _write(tmp_path / "o.zarr", Codecs("blosc", cname=cname))


def test_an_unknown_cname_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown blosc cname='nope'"):
        _write(tmp_path / "o.zarr", Codecs("blosc", cname="nope"))
