"""write_levels' transaction mode with a planar source: each write unit is filled over many
blocks and committed every region_size_mb, so one unit has several transactions. When the
unit's last piece went into a transaction closed earlier (a thread still writing into it)
while a newer one was open, the newer one was never committed and its pieces were lost:
about 1 run in 60 (dyna-zarr 0.1.2 and earlier). Timing-dependent, so written many times."""
import numpy as np
import pytest

from dyna_zarr import from_array, io

DATA = (np.random.default_rng(1).integers(0, 60, (24, 96, 128))
        + np.arange(128)).astype("uint16")
OUTPUTS = [dict(path="0", factors=(1, 1, 1), chunks=(24, 48, 64)),
           dict(path="1", factors=(1, 2, 2), chunks=(24, 48, 64))]
KW = dict(zarr_format=2, max_threads=4, region_size_mb=0.05)


def test_every_piece_of_a_unit_committed_several_times_is_kept(tmp_path):
    for run in range(200):
        out = tmp_path / f"r{run}.zarr"
        with io.capture_messages() as messages:
            io.write_levels(from_array(DATA, chunks=(1, 96, 128)),
                            [dict(o, path=str(out / o["path"])) for o in OUTPUTS], **KW)
        assert "written every" in "\n".join(messages)      # the planar transaction mode
        np.testing.assert_array_equal(io.read(str(out / "0")).compute(), DATA,
                                      err_msg=f"run {run}")
        np.testing.assert_array_equal(io.read(str(out / "1")).compute(),
                                      DATA[:, ::2, ::2], err_msg=f"run {run}")


def test_the_same_into_a_zip(tmp_path):
    pytest.importorskip("zarr_zip")
    for run in range(60):
        zip_path = tmp_path / f"r{run}.ome.zarr.zip"
        with io.open_zip(zip_path) as z:
            with io.capture_messages():
                io.write_levels(from_array(DATA, chunks=(1, 96, 128)),
                                [dict(o, path=z / o["path"]) for o in OUTPUTS], **KW)
        np.testing.assert_array_equal(io.read(str(zip_path / "0")).compute(), DATA,
                                      err_msg=f"run {run}")
