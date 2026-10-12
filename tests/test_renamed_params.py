"""max_workers -> max_threads (0.1.0): the old name still works, with a FutureWarning."""
import numpy as np
import pytest

import dyna_zarr
from dyna_zarr import rechunk as rc
from dyna_zarr.io import io


@pytest.fixture
def arr():
    return dyna_zarr.from_array(np.arange(16 * 16 * 16, dtype="uint16").reshape(16, 16, 16))


def test_old_name_warns_and_still_sets_the_threads(arr, tmp_path, capsys):
    with pytest.warns(FutureWarning, match="max_workers= is deprecated.*max_threads="):
        io.write(arr, str(tmp_path / "a.zarr"), max_workers=3, zarr_format=2,
                 region_shape=(4, 16, 16), chunks=(4, 16, 16))
    assert "3 threads" in capsys.readouterr().out
    np.testing.assert_array_equal(io.read(str(tmp_path / "a.zarr"))[...].compute(), arr[...].compute())


def test_positional_slot_is_max_threads(arr, tmp_path, capsys):
    io.write(arr, str(tmp_path / "b.zarr"), 2, zarr_format=2, region_shape=(4, 16, 16),
             chunks=(4, 16, 16))
    assert "2 threads" in capsys.readouterr().out


@pytest.mark.parametrize("call", [
    lambda a, p: io.write(a, p, max_threads=2, max_workers=2),
    lambda a, p: io.write_levels(a, [dict(path=p, factors=(1, 1, 1))], max_threads=2,
                                 max_workers=2),
    lambda a, p: rc.rechunk(a, (8, 8, 8), p, max_threads=2, max_workers=2),
])
def test_both_names_is_an_error(arr, tmp_path, call):
    with pytest.raises(TypeError, match="pass max_threads= only"):
        call(arr, str(tmp_path / "c.zarr"))


def test_write_levels_and_rechunk_accept_the_old_name(arr, tmp_path):
    with pytest.warns(FutureWarning, match="write_levels: max_workers"):
        io.write_levels(arr, [dict(path=str(tmp_path / "L0.zarr"), factors=(1, 1, 1))],
                        max_workers=2, zarr_format=2)
    with pytest.warns(FutureWarning, match="rechunk: max_workers"):
        rc.rechunk(arr, (8, 8, 8), str(tmp_path / "r.zarr"), max_workers=2)
