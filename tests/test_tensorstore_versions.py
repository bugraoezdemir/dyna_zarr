"""tensorstore releases dyna-zarr refuses: the pins in pyproject, and the warning for an
install that bypassed them (below 0.1.86 a write can wait forever when several run at
once; 0.1.85 also keeps memory while lower levels are built)."""
import sys
import tomllib
import warnings
from pathlib import Path

import pytest
from packaging.requirements import Requirement

import dyna_zarr  # noqa: F401  (imports the io module)

# the module itself: as an attribute, dyna_zarr.io is the io namespace class
io_module = sys.modules["dyna_zarr.io"]


def _tensorstore_requirement():
    """tensorstore's requirement: from pyproject.toml in a source checkout, else from the
    installed package's metadata (the wheel and dependency-floor jobs copy the tests away
    from the checkout, and there the installed pins are the ones that matter)."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if pyproject.is_file():
        deps = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
    else:
        from importlib.metadata import requires
        deps = requires("dyna-zarr") or []
    found = [Requirement(d) for d in deps if Requirement(d).name == "tensorstore"
             and Requirement(d).marker is None]
    assert found, "dyna-zarr declares no tensorstore requirement"
    return found[0]


@pytest.mark.parametrize("version", ["0.1.62", "0.1.76", "0.1.77", "0.1.81", "0.1.85"])
def test_pins_refuse_hanging_crashing_and_leaking_releases(version):
    assert not _tensorstore_requirement().specifier.contains(version)


@pytest.mark.parametrize("version", ["0.1.86", "0.1.87", "0.2.0"])
def test_pins_allow_good_releases(version):
    assert _tensorstore_requirement().specifier.contains(version)


def test_the_warning_floor_is_the_pin():
    floor = ".".join(str(n) for n in io_module._TENSORSTORE_FLOOR)
    req = _tensorstore_requirement().specifier
    assert req.contains(floor)
    below = io_module._TENSORSTORE_FLOOR[:2] + (io_module._TENSORSTORE_FLOOR[2] - 1,)
    assert not req.contains(".".join(str(n) for n in below))


@pytest.mark.parametrize("version", ["0.1.62", "0.1.76", "0.1.81"])
def test_old_release_warns_of_the_hang_with_the_way_out(version):
    with pytest.warns(UserWarning, match=rf"tensorstore {version} can leave a write waiting "
                                         r"forever.*tensorstore>=0\.1\.86") as caught:
        assert io_module._warn_old_tensorstore(version) is True
    assert "keeps memory" not in str(caught[0].message)


def test_leaking_release_warns_of_the_hang_and_the_leak():
    with pytest.warns(UserWarning, match=r"tensorstore 0\.1\.85 can leave a write waiting "
                                         r"forever.*keeps memory.*tensorstore>=0\.1\.86"):
        assert io_module._warn_old_tensorstore("0.1.85") is True


def test_installed_good_and_unknown_releases_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for version in ("0.1.86", "0.1.86.post1", "0.2.0", None, "unknown"):
            assert io_module._warn_old_tensorstore(version) is False
        installed = io_module._tensorstore_version()
        assert installed                                   # read from the package metadata
        if io_module._version_tuple(installed) >= io_module._TENSORSTORE_FLOOR:
            assert io_module._warn_old_tensorstore(installed) is False
