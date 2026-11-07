"""tensorstore releases dyna-zarr refuses: the pins in pyproject, and the warning for an
install that bypassed them (0.1.85 keeps memory while lower levels are built)."""
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


@pytest.mark.parametrize("version", ["0.1.77", "0.1.84", "0.1.85"])
def test_pins_refuse_crashing_and_leaking_releases(version):
    assert not _tensorstore_requirement().specifier.contains(version)


@pytest.mark.parametrize("version", ["0.1.62", "0.1.76", "0.1.86"])
def test_pins_allow_good_releases(version):
    assert _tensorstore_requirement().specifier.contains(version)


def test_leaking_release_warns_with_the_way_out():
    with pytest.warns(UserWarning, match=r'tensorstore 0\.1\.85 .*tensorstore>=0\.1\.86'):
        assert io_module._warn_leaking_tensorstore("0.1.85") is True


def test_installed_and_good_releases_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert io_module._warn_leaking_tensorstore("0.1.86") is False
        assert io_module._warn_leaking_tensorstore(None) is False
        installed = io_module._tensorstore_version()
        assert installed                                   # read from the package metadata
        if installed not in io_module._LEAKING_TENSORSTORE:
            assert io_module._warn_leaking_tensorstore(installed) is False
