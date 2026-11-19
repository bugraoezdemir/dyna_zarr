"""dyna-zarr's native engine (dyna-zarr-core, Workplan WP15 / DR1): finding it, the
switch, and the one function the first version has (array_info).

The switch tests run everywhere. The tests marked ``engine`` need dyna-zarr-core
installed; CI installs the wheels it builds and fails if the engine is missing, so
they cannot skip there unnoticed."""
import numpy as np
import pytest
import zarr

from dyna_zarr import _engine


@pytest.fixture(autouse=True)
def fresh_switch(monkeypatch):
    """Every test starts with no choice made and no environment override, and the
    module's state is restored afterwards."""
    monkeypatch.delenv(_engine.ENV, raising=False)
    saved = (_engine._module, _engine._problem, _engine._looked, _engine._choice)
    _engine._choice = None
    yield
    _engine._module, _engine._problem, _engine._looked, _engine._choice = saved


def _pretend_missing(monkeypatch):
    monkeypatch.setattr(_engine, "_looked", True)
    monkeypatch.setattr(_engine, "_module", None)
    monkeypatch.setattr(_engine, "_problem", "dyna-zarr-core is not installed (test)")


def test_without_the_engine_dyna_quietly_uses_the_python_path(monkeypatch):
    _pretend_missing(monkeypatch)
    assert not _engine.available()
    assert not _engine.enabled()
    assert _engine.engine() is None


@pytest.mark.parametrize("value", ["1", "on", "TRUE"])
def test_requiring_a_missing_engine_says_how_to_install_it(monkeypatch, value):
    _pretend_missing(monkeypatch)
    monkeypatch.setenv(_engine.ENV, value)
    with pytest.raises(_engine.EngineUnavailableError, match="pip install"):
        _engine.enabled()
    monkeypatch.delenv(_engine.ENV)
    with pytest.raises(_engine.EngineUnavailableError, match="dyna-zarr-core"):
        _engine.set_engine(True)


@pytest.mark.parametrize("value", ["0", "off", "False"])
def test_the_environment_switches_the_engine_off(monkeypatch, value):
    monkeypatch.setenv(_engine.ENV, value)
    assert not _engine.enabled()
    assert _engine.engine() is None


def test_an_unknown_environment_value_is_an_error(monkeypatch):
    monkeypatch.setenv(_engine.ENV, "maybe")
    with pytest.raises(ValueError, match=_engine.ENV):
        _engine.enabled()


def test_set_engine_wins_over_the_environment_and_none_forgets_it(monkeypatch):
    monkeypatch.setenv(_engine.ENV, "1")
    _engine.set_engine(False)
    assert not _engine.enabled()
    _pretend_missing(monkeypatch)
    _engine.set_engine(None)            # back to the environment: required but missing
    with pytest.raises(_engine.EngineUnavailableError):
        _engine.enabled()


def test_an_engine_outside_the_supported_range_is_not_used(monkeypatch):
    class Old:
        __version__ = "0.1.0"
    monkeypatch.setattr(_engine.importlib, "import_module", lambda name: Old)
    monkeypatch.setattr(_engine, "_looked", False)
    monkeypatch.setattr(_engine, "_module", None)
    assert not _engine.available()
    assert "0.1.0" in _engine._problem
    monkeypatch.setenv(_engine.ENV, "1")
    with pytest.raises(_engine.EngineUnavailableError, match="0.1.0"):
        _engine.enabled()


def test_threads_that_first_use_the_engine_at_once_all_find_it(monkeypatch):
    """eubi converts several scenes in parallel, so the engine's first use can come
    from several threads together. Each must wait for the one import and see its
    outcome; one used to see no engine and raise "required, but None"."""
    import threading
    import time

    class Engine:
        __version__ = "0.0.1"
    imports = []

    def slow_import(name):
        imports.append(name)
        time.sleep(0.2)                 # the others arrive while this one imports
        return Engine
    monkeypatch.setattr(_engine.importlib, "import_module", slow_import)
    monkeypatch.setattr(_engine, "_looked", False)
    monkeypatch.setattr(_engine, "_module", None)
    monkeypatch.setattr(_engine, "_problem", None)
    monkeypatch.setenv(_engine.ENV, "1")

    start = threading.Barrier(8)
    seen = []

    def first_use():
        start.wait()
        try:
            seen.append(_engine.engine())
        except Exception as e:          # noqa: BLE001 - the test reports what went wrong
            seen.append(e)
    threads = [threading.Thread(target=first_use) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == [Engine] * 8
    assert imports == ["dyna_zarr_core"]


@pytest.mark.parametrize("version, release", [("0.0.1", (0, 0, 1)), ("0.1.0rc1", (0, 1, 0)),
                                              ("1.2", (1, 2, 0)), ("0.0.2+local", (0, 0, 2))])
def test_release_numbers(version, release):
    assert _engine._release(version) == release


# ------------------------------------------------------------------ with the engine

@pytest.mark.engine
def test_the_engine_is_installed_and_used_by_default():
    core = pytest.importorskip("dyna_zarr_core")
    assert _engine.available()
    assert _engine.enabled()
    assert _engine.engine() is core
    assert _engine.version() == core.__version__


@pytest.mark.engine
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "float32", "float64"])
def test_array_info_matches_zarr_python(tmp_path, zarr_format, dtype):
    core = pytest.importorskip("dyna_zarr_core")
    path = tmp_path / f"a{zarr_format}.zarr"
    z = zarr.create_array(str(path), shape=(3, 50, 70), chunks=(1, 16, 32), dtype=dtype,
                          zarr_format=zarr_format)
    z[:] = np.arange(z.size, dtype=dtype).reshape(z.shape)
    info = core.array_info(str(path))
    assert info["shape"] == [3, 50, 70]
    assert info["chunks"] == [1, 16, 32]
    assert info["zarr_format"] == zarr_format
    assert info["dtype"] == np.dtype(dtype).name


@pytest.mark.engine
def test_array_info_on_a_folder_without_an_array_names_it(tmp_path):
    core = pytest.importorskip("dyna_zarr_core")
    with pytest.raises(ValueError, match="not a Zarr array"):
        core.array_info(str(tmp_path))
