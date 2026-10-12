"""io.capture_messages: io's messages collected per context, never via sys.stdout.

A caller that wants dyna's chunk/region notes in its own log used to redirect_stdout,
which swaps sys.stdout for the WHOLE process: concurrent writes (threads) then swallowed
each other's output and the host's (a GUI log), and restoring out of order could leave
sys.stdout on a dead buffer.
"""
import sys
import threading

import numpy as np

from dyna_zarr import operations as ops
from dyna_zarr.io import io


def _arr(seed):
    return ops.random((16, 64, 64), chunks=(8, 32, 32), seed=seed)


def test_messages_are_collected_not_printed(tmp_path, capsys):
    with io.capture_messages() as messages:
        io.write(_arr(0), str(tmp_path / "a.zarr"))
    assert any("Output chunks" in m for m in messages)
    assert capsys.readouterr().out == ""


def test_without_a_capture_they_are_printed(tmp_path, capsys):
    io.write(_arr(0), str(tmp_path / "a.zarr"))
    assert "Output chunks" in capsys.readouterr().out


def test_messages_from_dyna_worker_threads_are_captured(tmp_path, capsys):
    # the pipeline engine reports from its own reader/writer/monitor threads
    with io.capture_messages() as messages:
        io.write(_arr(0), str(tmp_path / "a.zarr"), engine="pipeline")
        io.write_levels(_arr(0), [dict(path=str(tmp_path / "l0.zarr"), factors=(1, 1, 1))])
    assert any(m.startswith("[Levels]") for m in messages)
    assert capsys.readouterr().out == ""


def test_concurrent_writes_keep_their_own_messages(tmp_path):
    stdout_before = sys.stdout
    results, errors = {}, []

    def job(i):
        try:
            with io.capture_messages() as messages:
                io.write(_arr(i), str(tmp_path / f"{i}.zarr"), engine="pipeline")
            results[i] = messages
        except BaseException as exc:
            errors.append(exc)
    threads = [threading.Thread(target=job, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert sys.stdout is stdout_before
    for i, messages in results.items():
        done = [m for m in messages if "Successfully wrote" in m]
        assert done == [f"Successfully wrote array to {tmp_path / f'{i}.zarr'}"], done
