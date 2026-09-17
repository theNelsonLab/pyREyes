"""
The subprocess stall detector.

A hung `xds_par` can block the monitor indefinitely. Two things are tested:

  1. the call has a timeout;
  2. the timeout kills the whole PROCESS GROUP. subprocess.run's own `timeout=`
     kills only the direct child, and xds_par spawns workers that would
     otherwise keep running.
"""

import os
import subprocess
import time
from subprocess import TimeoutExpired

import pytest

import pyREyes.REyes as reyes
from pyREyes.REyes import (
    DEFAULT_SUBPROCESS_TIMEOUT,
    REyesProcessor,
    run_with_timeout,
)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OSError):
        return False
    return True


def test_returns_output_like_subprocess_run():
    result = run_with_timeout(['sh', '-c', 'echo out; echo err >&2; exit 3'],
                              timeout=30)

    assert result.returncode == 3
    assert result.stdout.strip() == 'out'
    assert result.stderr.strip() == 'err'


def test_completes_normally_well_inside_the_timeout():
    assert run_with_timeout(['sh', '-c', 'exit 0'], timeout=30).returncode == 0


def test_raises_on_expiry():
    with pytest.raises(TimeoutExpired):
        run_with_timeout(['sh', '-c', 'sleep 30'], timeout=0.5)


def test_expiry_kills_the_whole_process_group(tmp_path):
    """The grandchild must die too, not just the shell we launched."""
    pidfile = tmp_path / 'grandchild.pid'
    script = f'sleep 30 & echo $! > {pidfile}; wait'

    with pytest.raises(TimeoutExpired):
        run_with_timeout(['sh', '-c', script], timeout=1.0)

    grandchild = int(pidfile.read_text().strip())

    deadline = time.time() + 5.0
    while time.time() < deadline and pid_alive(grandchild):
        time.sleep(0.05)

    assert not pid_alive(grandchild), (
        f'grandchild {grandchild} survived the timeout -- the process group '
        f'was not killed'
    )


def test_processor_defaults_to_the_stall_detector(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processor = REyesProcessor(working_directory=str(tmp_path))

    assert processor.subprocess_timeout == DEFAULT_SUBPROCESS_TIMEOUT
    # AutoSolve can legitimately run for hours, so it has no default limit.
    assert processor.autosolve_timeout is None


def test_run_module_reports_a_timeout_as_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    processor = REyesProcessor(working_directory=str(tmp_path))

    def always_times_out(cmd, cwd=None, timeout=None):
        raise TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(reyes, 'run_with_timeout', always_times_out)

    # Same contract as a non-zero exit: log it, report failure, do not retry.
    assert processor.run_module('eucentricity', {}) is False


def test_group_kill_without_process_groups_kills_the_child(monkeypatch):
    """Where os.killpg is unavailable (Windows), the process tree is killed
    with taskkill and the direct child is killed as well."""
    process = subprocess.Popen(['sh', '-c', 'sleep 30'])
    taskkill_calls = []

    monkeypatch.delattr(reyes.os, 'killpg', raising=False)
    monkeypatch.delattr(reyes.os, 'getpgid', raising=False)
    monkeypatch.setattr(reyes.os, 'name', 'nt')
    monkeypatch.setattr(reyes, '_kill_process_tree_windows', taskkill_calls.append)
    try:
        reyes._kill_process_group(process)
        returncode = process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert taskkill_calls == [process.pid]
    assert returncode is not None


def test_group_kill_tolerates_a_failing_taskkill(monkeypatch):
    process = subprocess.Popen(['sh', '-c', 'sleep 30'])

    def taskkill_missing(pid):
        raise FileNotFoundError('taskkill')

    monkeypatch.delattr(reyes.os, 'killpg', raising=False)
    monkeypatch.delattr(reyes.os, 'getpgid', raising=False)
    monkeypatch.setattr(reyes.os, 'name', 'nt')
    monkeypatch.setattr(reyes, '_kill_process_tree_windows', taskkill_missing)
    try:
        reyes._kill_process_group(process)
        returncode = process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert returncode is not None
