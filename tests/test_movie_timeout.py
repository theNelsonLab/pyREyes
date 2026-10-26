"""The per-movie autoprocess limit (--movie-timeout).

A movie that takes far longer than the others is usually a bad one, and the
session should not wait on it. The limit is a separate knob from
--subprocess-timeout, defaulting to the same value (as in 3.4.10 and earlier),
and a movie over it is abandoned: flagged as finished with, so it is not
offered again, and the session moves on.
"""

import os
from subprocess import TimeoutExpired

import pytest

from pyREyes.lib import REyes_autoproc as ap


class Result:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def flagged(movie_dir):
    return os.path.isfile(os.path.join(movie_dir, ap.AUTOPROCESS_FLAG))


def movie_dir(tmp_path, name='mov1', artefact=None):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    if artefact:
        (d / artefact).write_text('')
    return str(d)


# --------------------------------------------------------------------------- #
# The two limits are genuinely separate knobs
# --------------------------------------------------------------------------- #
def processor(tmp_path, monkeypatch, **kwargs):
    from pyREyes.REyes import REyesProcessor
    monkeypatch.chdir(tmp_path)          # REyes_logs/ is created in the CWD
    return REyesProcessor(working_directory=str(tmp_path), **kwargs)


def test_the_movie_limit_defaults_to_the_subprocess_timeout(tmp_path, monkeypatch):
    """3.4.10 gave autoprocess the --subprocess-timeout; so does the default."""
    from pyREyes.REyes import DEFAULT_SUBPROCESS_TIMEOUT

    assert processor(tmp_path, monkeypatch).movie_timeout == DEFAULT_SUBPROCESS_TIMEOUT
    assert processor(tmp_path, monkeypatch,
                     subprocess_timeout=7200).movie_timeout == 7200


def test_the_movie_limit_is_a_separate_knob(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch, subprocess_timeout=7200, movie_timeout=600)

    assert (p.subprocess_timeout, p.movie_timeout) == (7200, 600)


def test_the_cli_default_keeps_the_subprocess_timeout(monkeypatch, tmp_path):
    import pyREyes.REyes as reyes

    captured = {}

    class Stop(Exception):
        pass

    def fake_processor(**kwargs):
        captured.update(kwargs)
        raise Stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reyes, 'REyesProcessor', fake_processor)
    monkeypatch.setattr(reyes.sys, 'argv', ['reyes-monitor'])
    with pytest.raises(Stop):
        reyes.main()
    assert captured['movie_timeout'] is None
    assert captured['idle_timeout'] == 0


def test_the_monitor_passes_its_own_limit_to_autoprocess(tmp_path, monkeypatch):
    seen = {}

    def runner(argv, cwd, timeout):
        seen['timeout'] = timeout
        return Result(0)

    ap.run_for_movie(['autoprocess'], movie_dir(tmp_path, artefact='XDS.INP'),
                     600, runner)

    assert seen['timeout'] == 600


# --------------------------------------------------------------------------- #
# A movie over the limit is abandoned, not retried forever
# --------------------------------------------------------------------------- #
def test_a_timed_out_movie_is_flagged_so_it_is_never_offered_again(tmp_path):
    d = movie_dir(tmp_path)

    def runner(argv, cwd, timeout):
        raise TimeoutExpired('autoprocess', timeout)

    ok, note = ap.run_for_movie(['autoprocess'], d, 600, runner)

    assert ok is False
    assert flagged(d)
    assert 'abandoned' in note


def test_the_flag_says_abandoned_rather_than_complete(tmp_path):
    """Absence of the flag has to keep meaning "do this movie again", so the
    distinction lives in the flag's CONTENT."""
    d = movie_dir(tmp_path)

    def runner(argv, cwd, timeout):
        raise TimeoutExpired('autoprocess', timeout)

    ap.run_for_movie(['autoprocess'], d, 600, runner)
    content = open(os.path.join(d, ap.AUTOPROCESS_FLAG)).read()

    assert 'abandoned' in content
    assert '600' in content


def test_a_finished_movie_is_flagged_complete(tmp_path):
    d = movie_dir(tmp_path, artefact='CORRECT.LP')

    ok, note = ap.run_for_movie(['autoprocess'], d, 600,
                                lambda argv, cwd, timeout: Result(0))

    assert ok is True
    assert open(os.path.join(d, ap.AUTOPROCESS_FLAG)).read().strip() == \
        'autoprocess complete'


# --------------------------------------------------------------------------- #
# A zero exit with nothing processed
# --------------------------------------------------------------------------- #
def test_exit_zero_with_no_output_is_still_a_failure(tmp_path):
    """A missing --microscope-config makes autoprocess find nothing and exit 0.
    The post-condition is checked, not the child's stdout."""
    d = movie_dir(tmp_path)

    ok, note = ap.run_for_movie(['autoprocess'], d, 600,
                                lambda argv, cwd, timeout: Result(0))

    assert ok is False
    assert not flagged(d)
    assert 'no output' in note


def test_a_non_zero_exit_is_a_failure_and_leaves_no_flag(tmp_path):
    d = movie_dir(tmp_path, artefact='XDS.INP')

    ok, note = ap.run_for_movie(['autoprocess'], d, 600,
                                lambda argv, cwd, timeout: Result(3))

    assert ok is False
    assert not flagged(d)
