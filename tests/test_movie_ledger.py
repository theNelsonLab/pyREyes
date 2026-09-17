"""
The movie ledger and the autoprocess output check.

A restarted monitor re-reads every movie from the targets log, so completion is
recorded on disk: with --autoprocess the monitor writes autoprocess_done.flag in
the movie folder, and without it the renamed movie file is the marker.

autoprocess can exit 0 without processing anything (e.g. a --microscope-config
that does not match the movie format). A zero exit therefore only counts as
success when autoprocess output is found on disk.
"""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

import pyREyes.REyes as reyes
from pyREyes.REyes import REyesProcessor


@pytest.fixture(autouse=True)
def in_tmp_path(tmp_path, monkeypatch):
    # REyesProcessor writes REyes_logs/ relative to the CWD.
    monkeypatch.chdir(tmp_path)


def make_processor(tmp_path, autoprocess=False):
    kwargs = dict(working_directory=str(tmp_path), microscope='Arctica-CETA')
    if autoprocess:
        kwargs.update(autoprocess=True, microscope_config='Arctica-CETA-mrc-SM')
    return REyesProcessor(**kwargs)


def make_movie(tmp_path, folder='sample_mov1',
               raw_name='20250101_12345_integrated_movie.mrc'):
    movie_dir = tmp_path / 'movies' / folder
    movie_dir.mkdir(parents=True)
    (movie_dir / raw_name).write_text('')
    return movie_dir, {
        'folder': folder,
        'acquired_name': f'{folder}.mrc',
        'suffixes': ['_integrated_movie.mrc', '_0_movie.mrc'],
    }


# --------------------------------------------------------------------------- #
# Output check
# --------------------------------------------------------------------------- #
def test_a_folder_autoprocess_never_touched_is_not_success(tmp_path):
    """Only the movie is there, nothing else."""
    (tmp_path / '20250101_12345_0_movie.mrc').write_text('')
    (tmp_path / 'sample-mov1.mrc').write_text('')

    assert not make_processor(tmp_path)._autoprocess_produced_output(str(tmp_path))


def test_a_processed_folder_is_recognised(tmp_path):
    (tmp_path / 'autoprocess_logs').mkdir()

    assert make_processor(tmp_path)._autoprocess_produced_output(str(tmp_path))


def test_output_nested_under_the_movie_folder_still_counts(tmp_path):
    """autoprocess works in a subdirectory named for the movie."""
    nested = tmp_path / 'sample-mov1'
    nested.mkdir()
    (nested / 'CORRECT.LP').write_text('')

    assert make_processor(tmp_path)._autoprocess_produced_output(str(tmp_path))


def test_a_movie_that_failed_to_index_still_counts_as_having_run(tmp_path):
    """A bad result is a result. The question here is 'did it run at all'."""
    (tmp_path / 'CORRECT.LP').write_text('')

    assert make_processor(tmp_path)._autoprocess_produced_output(str(tmp_path))


def test_a_folder_that_cannot_be_read_is_not_claimed_as_success(tmp_path):
    p = make_processor(tmp_path)

    assert not p._autoprocess_produced_output(str(tmp_path / 'nonexistent'))


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #
def test_a_finished_movie_is_skipped_on_restart(tmp_path):
    p = make_processor(tmp_path, autoprocess=True)
    p._mark_movie_done(str(tmp_path))

    assert p._movie_already_done(str(tmp_path), 'sample-mov1.mrc')


def test_a_movie_interrupted_during_xds_is_not_skipped(tmp_path):
    """autoprocess creates autoprocess_logs early, so its presence does not
    mean the movie finished. Only the flag does."""
    (tmp_path / 'autoprocess_logs').mkdir()
    (tmp_path / 'CORRECT.LP').write_text('partial')

    p = make_processor(tmp_path, autoprocess=True)
    assert not p._movie_already_done(str(tmp_path), 'x.mrc')


def test_an_unindexable_movie_is_still_marked_done(tmp_path):
    """Requiring CORRECT.LP would reprocess such movies on every restart."""
    p = make_processor(tmp_path, autoprocess=True)
    p._mark_movie_done(str(tmp_path))          # no CORRECT.LP anywhere

    assert p._movie_already_done(str(tmp_path), 'x.mrc')


def test_without_autoprocess_the_renamed_movie_is_the_marker(tmp_path):
    (tmp_path / 'sample-mov1.mrc').write_text('')
    p = make_processor(tmp_path)

    assert p._movie_already_done(str(tmp_path), 'sample-mov1.mrc')
    assert not p._movie_already_done(str(tmp_path), 'sample-mov2.mrc')


def test_a_missing_movie_folder_is_not_done(tmp_path):
    p = make_processor(tmp_path, autoprocess=True)

    assert not p._movie_already_done(str(tmp_path / 'nope'), 'x.mrc')


# --------------------------------------------------------------------------- #
# process_movie end to end (autoprocess stubbed)
# --------------------------------------------------------------------------- #
def stub_autoprocess(monkeypatch, returncode=0, artefact=None):
    calls = []

    def fake_run(cmd, cwd=None, timeout=None):
        calls.append(cmd)
        if artefact is not None:
            (Path(cwd) / artefact).write_text('')
        return CompletedProcess(cmd, returncode, '', '')

    monkeypatch.setattr(reyes, 'run_with_timeout', fake_run)
    return calls


def test_zero_exit_without_output_is_a_failure(tmp_path, monkeypatch):
    movie_dir, info = make_movie(tmp_path)
    calls = stub_autoprocess(monkeypatch, returncode=0, artefact=None)
    p = make_processor(tmp_path, autoprocess=True)

    assert p.process_movie(info) is False
    assert len(calls) == 1
    assert not (movie_dir / REyesProcessor.AUTOPROCESS_FLAG).exists()


def test_zero_exit_with_output_writes_the_flag(tmp_path, monkeypatch):
    movie_dir, info = make_movie(tmp_path)
    stub_autoprocess(monkeypatch, returncode=0, artefact='CORRECT.LP')
    p = make_processor(tmp_path, autoprocess=True)

    assert p.process_movie(info) is True
    assert (movie_dir / REyesProcessor.AUTOPROCESS_FLAG).is_file()
    assert (movie_dir / 'sample_mov1.mrc').is_file()


def test_non_zero_exit_does_not_write_the_flag(tmp_path, monkeypatch):
    movie_dir, info = make_movie(tmp_path)
    stub_autoprocess(monkeypatch, returncode=1, artefact='CORRECT.LP')
    p = make_processor(tmp_path, autoprocess=True)

    assert p.process_movie(info) is False
    assert not (movie_dir / REyesProcessor.AUTOPROCESS_FLAG).exists()


def test_restart_skips_finished_movies_but_still_counts_them(tmp_path, monkeypatch):
    movie_dir, info = make_movie(tmp_path)
    (movie_dir / REyesProcessor.AUTOPROCESS_FLAG).write_text('autoprocess complete\n')
    calls = stub_autoprocess(monkeypatch, returncode=0, artefact='CORRECT.LP')
    p = make_processor(tmp_path, autoprocess=True)

    assert p.process_movie(info) is True
    assert calls == []
    assert p.movies_count == 1
    assert 'sample_mov1/sample_mov1.mrc' in p.processed_movies


def test_movie_count_includes_skipped_and_new_movies(tmp_path, monkeypatch):
    done_dir, done = make_movie(tmp_path, folder='sample_mov1')
    (done_dir / REyesProcessor.AUTOPROCESS_FLAG).write_text('autoprocess complete\n')
    _, new = make_movie(tmp_path, folder='sample_mov2')
    calls = stub_autoprocess(monkeypatch, returncode=0, artefact='CORRECT.LP')
    p = make_processor(tmp_path, autoprocess=True)

    assert p.process_movie(done) is True
    assert p.process_movie(new) is True
    assert len(calls) == 1
    assert p.movies_count == 2


def test_restart_without_autoprocess_counts_renamed_movies(tmp_path):
    movie_dir, info = make_movie(tmp_path)
    p = make_processor(tmp_path)
    assert p.process_movie(info) is True
    assert p.movies_count == 1

    restarted = make_processor(tmp_path)
    assert restarted.process_movie(info) is True
    assert restarted.movies_count == 1
