"""
Shared fixtures for the REyes test suite.

The centrepiece is `TracingProcessor`: a REyesProcessor with every side effect
replaced by a trace entry. What it records -- the exact sequence of states
entered and modules invoked, with their arguments -- is the contract the
monitor state machine must keep.

The session fixtures are synthetic rather than copies of real data: a session is
driven entirely by log filenames and by case-insensitive trigger phrases matched
over whole files, so a handful of small text files reproduces the state machine's
inputs exactly. Nothing here opens an MRC.
"""

import os
import time

import pytest

from pyREyes.REyes import REyesProcessor


# --------------------------------------------------------------------------- #
# Session fixtures
# --------------------------------------------------------------------------- #
def build_session(root, prefix='sample', lm='115', dims='8x8',
                  n_blocks=2, n_movies=2):
    """Write a complete, already-finished session directory.

    Every file the monitor looks for is present and carries its completion
    phrase, so a processor driven over this directory runs start to finish
    without waiting.
    """
    root = str(root)
    os.makedirs(os.path.join(root, 'dif_maps'), exist_ok=True)

    # Navigator: the filename is what find_nav_files() parses for prefix, LM
    # number and montage dimensions. Contents are not read in this path.
    with open(os.path.join(root, f'{prefix}_{lm}LM_{dims}_nav.nav'), 'w') as f:
        f.write('AdocVersion = 2.00\n\n[Item = 1]\nType = 2\n')

    # Montage log. The trigger embeds the LM number taken from the filename.
    with open(os.path.join(root, f'{prefix}_{lm}LM_{dims}.log'), 'w') as f:
        f.write(f'Montage started\nCompleted {lm}x LM Montage data collection\n')

    with open(os.path.join(root, f'{prefix}_grid_squares.log'), 'w') as f:
        f.write('Completed fine eucentricity for all grid squares\n')

    # One log per diffraction block. Only the LAST carries the "all grid
    # squares" phrase, so the monitor advances block by block exactly as it
    # would live.
    for block in range(1, n_blocks + 1):
        lines = ['Completed stepscan for current grid square\n']
        if block == n_blocks:
            lines.append('Completed stepscan for all grid squares\n')
        with open(os.path.join(root, f'{prefix}_{block}.log'), 'w') as f:
            f.writelines(lines)

    # Target generation reads this and requires at least one viable target.
    with open(os.path.join(root, 'dif_maps', 'dif_map_sums.csv'), 'w') as f:
        f.write('Coordinates,Sum,DifQuality\n')
        f.write('"(1.0, 2.0, 3.0)",1000,Good diffraction\n')
        f.write('"(4.0, 5.0, 6.0)",900,Bad diffraction\n')
        f.write('"(7.0, 8.0, 9.0)",800,Grid\n')

    # Movie log: folder/movie pairs are read in order and must stay paired.
    with open(os.path.join(root, f'{prefix}_targets.log'), 'w') as f:
        for movie in range(1, n_movies + 1):
            f.write(f'Movie folder: {prefix}_mov{movie}\n')
            f.write(f'Acquired movie: {prefix}_mov{movie}_integrated_movie.mrc\n')
        f.write('Completed acquiring tilt movies for all crystals!\n')

    return root


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A finished session directory, with the CWD moved into it.

    REyesProcessor writes REyes_logs/ relative to the working directory at
    construction time, so the chdir is part of the fixture rather than the test.
    """
    monkeypatch.chdir(tmp_path)
    return build_session(tmp_path)


@pytest.fixture
def manual_session(tmp_path, monkeypatch):
    """A session where manual grid squares have already been added.

    Needs two things the standard fixture does not have: a navigator carrying an
    item other than [Item = 1] (that is how check_nav_file_ready decides squares
    were added), and the manual_squares log with its completion phrase, since
    run_module is stubbed and will not write one.
    """
    monkeypatch.chdir(tmp_path)
    root = build_session(tmp_path)

    with open(os.path.join(root, 'sample_115LM_8x8_nav.nav'), 'w') as f:
        f.write('AdocVersion = 2.00\n\n[Item = 1]\nType = 2\n\n[Item = 2]\nType = 0\n')

    os.makedirs(os.path.join(root, 'REyes_logs'), exist_ok=True)
    with open(os.path.join(root, 'REyes_logs', 'manual_squares.log'), 'w') as f:
        f.write('Processing completed successfully!\n')

    return root


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
class TracingProcessor(REyesProcessor):
    """REyesProcessor that records transitions instead of doing work.

    Subprocesses are never launched: run_module, process_movie and run_autosolve
    all succeed immediately and append to `trace`. process_movie still performs
    the bookkeeping the real one does (`processed_movies`, `movies_count`),
    because the monitor's exit condition depends on it -- without that, a
    finished session looks like it has new movies forever.
    """

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        # setdefault so this works during __init__, which assigns self.state
        # before any attribute of ours could exist.
        self.__dict__.setdefault('trace', []).append(f'STATE {value.name}')
        self.__dict__['_state'] = value

    def run_module(self, module_name, args=None):
        rendered = ' '.join(f'{key}={args[key]}' for key in sorted(args)) if args else ''
        self.trace.append(f'MODULE {module_name} {rendered}'.rstrip())
        return True

    def process_movie(self, movie_info):
        movie_id = f"{movie_info['folder']}/{movie_info['acquired_name']}"
        self.processed_movies.add(movie_id)
        self.movies_count += 1
        self.trace.append(f"MOVIE {movie_info['acquired_name']}")
        return True

    def run_autosolve(self):
        self.trace.append('AUTOSOLVE')
        return True


def drive(processor, max_iterations=50):
    """Run monitor() to completion with the polling delay removed.

    monitor() returns of its own accord once a session completes. If a fixture
    never gets there, the iteration budget runs out and we raise KeyboardInterrupt
    -- which monitor() already treats as a clean shutdown -- so a stuck fixture
    fails as a trace mismatch instead of hanging the suite.
    """
    iterations = {'count': 0}

    def bounded_sleep(_seconds):
        iterations['count'] += 1
        if iterations['count'] > max_iterations:
            raise KeyboardInterrupt

    original_sleep = time.sleep
    time.sleep = bounded_sleep
    try:
        processor.monitor()
    finally:
        time.sleep = original_sleep
    return processor.trace
