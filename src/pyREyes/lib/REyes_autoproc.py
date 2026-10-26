"""Running autoprocess on ONE movie.

A leaf module: it imports nothing from the rest of pyREyes.

A movie gets its own wall-clock limit (`--movie-timeout`), separate from the
stages' `--subprocess-timeout`: a movie that takes far longer than the others
is usually a bad movie, and the session should not wait on it. A movie that
exceeds the limit is marked as finished with (the flag says it was abandoned),
so a restarted monitor does not offer it again.
"""

import os
from subprocess import TimeoutExpired

# Anything autoprocess leaves behind when it has actually indexed. Checked as a
# set of names rather than one, because which appear depends on how far XDS got
# -- a movie that failed to index still wrote CORRECT.LP, and that is a real
# (bad) result, not "nothing happened". The distinction being drawn here is
# "did it run at all", not "did it succeed".
AUTOPROCESS_ARTEFACTS = ('XDS.INP', 'CORRECT.LP', 'IDXREF.LP',
                         'XDS_ASCII.HKL', 'autoprocess_logs')

# Written by the monitor once a movie is finished with: a marker whose meaning
# the monitor controls, unlike the files autoprocess leaves behind.
AUTOPROCESS_FLAG = 'autoprocess_done.flag'


def produced_output(movie_dir, artefacts=AUTOPROCESS_ARTEFACTS):
    """True if autoprocess left anything behind in or under `movie_dir`."""
    try:
        for _root, dirs, files in os.walk(movie_dir):
            if (set(files) | set(dirs)) & set(artefacts):
                return True
    except OSError:
        # Cannot look -- do not claim success we have not verified.
        return False
    return False


def mark_done(movie_dir, note='autoprocess complete'):
    """Record that this movie is finished with. Returns True if written.

    `note` is the file's whole content, so an abandoned movie is distinguishable
    from a processed one by reading the flag rather than by its absence -- the
    absence has to keep meaning "do this movie again".
    """
    try:
        with open(os.path.join(movie_dir, AUTOPROCESS_FLAG), 'w',
                  newline='\n') as handle:
            handle.write(note if note.endswith('\n') else note + '\n')
        return True
    except OSError:
        return False


def run_for_movie(argv, movie_dir, timeout, runner, logger=None):
    """Run autoprocess for one movie. Returns (ok, note).

    `runner(cmd, cwd, timeout)` is the caller's `run_with_timeout`, injected so
    this module stays free of REyes imports and is trivially testable.

    A timeout is NOT a retryable failure: the flag is written anyway, because a
    movie that blew the limit once will blow it again and the monitor would
    otherwise offer it forever.
    """
    def say(level, message):
        if logger is not None:
            getattr(logger, level)(message)

    try:
        result = runner(argv, cwd=movie_dir, timeout=timeout)
    except TimeoutExpired:
        note = (f'autoprocess abandoned: exceeded the {timeout}s per-movie '
                f'limit')
        say('error', f"{note} in {movie_dir}. Its process group was killed; "
                     f"marking it finished so the session moves on.")
        if not mark_done(movie_dir, note):
            say('warning', f"could not write {AUTOPROCESS_FLAG} in {movie_dir}; "
                           f"a restarted monitor will offer this movie again")
        return False, note
    except Exception as exc:                        # noqa: BLE001 - reported
        say('error', f"Error running autoprocess in {movie_dir}: {exc}")
        return False, f'autoprocess raised: {exc}'

    if getattr(result, 'stdout', None):
        say('info', f"Autoprocess stdout:\n{result.stdout}")
    if getattr(result, 'stderr', None):
        say('warning', f"Autoprocess stderr:\n{result.stderr}")

    if result.returncode != 0:
        say('error', f"Autoprocess failed with return code "
                     f"{result.returncode}")
        return False, f'autoprocess exit {result.returncode}'

    # A zero exit is necessary but NOT sufficient: autoprocess exits 0 when it
    # finds nothing to process (e.g. a --microscope-config that does not match
    # the movie format). Check the POST-CONDITION.
    if not produced_output(movie_dir):
        say('error', f"Autoprocess exited 0 but produced no output in "
                     f"{movie_dir}. Treating as a failure: a zero exit with "
                     f"nothing processed is what a missing "
                     f"--microscope-config looks like.")
        return False, 'autoprocess exited 0 with no output'

    if not mark_done(movie_dir):
        # Not fatal: the movie is processed. Without the flag a restart
        # processes it again.
        say('warning', f"could not write {AUTOPROCESS_FLAG} in {movie_dir}")
    return True, 'autoprocess complete'
