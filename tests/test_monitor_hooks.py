"""The monitor's extension points.

REyesProcessor is extended by subclassing. These tests pin the hooks a subclass
may rely on -- the stage table, the exit code of a stage, the status document,
the abort and idle state sets, the state after eucentricity, the per-movie
callback and the idle backstop's `busy_elsewhere` -- and that the monitor
behaves exactly as before when none of them is overridden.
"""

import json
import os
from enum import Enum, auto
from subprocess import TimeoutExpired

import pytest

from conftest import TracingProcessor, drive
import pyREyes.REyes as reyes
from pyREyes.REyes import (ABORT_STATES, IDLE_STATES, MONITOR_OWNED,
                           ProcessingState, REyesProcessor)


class Result:
    def __init__(self, returncode=0, stdout='', stderr=''):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def processor(tmp_path, monkeypatch, cls=REyesProcessor, **kwargs):
    monkeypatch.chdir(tmp_path)          # REyes_logs/ is created in the CWD
    return cls(working_directory=str(tmp_path), **kwargs)


def status(root):
    return json.loads(open(os.path.join(str(root), 'session_status.json')).read())


# --------------------------------------------------------------------------- #
# module_commands / run_module_code / run_module
# --------------------------------------------------------------------------- #
class WithExtraStage(REyesProcessor):
    def module_commands(self):
        commands = super().module_commands()
        # exits with the status given as its positional argument
        commands['exit_with'] = ['sh', '-c', 'exit "$1"', 'sh']
        return commands


def test_the_stage_table_lists_the_shipped_stages(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)
    assert p.module_commands() == {
        'grid_squares': ['grid-squares-0'],
        'manual_squares': ['manual-squares-0-1'],
        'eucentricity': ['eucentricity-1'],
        'dif_map': ['dif-map-2'],
        'write_targets': ['write-targets-3'],
        'append_targets': ['append-targets-3-1'],
        'create_final_targets': ['create-final-targets-4'],
        'mnt_maps_targets': ['mnt-maps-targets-5'],
    }
    # a new dict every time: a caller may edit its copy
    table = p.module_commands()
    table['x'] = ['x']
    assert 'x' not in p.module_commands()


def test_a_subclass_adds_a_stage_and_runs_it(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch, WithExtraStage)
    assert set(REyesProcessor.module_commands(p)) < set(p.module_commands())
    assert p.run_module('exit_with', {}) is False        # no status given
    assert p.run_module_code('exit_with', positional=('0',)) == 0

    base = processor(tmp_path, monkeypatch)
    assert base.run_module('exit_with') is False         # unknown stage
    assert base.run_module_code('exit_with') == 1


def test_run_module_code_returns_the_exit_status_and_run_module_the_bool(
        tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch, WithExtraStage)
    assert p.run_module_code('exit_with', positional=('4',)) == 4
    assert p.run_module_code('exit_with', positional=(0,)) == 0
    assert p.run_module_code('exit_with', positional=('1',)) == 1


def test_a_timeout_or_an_exception_gives_1(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)

    def times_out(cmd, cwd=None, timeout=None):
        raise TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(reyes, 'run_with_timeout', times_out)
    assert p.run_module_code('eucentricity') == 1
    assert p.run_module('eucentricity', {}) is False

    def explodes(cmd, cwd=None, timeout=None):
        raise OSError('no such command')

    monkeypatch.setattr(reyes, 'run_with_timeout', explodes)
    assert p.run_module_code('eucentricity') == 1


def test_options_are_rendered_before_the_positionals(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)
    seen = {}

    def capture(cmd, cwd=None, timeout=None):
        seen['cmd'] = list(cmd)
        seen['cwd'] = cwd
        return Result(0)

    monkeypatch.setattr(reyes, 'run_with_timeout', capture)
    assert p.run_module_code('dif_map', {'skip_processed': True, 'verbose': False,
                                         'proc_blocks': 1,
                                         'target_lists': ['sum', 'dqi']},
                             positional=('7', 8)) == 0
    assert seen['cmd'] == ['dif-map-2', '--skip-processed', '--proc-blocks', '1',
                           '--target-lists', 'sum', 'dqi', '7', '8']
    assert seen['cwd'] == p.working_directory


def test_the_run_module_contract_is_unchanged(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)
    codes = iter([0, 3])
    monkeypatch.setattr(reyes, 'run_with_timeout',
                        lambda cmd, cwd=None, timeout=None: Result(next(codes)))
    assert p.run_module('eucentricity', {}) is True
    assert p.run_module('eucentricity', {}) is False


# --------------------------------------------------------------------------- #
# status_payload
# --------------------------------------------------------------------------- #
class WithMode(REyesProcessor):
    def status_payload(self, state, detail, reason=None):
        payload = super().status_payload(state, detail, reason)
        payload['mode'] = 'extended'
        return payload


def test_the_default_payload_has_exactly_the_documented_keys(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)
    assert list(p.status_payload('failed', 'why')) == ['state', 'detail', 'utc']
    payload = p.status_payload('interrupted', 'why', reason='sigterm')
    assert list(payload) == ['state', 'detail', 'utc', 'reason']
    assert payload['state'] == 'interrupted' and payload['reason'] == 'sigterm'
    assert payload['utc'].endswith('Z')


def test_a_status_payload_override_lands_in_session_status_json(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch, WithMode)
    p._write_session_status('complete', 'all done', reason='none_needed')
    written = status(tmp_path)
    assert written['mode'] == 'extended'
    assert {k: written[k] for k in ('state', 'detail', 'reason')} == {
        'state': 'complete', 'detail': 'all done', 'reason': 'none_needed'}
    assert list(written) == ['state', 'detail', 'utc', 'reason', 'mode']


def test_the_status_is_written_at_status_path(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)
    assert p.status_path == os.path.join(p.working_directory, 'session_status.json')
    assert p.targets_dir == os.path.join(p.working_directory, 'targets')
    elsewhere = tmp_path / 'out' / 'status.json'
    elsewhere.parent.mkdir()
    p.status_path = str(elsewhere)
    p._write_session_status('failed', 'moved')
    assert json.loads(elsewhere.read_text())['detail'] == 'moved'
    assert not (tmp_path / 'session_status.json').exists()


# --------------------------------------------------------------------------- #
# abort_states / idle_states
# --------------------------------------------------------------------------- #
class ExtraState(Enum):
    WAITING_ELSEWHERE = auto()


class WaitsElsewhere(TracingProcessor):
    """Adds a waiting state of its own, with and without abort handling."""
    abort_elsewhere = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.abort_elsewhere:
            self.abort_states = self.abort_states | {ExtraState.WAITING_ELSEWHERE}
        self.state = ExtraState.WAITING_ELSEWHERE

    def state_handlers(self):
        handlers = super().state_handlers()
        handlers[ExtraState.WAITING_ELSEWHERE] = lambda: None
        return handlers


class AbortsElsewhere(WaitsElsewhere):
    abort_elsewhere = True


# three lines before the banner: a log this short would be taken for a stub
ABORT_BANNER = 'x\ny\nz\n!!!!! ERROR DURING PIPELINE EXECUTION !!!!!\n'


def test_the_state_sets_default_to_the_module_constants(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)
    assert p.abort_states == ABORT_STATES and p.idle_states == IDLE_STATES
    assert isinstance(p.abort_states, frozenset)


def test_a_subclass_abort_state_is_stopped_by_a_serialem_abort(tmp_path, monkeypatch):
    (tmp_path / 'g_1.log').write_text(ABORT_BANNER)
    p = processor(tmp_path, monkeypatch, AbortsElsewhere)
    drive(p, max_iterations=5)
    verdict = status(tmp_path)
    assert verdict['state'] == 'failed' and verdict['reason'] == 'serialem_aborted'
    assert 'WAITING_ELSEWHERE' in verdict['detail']


def test_without_the_state_in_abort_states_the_abort_is_left_to_the_handler(
        tmp_path, monkeypatch):
    (tmp_path / 'g_1.log').write_text(ABORT_BANNER)
    p = processor(tmp_path, monkeypatch, WaitsElsewhere)
    drive(p, max_iterations=5)          # runs out of ticks: KeyboardInterrupt
    assert status(tmp_path)['state'] == 'interrupted'


def test_a_subclass_idle_state_is_covered_by_the_idle_backstop(tmp_path, monkeypatch):
    (tmp_path / 'g_1.log').write_text('something\n')
    clock = {'t': 1000.0}
    monkeypatch.setattr(reyes.time, 'monotonic', lambda: clock['t'])

    def advance(_):
        clock['t'] += 1800

    monkeypatch.setattr(reyes.time, 'sleep', advance)
    p = processor(tmp_path, monkeypatch, WaitsElsewhere, idle_timeout=3600)
    p.idle_states = p.idle_states | {ExtraState.WAITING_ELSEWHERE}
    p.monitor()
    assert status(tmp_path)['reason'] == 'idle_timeout'


# --------------------------------------------------------------------------- #
# after_eucentricity_state
# --------------------------------------------------------------------------- #
class SkipsTheStepScan(TracingProcessor):
    def after_eucentricity_state(self):
        return ProcessingState.GENERATING_TARGETS


def test_after_eucentricity_state_changes_the_transition(session):
    p = SkipsTheStepScan(working_directory=session, microscope='Arctica-Apollo-SM')
    trace = drive(p)
    assert trace[:5] == [
        'STATE WAITING_FOR_MONTAGE',
        'MODULE grid_squares filtering=default microscope=Arctica-Apollo-SM',
        'STATE WAITING_FOR_GRID_SQUARES',
        'MODULE eucentricity',
        'STATE GENERATING_TARGETS',
    ]
    assert 'STATE WAITING_FOR_DIFFRACTION_MAP' not in trace
    assert not any(t.startswith('MODULE dif_map') for t in trace)
    assert trace[-1] == 'STATE COMPLETED'


def test_the_default_after_eucentricity_state_is_the_step_scan(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch)
    assert p.after_eucentricity_state() is ProcessingState.WAITING_FOR_DIFFRACTION_MAP


# --------------------------------------------------------------------------- #
# after_movie
# --------------------------------------------------------------------------- #
class RecordsMovies(REyesProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen = []

    def after_movie(self, movie_dir, ok):
        self.seen.append((os.path.basename(movie_dir), ok))


class HookExplodes(RecordsMovies):
    def after_movie(self, movie_dir, ok):
        super().after_movie(movie_dir, ok)
        raise RuntimeError('hook fault')


def movie(root, folder):
    name = f'{folder}_integrated_movie.mrc'
    movie_dir = root / 'movies' / folder
    movie_dir.mkdir(parents=True)
    (movie_dir / name).write_bytes(b'')
    return {'folder': folder, 'acquired_name': name,
            'suffixes': ['_integrated_movie.mrc']}


def test_after_movie_is_called_with_the_directory_and_the_verdict(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch, RecordsMovies,
                  autoprocess=True, microscope_config='X')
    verdicts = iter([(True, 'autoprocess complete'), (False, 'autoprocess exit 1')])
    monkeypatch.setattr(reyes, 'run_for_movie',
                        lambda *args, **kwargs: next(verdicts))

    assert p.process_movie(movie(tmp_path, 'mov1')) is True
    assert p.process_movie(movie(tmp_path, 'mov2')) is False
    assert p.seen == [('mov1', True), ('mov2', False)]


def test_after_movie_is_not_called_for_a_movie_already_done(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch, RecordsMovies,
                  autoprocess=True, microscope_config='X')
    info = movie(tmp_path, 'mov1')
    (tmp_path / 'movies' / 'mov1' / p.AUTOPROCESS_FLAG).write_text('done\n')
    monkeypatch.setattr(reyes, 'run_for_movie',
                        lambda *args, **kwargs: pytest.fail('must not run'))
    assert p.process_movie(info) is True
    assert p.seen == []


def test_a_fault_in_after_movie_does_not_change_the_verdict(tmp_path, monkeypatch):
    p = processor(tmp_path, monkeypatch, HookExplodes,
                  autoprocess=True, microscope_config='X')
    monkeypatch.setattr(reyes, 'run_for_movie',
                        lambda *args, **kwargs: (True, 'autoprocess complete'))
    assert p.process_movie(movie(tmp_path, 'mov1')) is True
    assert p.seen == [('mov1', True)]


# --------------------------------------------------------------------------- #
# busy_elsewhere
# --------------------------------------------------------------------------- #
class BusyForAWhile(TracingProcessor):
    """Reports work in flight elsewhere for the first `busy_ticks` ticks."""
    busy_ticks = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ticks = 0
        self.seen_on_entry = []

    def _handle_waiting_for_montage(self):
        self.seen_on_entry.append(self.busy_elsewhere)
        self.ticks += 1
        if self.ticks <= self.busy_ticks:
            self.busy_elsewhere = True
        return None


def test_busy_elsewhere_suppresses_the_idle_timeout_for_that_tick(tmp_path, monkeypatch):
    (tmp_path / 'g_1.log').write_text('something\n')
    clock = {'t': 1000.0}
    monkeypatch.setattr(reyes.time, 'monotonic', lambda: clock['t'])

    def advance(_):
        clock['t'] += 1800

    monkeypatch.setattr(reyes.time, 'sleep', advance)
    p = processor(tmp_path, monkeypatch, BusyForAWhile, idle_timeout=3 * 3600)
    p.monitor()

    assert status(tmp_path)['reason'] == 'idle_timeout'
    # ten busy ticks (5 h) did not count; the idle clock then ran its 3 h
    assert p.ticks == p.busy_ticks + 1 + 3 * 3600 // 1800
    assert clock['t'] - 1000.0 >= (p.busy_ticks + 6) * 1800
    # reset before every tick: a handler always finds it False
    assert p.seen_on_entry and not any(p.seen_on_entry)


def test_busy_elsewhere_starts_false_and_resets_the_idle_clock(tmp_path, monkeypatch):
    (tmp_path / 'g_1.log').write_text('something\n')
    p = processor(tmp_path, monkeypatch, idle_timeout=3600)
    assert p.busy_elsewhere is False
    assert not p._idle_expired()
    assert p._idle_clock is not None
    p.busy_elsewhere = True
    assert not p._idle_expired()
    assert p._idle_clock is None


# --------------------------------------------------------------------------- #
# MONITOR_OWNED
# --------------------------------------------------------------------------- #
class OwnsMore(REyesProcessor):
    MONITOR_OWNED = REyesProcessor.MONITOR_OWNED + ('extra_outputs',)


def test_monitor_owned_is_a_class_attribute_a_subclass_extends(tmp_path, monkeypatch):
    assert REyesProcessor.MONITOR_OWNED == MONITOR_OWNED
    assert isinstance(REyesProcessor.MONITOR_OWNED, tuple)
    (tmp_path / 'g_1.log').write_text('a\n')
    base = processor(tmp_path, monkeypatch)
    more = processor(tmp_path, monkeypatch, OwnsMore)
    before = base._arrival_signature()
    assert more._arrival_signature() == before
    (tmp_path / 'extra_outputs').mkdir()
    assert more._arrival_signature() == before
    assert base._arrival_signature() != before
