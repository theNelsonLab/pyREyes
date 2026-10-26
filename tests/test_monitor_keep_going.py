"""
The monitor neither hangs nor loses a grid over a set-aside block or movie, or a
SerialEM abort, and every session ends with session_status.json.

The fixtures in `tests/data/session_logs/` are real SerialEM logs of a
four-grid session, with paths and dates neutralised:

    sample-J-reyes   J_3 valve-closed square; J_16 "StepScan incomplete"
                     + the START_REYES abort banner; the next grid's stub log
    sample-L-reyes   L_11 valve-closed square; L_16 like J_16
    sample-I-reyes   I_16: all squares mapped, then an error while waiting
                     for the targets navigator -> banner
    sample-K-reyes   K_16: the grid that SUCCEEDED (must never fire)

Records written only by the newer macros (BLOCK_SKIPPED, STPSCN_RESULT,
ITEM_SKIPPED) are built here from their exact strings.
"""

import json
import os
import shutil
import signal
import time
from pathlib import Path

import pytest

from conftest import TracingProcessor
from pyREyes import REyes as reyes_mod
from pyREyes.REyes import (ProcessingState, STOP, install_sigterm_handler)
from pyREyes.lib import REyes_serialem as serialem

LOGS = Path(__file__).resolve().parent / 'data' / 'session_logs'


def touch_ordered(paths, start=None):
    """Give files increasing mtimes, in the order given (seconds apart).

    By default the mtimes lie just ahead of now, as for logs SerialEM writes
    while the monitor runs.
    """
    if start is None:
        start = time.time() + 60
    for i, path in enumerate(paths):
        os.utime(path, (start + 10 * i, start + 10 * i))


# --------------------------------------------------------------------------- #
# Ordering the logs of one stage
# --------------------------------------------------------------------------- #
def test_resume_logs_come_after_their_original():
    names = ['x_targets-1.log', 'x_targets.log',
             'x_targets-10.log', 'x_targets-2.log']
    assert serialem.ordered_logs(names) == [
        'x_targets.log', 'x_targets-1.log',
        'x_targets-2.log', 'x_targets-10.log']


def test_without_resume_logs_the_order_is_plain_sorted():
    names = ['b_targets.log', 'a_115LM_8x8.log', 'a_grid_squares.log', 'c_1.log']
    assert serialem.ordered_logs(names) == sorted(names)


# --------------------------------------------------------------------------- #
# Record parsing: exact prefix only
# --------------------------------------------------------------------------- #
def test_records_are_matched_by_exact_prefix():
    assert serialem.parse_record(
        'ITEM_SKIPPED | stage=DFMV | item=4 | reason=valve_closed') == \
        ('ITEM_SKIPPED', {'stage': 'DFMV', 'item': '4',
                          'reason': 'valve_closed'})
    # Inside an ordinary Echo, or another record's field: not a record.
    assert serialem.parse_record(
        'Echo said ITEM_SKIPPED | stage=DFMV | item=4') is None
    assert serialem.parse_record(
        'OTHER_RECORD | stage=DFMV | item=4 | what=ITEM_SKIPPED') is None
    assert serialem.parse_record('ITEM_SKIPPEDX | stage=DFMV | item=4') is None


def test_skipped_items_reads_both_forms_and_ignores_other_stages(tmp_path):
    log = tmp_path / 'x_targets.log'
    log.write_text(
        'VALVE_CLOSED | stage=DFMV | item=3 | folder=x_3 | action=not_copied_not_marked_done\n'
        'ITEM_SKIPPED | stage=DFMV | item=3 | reason=valve_closed\n'
        'ITEM_SKIPPED | stage=DFMV | item=5 | reason=error_acquire\n'
        'VALVE_CLOSED | stage=STPSCN | item=9 | folder=y_9 | action=x\n'
        'ITEM_SKIPPED | stage=OTHER | item=7 | reason=error_move\n'
        'OTHER_RECORD | stage=DFMV | item=6 | reason=valve_closed\n')
    found = serialem.skipped_items([str(log)], ('DFMV',))
    assert {i: e['reason'] for i, e in found.items()} == \
        {3: 'valve_closed', 5: 'error_acquire'}
    assert found[3]['record'] == 'ITEM_SKIPPED'


# --------------------------------------------------------------------------- #
# Step-scan block records on the real logs
# --------------------------------------------------------------------------- #
def read(path):
    return serialem.read_text(str(path))


def test_the_real_valve_closed_squares_are_recognised():
    assert serialem.block_skip(read(LOGS / 'sample-J-reyes' / 'sample-J_3.log'), 3) \
        == ('valve_closed', 'VALVE_CLOSED')
    assert serialem.block_skip(read(LOGS / 'sample-L-reyes' / 'sample-L_11.log'), 11) \
        == ('valve_closed', 'VALVE_CLOSED')
    for log, square in (('sample-J-reyes/sample-J_4.log', 4),
                        ('sample-L-reyes/sample-L_12.log', 12),
                        ('sample-K-reyes/sample-K_16.log', 16)):
        assert serialem.block_skip(read(LOGS / log), square) is None


def test_a_skip_is_keyed_on_its_square_not_its_file():
    """A square that fails before opening its own log has its BLOCK_SKIPPED
    written into the PREVIOUS square's log."""
    text = ('Completed StepScan for current grid square\n'
            'BLOCK_SKIPPED | stage=STPSCN | square=6 | item=7 | reason=error_open_log\n')
    assert serialem.block_skip(text, 5) is None
    assert serialem.block_skip(text, 6) == ('error_open_log', 'BLOCK_SKIPPED')


def test_block_skipped_wins_over_the_older_valve_line():
    text = ('VALVE_CLOSED | stage=STPSCN | square=3 | action=not_copied_not_marked_done\n'
            'BLOCK_SKIPPED | stage=STPSCN | square=3 | item=4 | reason=valve_closed\n')
    assert serialem.block_skip(text, 3) == ('valve_closed', 'BLOCK_SKIPPED')


def test_the_end_of_the_real_short_runs_reads_partial():
    assert serialem.stpscn_result(read(LOGS / 'sample-J-reyes' / 'sample-J_16.log')) == 'partial'
    assert serialem.stpscn_result(read(LOGS / 'sample-L-reyes' / 'sample-L_16.log')) == 'partial'
    # A complete run carries the summary but not "StepScan incomplete".
    assert serialem.stpscn_result(read(LOGS / 'sample-K-reyes' / 'sample-K_16.log')) is None
    assert serialem.stpscn_result(read(LOGS / 'sample-I-reyes' / 'sample-I_16.log')) is None


@pytest.mark.parametrize('result', ['complete', 'partial', 'failed'])
def test_the_result_record(result):
    text = (f'STPSCN_SUMMARY | squares=16 | collected=15 | skipped=0 | set_aside=1\n'
            f'STPSCN_RESULT | result={result} | collected=15 | set_aside=1\n')
    assert serialem.stpscn_result(text) == result


# --------------------------------------------------------------------------- #
# SerialEM abandoned the grid
# --------------------------------------------------------------------------- #
def grid_copy(tmp_path, name, files=None):
    src = LOGS / name
    dst = tmp_path / 'session' / name
    dst.mkdir(parents=True)
    for f in sorted(src.iterdir()):
        if files is None or f.name in files:
            shutil.copy(f, dst / f.name)
    return dst


@pytest.mark.parametrize('grid,log', [('sample-J-reyes', 'sample-J_16.log'),
                                      ('sample-L-reyes', 'sample-L_16.log'),
                                      ('sample-I-reyes', 'sample-I_16.log')])
def test_the_real_aborted_grids_are_seen(tmp_path, grid, log):
    folder = grid_copy(tmp_path, grid)
    # As written live: the blocks, then the banner's log, then the next
    # grid's stub log.
    others = sorted(p for p in folder.iterdir()
                    if p.name != log and p.stat().st_size > 64)
    stubs = [p for p in folder.iterdir() if p.stat().st_size <= 64]
    touch_ordered(others + [folder / log] + stubs)
    found = serialem.serialem_abort(folder)
    assert found is not None and found['reason'] == 'serialem_aborted'
    assert found['log'] == log
    assert 'ERROR DURING PIPELINE EXECUTION' in found['text']


def test_the_grid_that_succeeded_is_never_called_aborted(tmp_path):
    folder = grid_copy(tmp_path, 'sample-K-reyes')
    assert serialem.serialem_abort(folder) is None


def test_a_banner_naming_another_grid_is_not_ours(tmp_path):
    folder = grid_copy(tmp_path, 'sample-J-reyes')
    renamed = folder.parent / 'sample-X-reyes'
    folder.rename(renamed)
    assert serialem.serialem_abort(renamed) is None


def test_a_recovered_grid_is_not_halted_by_its_old_abort(tmp_path):
    """A grid aborted (banner), then was resumed and wrote new logs. A monitor
    restarted for the resume must not halt on the abort it is recovering from."""
    folder = tmp_path / 'sample-J-reyes'
    folder.mkdir()
    shutil.copy(LOGS / 'sample-J-reyes' / 'sample-J_16.log', folder)
    assert serialem.serialem_abort(folder)['log'] == 'sample-J_16.log'

    (folder / 'sample-J_16-1.log').write_text(
        'x\ny\nz\nCompleted StepScan for current grid square\n')
    touch_ordered([folder / 'sample-J_16.log', folder / 'sample-J_16-1.log'])
    assert serialem.serialem_abort(folder) is None


def test_the_run_summary_names_a_failed_grid(tmp_path):
    """L is the last grid: SerialEM's summary goes into the session folder
    after L_16. A FAILED line for THIS grid counts; others do not."""
    session = tmp_path / 'session'
    folder = session / 'sample-L-reyes'
    folder.mkdir(parents=True)
    (folder / 'sample-L_12.log').write_text('a\nb\nc\nCompleted StepScan for current grid square\n')
    summary = session / 'MultiGrid_FSD_20250103_summary.log'
    summary.write_text('Slot 11 (sample-K-reyes): SUCCESS - None\n'
                       'Slot 12 (sample-L-reyes): FAILED - All_blocks_failed\n')
    touch_ordered([folder / 'sample-L_12.log', summary])
    found = serialem.serialem_abort(folder)
    assert found['reason'] == 'All_blocks_failed'

    other = session / 'sample-K-reyes'
    other.mkdir()
    (other / 'sample-K_16.log').write_text('a\nb\nc\nd\n')
    touch_ordered([other / 'sample-K_16.log', summary])
    assert serialem.serialem_abort(other) is None


def test_a_set_aside_success_line_is_not_a_failure(tmp_path):
    session = tmp_path / 'S'
    folder = session / 'g-reyes'
    folder.mkdir(parents=True)
    (folder / 'g_1.log').write_text('a\nb\nc\nd\n')
    summary = session / 'MultiGrid_FSD_summary.log'
    summary.write_text('Slot 1 (g-reyes): SUCCESS - Blocks_set_aside_2\n')
    touch_ordered([folder / 'g_1.log', summary])
    assert serialem.serialem_abort(folder) is None


# --------------------------------------------------------------------------- #
# The monitor end to end on the real J / L / I / K logs
# --------------------------------------------------------------------------- #
def log_session(tmp_path, grid, blocks):
    """A session whose block N log is `blocks[N]` (a real log file).

    The montage/grid-squares prefix is synthetic; the step-scan is real.
    """
    sample = grid[:-len('-reyes')]
    root = tmp_path / grid
    (root / 'dif_maps').mkdir(parents=True)
    (root / f'{sample}_115LM_6x6_nav.nav').write_text('AdocVersion = 2.00\n\n[Item = 1]\n')
    (root / f'{sample}_115LM_6x6.log').write_text(
        'x\ny\nz\nCompleted 115x LM Montage data collection\n')
    (root / f'{sample}_grid_squares.log').write_text(
        'x\ny\nz\nCompleted fine eucentricity for all grid squares\n')
    (root / 'dif_maps' / 'dif_map_sums.csv').write_text(
        'Coordinates,Sum,DifQuality\n"(1.0, 2.0, 3.0)",1000,Good diffraction\n')
    for prefix_file in root.iterdir():           # written before the blocks
        if prefix_file.is_file():
            os.utime(prefix_file, (1_600_000_000, 1_600_000_000))
    order = []
    for block, source in sorted(blocks.items()):
        dst = root / f'{sample}_{block}.log'
        shutil.copy(source, dst)
        order.append(dst)
    touch_ordered(order)
    return root


def drive_until_stop(processor, max_ticks=200):
    """The monitor loop with the sleep removed; returns the trace."""
    original = time.sleep
    ticks = {'n': 0}

    def bounded(_):
        ticks['n'] += 1
        if ticks['n'] > max_ticks:
            raise KeyboardInterrupt

    time.sleep = bounded
    try:
        processor.monitor()
    finally:
        time.sleep = original
    return processor.trace


def status(root):
    return json.loads((Path(root) / 'session_status.json').read_text())


def j_like_blocks(grid, bad_block, bad_log, last_log, good_log):
    blocks = {n: good_log for n in range(1, 16)}
    blocks[bad_block] = bad_log
    blocks[16] = last_log
    return blocks


@pytest.mark.parametrize('grid,bad', [('sample-J-reyes', 3), ('sample-L-reyes', 11)])
def test_the_real_valve_closed_grid_maps_its_other_15_blocks(tmp_path, monkeypatch,
                                                             grid, bad):
    """A valve-closed square used to hold the monitor at that block. Now: the
    square is set aside, the other 15 are mapped, targets are generated, and
    the session ends FAILED only because SerialEM abandoned the grid."""
    g = LOGS / grid
    s = grid[:-len('-reyes')]
    good = g / (f'{s}_4.log' if bad == 3 else f'{s}_12.log')
    blocks = j_like_blocks(grid, bad, g / f'{s}_{bad}.log', g / f'{s}_16.log', good)
    root = log_session(tmp_path, grid, blocks)
    monkeypatch.chdir(root)
    p = TracingProcessor(working_directory=str(root), microscope='Arctica-Apollo-SM')

    trace = drive_until_stop(p)

    mapped = [t for t in trace if t.startswith('MODULE dif_map')]
    assert len(mapped) == 15
    assert not any(f'folder={s}_{bad} ' in t for t in mapped)
    assert 'STATE GENERATING_TARGETS' in trace
    assert any(t.startswith('MODULE create_final_targets') for t in trace)
    assert trace[-1] == 'STATE WAITING_FOR_MOVIES'
    assert p.skipped_blocks[bad]['reason'] == 'valve_closed'
    verdict = status(root)
    assert verdict['state'] == 'failed' and verdict['reason'] == 'serialem_aborted'
    listed = serialem.load_set_aside(root / 'targets')
    assert [(e['stage'], e['item'], e['reason']) for e in listed] == \
        [('STPSCN', bad, 'valve_closed')]


def test_the_real_i_abort_after_targets_ends_failed(tmp_path, monkeypatch):
    g = LOGS / 'sample-I-reyes'
    root = log_session(tmp_path, 'sample-I-reyes',
                           {1: g / 'sample-I_16.log'})
    # One block whose log carries the all-squares phrase, then the banner.
    os.rename(root / 'sample-I_1.log', root / 'sample-I_1.log')
    monkeypatch.chdir(root)
    p = TracingProcessor(working_directory=str(root), microscope='Arctica-Apollo-SM')

    trace = drive_until_stop(p)

    assert 'STATE WAITING_FOR_MOVIES' in trace
    assert status(root)['reason'] == 'serialem_aborted'


def test_the_real_successful_grid_never_halts(tmp_path, monkeypatch):
    g = LOGS / 'sample-K-reyes'
    root = log_session(tmp_path, 'sample-K-reyes', {1: g / 'sample-K_16.log'})
    shutil.copy(g / 'sample-L_FSD_20250103.log', root)
    monkeypatch.chdir(root)
    p = TracingProcessor(working_directory=str(root), microscope='Arctica-Apollo-SM')

    trace = drive_until_stop(p, max_ticks=20)

    assert trace[-1] == 'STATE WAITING_FOR_MOVIES'
    assert status(root)['state'] == 'interrupted'      # the test stopped it
    assert p.serialem_abort is None


# --- records of the newer macros, synthetic -------------------------------- #
DONE = 'x\ny\nz\nCompleted StepScan for current grid square\n'
ALL = 'Completed StepScan for all grid squares\n'


def records_session(tmp_path, monkeypatch, logs):
    root = log_session(tmp_path, 'g-reyes', {})
    for i, (block, text) in enumerate(sorted(logs.items())):
        path = root / f'g_{block}.log'
        path.write_text(text)
        os.utime(path, (time.time() + 100 + i, time.time() + 100 + i))
    monkeypatch.chdir(root)
    return root, TracingProcessor(working_directory=str(root),
                                  microscope='Arctica-Apollo-SM',
                                  stepscan_only=True)


def test_skipped_block_in_its_own_log(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {
        1: DONE,
        2: 'x\nBLOCK_SKIPPED | stage=STPSCN | square=2 | item=3 | reason=error_acquire\n',
        3: DONE + 'STPSCN_SUMMARY | squares=3 | collected=2 | skipped=0 | set_aside=1\n'
                  'STPSCN_RESULT | result=partial | collected=2 | set_aside=1\n' + ALL})
    trace = drive_until_stop(p)
    assert [t.split()[2] for t in trace if t.startswith('MODULE dif_map')] == \
        ['folder=g_1', 'folder=g_3']
    assert trace[-1] == 'STATE COMPLETED'
    assert status(root)['state'] == 'complete'
    assert p.skipped_blocks == {2: {'reason': 'error_acquire', 'log': 'g_2.log'}}


def test_skip_before_the_square_opened_its_log(tmp_path, monkeypatch):
    """Square 2 failed before opening its log: its BLOCK_SKIPPED is in square 1's log and
    square 2 has no log. The monitor must not wait for it."""
    root, p = records_session(tmp_path, monkeypatch, {
        1: DONE + 'BLOCK_SKIPPED | stage=STPSCN | square=2 | item=3 | reason=error_open_log\n',
        3: DONE + ALL})
    trace = drive_until_stop(p)
    assert [t.split()[2] for t in trace if t.startswith('MODULE dif_map')] == \
        ['folder=g_1', 'folder=g_3']
    assert p.skipped_blocks[2]['reason'] == 'error_open_log'
    assert trace[-1] == 'STATE COMPLETED'


def test_last_square_skipped_before_its_log(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {
        1: DONE,
        2: DONE + 'BLOCK_SKIPPED | stage=STPSCN | square=3 | item=4 | reason=error_resume_check\n'
                  'STPSCN_RESULT | result=partial | collected=2 | set_aside=1\n' + ALL})
    trace = drive_until_stop(p)
    assert trace[-1] == 'STATE COMPLETED'
    # block 2's log carries the all-squares phrase, so the monitor goes
    # straight to targets after it.
    assert len([t for t in trace if t.startswith('MODULE dif_map')]) == 2


def test_failed_result_halts_cleanly(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {
        1: 'x\nBLOCK_SKIPPED | stage=STPSCN | square=1 | item=2 | reason=valve_closed\n'
           'STPSCN_SUMMARY | squares=1 | collected=0 | skipped=0 | set_aside=1\n'
           'STPSCN_RESULT | result=failed | collected=0 | set_aside=1\n'})
    trace = drive_until_stop(p)
    assert not any(t.startswith('MODULE dif_map') for t in trace)
    assert status(root)['state'] == 'failed'
    assert status(root)['reason'] == 'stepscan_failed'


def test_the_save_order_race_on_the_last_block(tmp_path, monkeypatch):
    """The last block's completion phrase and the all-squares phrase arrive in
    two saves; a monitor that read between them used to wait for block 3."""
    root, p = records_session(tmp_path, monkeypatch, {1: DONE, 2: DONE})
    handler = p.state_handlers()
    for _ in range(10):
        handler[p.state]()
        if p.state == ProcessingState.WAITING_FOR_DIFFRACTION_MAP and p.current_block == 3:
            break
    assert p.current_block == 3
    with open(root / 'g_2.log', 'a') as f:
        f.write(ALL)
    handler[p.state]()
    assert p.state == ProcessingState.GENERATING_TARGETS


def test_a_block_that_never_finished_is_set_aside_once_serialem_moved_on(
        tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {
        1: DONE, 2: 'x\ny\nz\nstill going\n', 3: DONE + ALL})
    trace = drive_until_stop(p)
    assert p.skipped_blocks[2]['reason'] == 'no_completion_phrase'
    assert trace[-1] == 'STATE COMPLETED'


def test_no_block_mapped_and_serialem_aborted_halts(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {
        1: 'x\ny\n\n!!!!! ERROR DURING PIPELINE EXECUTION !!!!!\nGrid: g-reyes (Slot 1)\n'})
    trace = drive_until_stop(p)
    assert status(root)['reason'] == 'serialem_aborted'
    assert 'STATE GENERATING_TARGETS' not in trace


# --------------------------------------------------------------------------- #
# Movies (resume logs, set-aside movies)
# --------------------------------------------------------------------------- #
def test_movies_of_a_resumed_movie_stage_are_processed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 's_targets.log').write_text(
        'Movie folder: m1\nAcquired movie: m1_integrated_movie.mrc\n')
    (tmp_path / 's_targets-1.log').write_text(
        'VALVE_CLOSED | stage=DFMV | item=7 | folder=m2 | action=not_copied_not_marked_done\n'
        'ITEM_SKIPPED | stage=DFMV | item=7 | reason=valve_closed\n'
        'Movie folder: m3\nAcquired movie: m3_integrated_movie.mrc\n'
        'Completed acquiring tilt movies for all crystals!\n')
    p = TracingProcessor(working_directory=str(tmp_path), microscope='Arctica-Apollo-SM')
    p.state = ProcessingState.WAITING_FOR_MOVIES

    assert p._handle_waiting_for_movies() is STOP
    assert [t for t in p.trace if t.startswith('MOVIE')] == [
        'MOVIE m1_integrated_movie.mrc', 'MOVIE m3_integrated_movie.mrc']
    listed = serialem.load_set_aside(tmp_path / 'targets')
    assert [(e['stage'], e['item']) for e in listed] == [('DFMV', 7)]



def test_sigterm_leaves_an_interrupted_verdict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = TracingProcessor(working_directory=str(tmp_path), microscope='Arctica-Apollo-SM')
    previous = signal.getsignal(signal.SIGTERM)
    try:
        assert install_sigterm_handler(p)
        monkeypatch.setattr(p, '_handle_waiting_for_montage',
                            lambda: os.kill(os.getpid(), signal.SIGTERM))
        p.monitor()
    finally:
        signal.signal(signal.SIGTERM, previous)
    verdict = status(tmp_path)
    assert verdict['state'] == 'interrupted' and verdict['reason'] == 'sigterm'
    assert 'WAITING_FOR_MONTAGE' in verdict['detail']


def test_the_idle_backstop_stops_a_monitor_serialem_forgot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'g_1.log').write_text('something\n')
    clock = {'t': 1000.0}
    monkeypatch.setattr(reyes_mod.time, 'monotonic', lambda: clock['t'])

    def advance(_):
        clock['t'] += 1800

    monkeypatch.setattr(reyes_mod.time, 'sleep', advance)
    p = TracingProcessor(working_directory=str(tmp_path),
                         microscope='Arctica-Apollo-SM', idle_timeout=3 * 3600)
    p.monitor()
    verdict = status(tmp_path)
    assert verdict['state'] == 'interrupted' and verdict['reason'] == 'idle_timeout'
    assert clock['t'] - 1000.0 >= 3 * 3600


def test_arrivals_keep_the_idle_clock_from_running_out(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    log = tmp_path / 'g_1.log'
    log.write_text('a\n')
    clock = {'t': 1000.0, 'n': 0}
    monkeypatch.setattr(reyes_mod.time, 'monotonic', lambda: clock['t'])

    def advance(_):
        clock['t'] += 1800
        clock['n'] += 1
        with open(log, 'a') as f:
            f.write('more\n')
        if clock['n'] > 20:
            raise KeyboardInterrupt

    monkeypatch.setattr(reyes_mod.time, 'sleep', advance)
    p = TracingProcessor(working_directory=str(tmp_path),
                         microscope='Arctica-Apollo-SM', idle_timeout=3 * 3600)
    p.monitor()
    assert status(tmp_path).get('reason') != 'idle_timeout'


def test_the_idle_backstop_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = TracingProcessor(working_directory=str(tmp_path),
                         microscope='Arctica-Apollo-SM', idle_timeout=0)
    assert p.idle_timeout is None and not p._idle_expired()


def test_the_monitor_parser_takes_an_idle_timeout(monkeypatch, tmp_path):
    captured = {}

    class Stop(Exception):
        pass

    def fake_processor(**kwargs):
        captured.update(kwargs)
        raise Stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reyes_mod, 'REyesProcessor', fake_processor)
    monkeypatch.setattr(reyes_mod.sys, 'argv',
                        ['reyes-monitor', '--idle-timeout', '3600'])
    with pytest.raises(Stop):
        reyes_mod.main()
    assert captured['idle_timeout'] == 3600


# --------------------------------------------------------------------------- #
# Further records of the newer Apollo macros
# --------------------------------------------------------------------------- #
NO_DATA_BANNER = (
    '############################################################\n'
    '# NOT reporting completion: NO grid square has data.\n'
    '#   squares=4  mapped=0  skipped=0  set_aside=4\n'
    '# Either the navigator held no grid squares, or every square\n'
    '# failed. The grid is FAILED (All_blocks_failed). Re-run to map\n'
    '# them again; squares already mapped are skipped.\n'
    '############################################################\n')


def test_all_blocks_failed_in_the_newest_log_is_an_abort(tmp_path):
    folder = tmp_path / 'g-reyes'
    folder.mkdir()
    (folder / 'g_4.log').write_text(
        'x\nBLOCK_SKIPPED | stage=STPSCN | square=4 | item=5 | reason=valve_closed\n'
        'STPSCN_SUMMARY | squares=4 | collected=0 | skipped=0 | set_aside=4\n'
        'STPSCN_RESULT | result=failed | collected=0 | set_aside=4\n' + NO_DATA_BANNER)
    found = serialem.serialem_abort(folder)
    assert found['reason'] == 'All_blocks_failed'


def test_the_monitor_halts_on_a_step_scan_with_no_data(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {
        1: 'x\nBLOCK_SKIPPED | stage=STPSCN | square=1 | item=2 | reason=error_acquire\n'
           'STPSCN_SUMMARY | squares=1 | collected=0 | skipped=0 | set_aside=1\n'
           'STPSCN_RESULT | result=failed | collected=0 | set_aside=1\n' + NO_DATA_BANNER})
    trace = drive_until_stop(p)
    assert not any(t.startswith('MODULE dif_map') for t in trace)
    assert status(root)['state'] == 'failed'
    assert status(root)['reason'] == 'stepscan_failed'


@pytest.mark.parametrize('reason', ['Items_set_aside_3',
                                    'Blocks_set_aside_1Items_set_aside_2'])
def test_a_success_with_set_aside_movies_is_not_a_failure(tmp_path, reason):
    session = tmp_path / 'S'
    folder = session / 'g-reyes'
    folder.mkdir(parents=True)
    (folder / 'g_1.log').write_text('a\nb\nc\nd\n')
    summary = session / 'MultiGrid_FSD_summary.log'
    summary.write_text(f'Slot 1 (g-reyes): SUCCESS - {reason}\n')
    touch_ordered([folder / 'g_1.log', summary])
    assert serialem.serialem_abort(folder) is None


def test_wait_poll_errors_are_informational(tmp_path, monkeypatch):
    line = 'WAIT_POLL_ERROR | wait=targets_nav | errors=2'
    assert serialem.parse_record(line) == (
        'WAIT_POLL_ERROR', {'wait': 'targets_nav', 'errors': '2'})
    root, p = records_session(tmp_path, monkeypatch, {
        1: 'WAIT_POLL_ERROR | wait=ecnt_corrected | errors=1\n' + DONE,
        2: DONE + ALL + line + '\n'})
    trace = drive_until_stop(p)
    assert trace[-1] == 'STATE COMPLETED'
    assert status(root)['state'] == 'complete'
    assert p.serialem_abort is None and not p.skipped_blocks


def test_a_resumed_movie_stage_finishes_the_session(tmp_path, monkeypatch):
    """A resumed movie stage reads targets/range_dfmv.txt and writes
    `<sample>_targets-1.log`. Movies from both logs are processed, the set-aside
    movie is listed, and the end phrase of the resume log completes the session."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'targets').mkdir()
    (tmp_path / 'targets' / 'range_dfmv.txt').write_text('12\n14\n')
    (tmp_path / 's_targets.log').write_text(
        'Movie folder: s_mov1\nAcquired movie: s_mov1_0_movie.mrc\n'
        '!!!!! ERROR DURING PIPELINE EXECUTION !!!!!\nGrid: other-reyes\n')
    (tmp_path / 's_targets-1.log').write_text(
        'ITEM_SKIPPED | stage=DFMV | item=2 | reason=error_acquire\n'
        'Movie folder: s_mov3\nAcquired movie: s_mov3_0_movie.mrc\n'
        'Completed acquiring tilt movies for all crystals!\n')
    p = TracingProcessor(working_directory=str(tmp_path),
                         microscope='Arctica-Apollo-SM',
                         start_state='WAITING_FOR_MOVIES')

    trace = drive_until_stop(p, max_ticks=5)

    assert [t for t in trace if t.startswith('MOVIE')] == [
        'MOVIE s_mov1_0_movie.mrc', 'MOVIE s_mov3_0_movie.mrc']
    assert trace[-1] == 'STATE COMPLETED'
    assert status(tmp_path)['state'] == 'complete'
    listed = serialem.load_set_aside(tmp_path / 'targets')
    assert [(e['stage'], e['item'], e['reason']) for e in listed] == \
        [('DFMV', 2, 'error_acquire')]


def test_the_end_phrase_of_the_original_log_is_superseded_by_a_resume(tmp_path, monkeypatch):
    """Only the LATEST movie log's end phrase counts."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / 's_targets.log').write_text(
        'Movie folder: s_mov1\nAcquired movie: s_mov1_0_movie.mrc\n'
        'Completed acquiring tilt movies for all crystals!\n')
    (tmp_path / 's_targets-1.log').write_text(
        'Movie folder: s_mov2\nAcquired movie: s_mov2_0_movie.mrc\n')
    p = TracingProcessor(working_directory=str(tmp_path),
                         microscope='Arctica-Apollo-SM',
                         start_state='WAITING_FOR_MOVIES')

    assert p._handle_waiting_for_movies() is None
    assert p.state == ProcessingState.WAITING_FOR_MOVIES
    assert p.movies_count == 2


def test_monitor_written_folders_do_not_reset_the_idle_clock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'g_1.log').write_text('a\n')
    p = TracingProcessor(working_directory=str(tmp_path), microscope='Arctica-Apollo-SM')
    before = p._arrival_signature()
    (tmp_path / 'targets').mkdir()
    (tmp_path / 'targets' / 'range_dfmv.txt').write_text('1\n')
    (tmp_path / 'session_status.json').write_text('{}')
    assert p._arrival_signature() == before


# --------------------------------------------------------------------------- #
# Resumed grid squares: `<prefix>_<N>-k.log`
# --------------------------------------------------------------------------- #
def dif_map_folders(trace):
    return [t.split()[2] for t in trace if t.startswith('MODULE dif_map')]


def test_a_square_set_aside_and_mapped_again_on_resume_is_mapped(tmp_path, monkeypatch):
    """The original log set square 2 aside; the resume mapped it into
    `g_2-1.log`, which also ends the step-scan. Block 2 is mapped (folder
    `g_2`, without the `-1`) and the stage completes."""
    root, p = records_session(tmp_path, monkeypatch, {
        1: DONE,
        2: 'x\nBLOCK_SKIPPED | stage=STPSCN | square=2 | item=3 | reason=error_move\n'})
    resumed = root / 'g_2-1.log'
    resumed.write_text(DONE + 'STPSCN_RESULT | result=complete | collected=2 | set_aside=0\n' + ALL)
    os.utime(resumed, (time.time() + 200, time.time() + 200))

    trace = drive_until_stop(p, max_ticks=12)

    assert dif_map_folders(trace) == ['folder=g_1', 'folder=g_2']
    assert trace[-1] == 'STATE COMPLETED'
    assert not p.skipped_blocks
    assert status(root)['state'] == 'complete'


def test_the_last_square_completing_in_a_resume_log_ends_the_stage(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {1: DONE, 2: 'x\nstill going\n'})
    resumed = root / 'g_2-1.log'
    resumed.write_text(DONE + ALL)
    os.utime(resumed, (time.time() + 200, time.time() + 200))

    trace = drive_until_stop(p, max_ticks=12)

    assert dif_map_folders(trace) == ['folder=g_1', 'folder=g_2']
    assert trace[-1] == 'STATE COMPLETED'


def test_the_newest_of_several_resume_logs_is_the_blocks_log(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {1: DONE + ALL})
    for name in ('g_1-1.log', 'g_1-2.log', 'g_1-10.log'):
        (root / name).write_text('x\n')
    assert p._block_logs('g') == {1: 'g_1-10.log'}
    assert p._block_folder('g_1-10.log') == 'g_1'
    assert p._block_folder('g_12.log') == 'g_12'


def test_without_resume_logs_block_logs_are_the_plain_names(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {1: DONE, 2: DONE, 11: DONE + ALL})
    assert p._block_logs('g') == {1: 'g_1.log', 2: 'g_2.log', 11: 'g_11.log'}
    assert p.find_diffraction_map_logs({'prefix': 'g'}) == ['g_1.log']


def test_dif_map_reads_positions_from_the_newest_log_naming_the_frame(tmp_path):
    import pyREyes.dif_map as dm
    (tmp_path / 'REyes_logs').mkdir()
    for name in ('s_2.log', 's_2-1.log', 's_21.log', 'x_s_2.log'):
        (tmp_path / name).write_text('x\n')
    logs = [p.name for p in dm.block_logs('s_2', tmp_path)]
    assert logs[0] == 's_2-1.log' and set(logs) == {'s_2-1.log', 's_2.log', 'x_s_2.log'}


# --------------------------------------------------------------------------- #
# A restarted monitor
# --------------------------------------------------------------------------- #
BANNER = ("x\ny\nz\n!!!!! ERROR DURING PIPELINE EXECUTION !!!!!\n"
          "Grid: {g} (Slot 3)\nThe pipeline encountered an error and stopped\n")


def backdate(path, seconds=3600):
    then = time.time() - seconds
    os.utime(path, (then, then))


def make_aborted_grid(tmp_path, monkeypatch):
    d = tmp_path / 'sample-reyes'
    d.mkdir()
    monkeypatch.chdir(d)
    (d / 'sample_115LM_8x8_nav.nav').write_text('AdocVersion = 2.00\n\n[Item = 1]\n')
    (d / 'sample_115LM_8x8.log').write_text('a\nb\nc\nstarted montage\n')
    (d / 'sample_3.log').write_text(BANNER.format(g='sample-reyes'))
    for f in d.iterdir():
        backdate(f, 7200)
    return d


def test_a_monitor_started_after_an_abort_ends_with_it(tmp_path, monkeypatch):
    """Restarted after an abort but before SerialEM resumed the grid: nothing
    says SerialEM went on, so the abort is the session's verdict, exactly as
    for a session processed after the fact. Restart once the resume began."""
    d = make_aborted_grid(tmp_path, monkeypatch)
    p = TracingProcessor(working_directory=str(d), microscope='Arctica-CETA')

    drive_until_stop(p, max_ticks=3)

    assert status(d)['state'] == 'failed'
    assert status(d)['reason'] == 'serialem_aborted'


def test_an_abort_superseded_by_a_resume_log_is_ignored(tmp_path, monkeypatch):
    """SerialEM resumed after the abort: a newer substantive log of the grid
    exists, so the old banner is the abort being recovered from."""
    d = make_aborted_grid(tmp_path, monkeypatch)
    resumed = d / 'sample_3-1.log'
    resumed.write_text('a\nb\nc\nresumed at stage 4\n')
    backdate(resumed, 600)
    p = TracingProcessor(working_directory=str(d), microscope='Arctica-CETA')

    drive_until_stop(p, max_ticks=3)

    assert p.serialem_abort is None
    assert status(d)['state'] == 'interrupted'          # the test stopped it


def test_a_finished_session_processed_offline_ends_with_serialem_s_verdict(
        tmp_path, monkeypatch):
    """The real sample-J logs, every file written long before the monitor
    starts: the valve-closed block is set aside, the rest are mapped, targets
    are generated and the session ends failed, not waiting for movies."""
    g = LOGS / 'sample-J-reyes'
    blocks = j_like_blocks('sample-J-reyes', 3, g / 'sample-J_3.log',
                           g / 'sample-J_16.log', g / 'sample-J_4.log')
    root = log_session(tmp_path, 'sample-J-reyes', blocks)
    old = time.time() - 30 * 86400
    for block in sorted(blocks):              # same order, a month ago
        f = root / f'sample-J_{block}.log'
        os.utime(f, (old + 60 * block, old + 60 * block))
    monkeypatch.chdir(root)
    p = TracingProcessor(working_directory=str(root), microscope='Arctica-Apollo-SM')

    trace = drive_until_stop(p, max_ticks=60)

    assert len([t for t in trace if t.startswith('MODULE dif_map')]) == 15
    assert any(t.startswith('MODULE create_final_targets') for t in trace)
    assert status(root)['state'] == 'failed'
    assert status(root)['reason'] == 'serialem_aborted'


def test_an_abort_written_while_the_monitor_runs_still_stops_it(tmp_path, monkeypatch):
    d = tmp_path / 'sample-reyes'
    d.mkdir()
    monkeypatch.chdir(d)
    (d / 'sample_115LM_8x8_nav.nav').write_text('AdocVersion = 2.00\n\n[Item = 1]\n')
    old = d / 'sample_3.log'
    old.write_text(BANNER.format(g='sample-reyes'))
    backdate(old)
    p = TracingProcessor(working_directory=str(d), microscope='Arctica-CETA')
    new = d / 'sample_3-1.log'
    new.write_text(BANNER.format(g='sample-reyes'))
    os.utime(new, (time.time() + 5, time.time() + 5))

    drive_until_stop(p, max_ticks=3)

    assert status(d)['state'] == 'failed'
    assert status(d)['reason'] == 'serialem_aborted'
    assert p.serialem_abort['log'] == 'sample_3-1.log'


def test_a_restart_at_a_later_block_keeps_the_blocks_already_mapped(tmp_path, monkeypatch):
    """--current-block 5 after blocks 1-4 were mapped by an earlier monitor:
    an abort still generates targets from them."""
    d = tmp_path / 'sample-reyes'
    d.mkdir()
    monkeypatch.chdir(d)
    (d / 'sample_115LM_8x8_nav.nav').write_text('AdocVersion = 2.00\n\n[Item = 1]\n')
    (d / 'dif_maps').mkdir()
    (d / 'dif_maps' / 'dif_map_sums.csv').write_text(
        'Coordinates,Sum,DifQuality\n"(1.0, 2.0, 3.0)",1000,Good diffraction\n')
    for b in range(1, 5):
        (d / f'sample_{b}.log').write_text(DONE)
    p = TracingProcessor(working_directory=str(d), microscope='Arctica-CETA',
                         start_state='WAITING_FOR_DIFFRACTION_MAP', current_block=5)
    banner = d / 'sample_5.log'
    banner.write_text(BANNER.format(g='sample-reyes'))
    os.utime(banner, (time.time() + 5, time.time() + 5))

    trace = drive_until_stop(p, max_ticks=8)

    assert any(t.startswith('MODULE write_targets') for t in trace)
    assert any(t.startswith('MODULE create_final_targets') for t in trace)
    assert status(d)['state'] == 'failed' and status(d)['reason'] == 'serialem_aborted'


def test_the_polls_before_a_set_aside_restart_with_every_block(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {1: DONE, 2: DONE})
    p._missing_block_ticks = 2                # left over from an earlier block
    p._after_block('g_1.log')
    assert p._missing_block_ticks == 0


def test_a_monitor_set_aside_is_recorded_as_such(tmp_path, monkeypatch):
    root, p = records_session(tmp_path, monkeypatch, {
        1: DONE, 2: 'x\ny\nz\nstill going\n', 3: DONE + ALL})
    drive_until_stop(p)
    listed = serialem.load_set_aside(root / 'targets')
    assert [(e['item'], e['record'], e['source']) for e in listed] == \
        [(2, reyes_mod.MONITOR_SET_ASIDE, 'pyreyes')]


def test_sigterm_gives_exit_status_143(monkeypatch, tmp_path):
    class Stopped:
        logger = None

        def __init__(self, **kwargs):
            self.stop_reason = None

        def monitor(self):
            self.stop_reason = 'SIGTERM received in state WAITING_FOR_MONTAGE'

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(reyes_mod, 'REyesProcessor', Stopped)
    monkeypatch.setattr(reyes_mod, 'install_sigterm_handler', lambda p: True)
    monkeypatch.setattr(reyes_mod.sys, 'argv', ['reyes-monitor'])
    assert reyes_mod.main() == 143
