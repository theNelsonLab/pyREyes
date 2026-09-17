"""
Golden traces for the REyes monitor state machine.

A diff in one of these lists is a behaviour change, not a formatting change.

The traces are written out in full rather than computed, so that a change shows
up as a readable diff in review rather than as a recomputed expectation that
silently agrees with whatever the code now does.
"""

from conftest import TracingProcessor, drive


def test_standard_session(session):
    """Montage -> grid squares -> eucentricity -> two blocks -> targets -> movies."""
    processor = TracingProcessor(working_directory=session,
                                 microscope='Arctica-Apollo-SM')

    assert drive(processor) == [
        'STATE WAITING_FOR_MONTAGE',
        'MODULE grid_squares filtering=default microscope=Arctica-Apollo-SM',
        'STATE WAITING_FOR_GRID_SQUARES',
        'MODULE eucentricity',
        'STATE WAITING_FOR_DIFFRACTION_MAP',
        'MODULE dif_map folder=sample_1 microscope=Arctica-Apollo-SM '
        'proc_blocks=1 skip_processed=True',
        'MODULE dif_map folder=sample_2 microscope=Arctica-Apollo-SM '
        'proc_blocks=1 skip_processed=True',
        'STATE GENERATING_TARGETS',
        'MODULE write_targets microscope=Arctica-Apollo-SM',
        'MODULE create_final_targets target-lists=[\'sum\', \'peaks\', \'dqi\'] '
        'top-target-per-category=2',
        'STATE GENERATING_FINAL_MAP',
        'MODULE mnt_maps_targets microscope=Arctica-Apollo-SM',
        'STATE WAITING_FOR_MOVIES',
        'MOVIE sample_mov1_integrated_movie.mrc',
        'MOVIE sample_mov2_integrated_movie.mrc',
        'STATE COMPLETED',
    ]


def test_montage_only_stops_after_grid_squares(session):
    """--montage-only must not fall through into eucentricity."""
    processor = TracingProcessor(working_directory=session,
                                 microscope='Arctica-Apollo-SM',
                                 montage_only=True)

    assert drive(processor) == [
        'STATE WAITING_FOR_MONTAGE',
        'MODULE grid_squares filtering=default microscope=Arctica-Apollo-SM',
        'STATE COMPLETED',
    ]


def test_stepscan_only_stops_before_movies(session):
    """--stepscan-only produces the final map and then exits."""
    processor = TracingProcessor(working_directory=session,
                                 microscope='Arctica-Apollo-SM',
                                 stepscan_only=True)

    assert drive(processor) == [
        'STATE WAITING_FOR_MONTAGE',
        'MODULE grid_squares filtering=default microscope=Arctica-Apollo-SM',
        'STATE WAITING_FOR_GRID_SQUARES',
        'MODULE eucentricity',
        'STATE WAITING_FOR_DIFFRACTION_MAP',
        'MODULE dif_map folder=sample_1 microscope=Arctica-Apollo-SM '
        'proc_blocks=1 skip_processed=True',
        'MODULE dif_map folder=sample_2 microscope=Arctica-Apollo-SM '
        'proc_blocks=1 skip_processed=True',
        'STATE GENERATING_TARGETS',
        'MODULE write_targets microscope=Arctica-Apollo-SM',
        'MODULE create_final_targets target-lists=[\'sum\', \'peaks\', \'dqi\'] '
        'top-target-per-category=2',
        'STATE GENERATING_FINAL_MAP',
        'MODULE mnt_maps_targets microscope=Arctica-Apollo-SM',
        'STATE COMPLETED',
    ]


def test_autosolve_runs_after_movies(session):
    """--autosolve adds one state at the end and nothing else."""
    processor = TracingProcessor(working_directory=session,
                                 microscope='Arctica-Apollo-SM',
                                 autosolve=True)

    trace = drive(processor)
    assert trace[-5:] == [
        'MOVIE sample_mov1_integrated_movie.mrc',
        'MOVIE sample_mov2_integrated_movie.mrc',
        'STATE RUNNING_AUTOSOLVE',
        'AUTOSOLVE',
        'STATE COMPLETED',
    ]
    assert trace[-6] == 'STATE WAITING_FOR_MOVIES'


def test_manual_squares_inserts_one_stage(manual_session):
    """--manual-squares routes montage -> manual squares -> eucentricity.

    Worth pinning because this is the only state whose handler can skip the
    loop's polling delay once the squares arrive.
    """
    processor = TracingProcessor(working_directory=manual_session,
                                 microscope='Arctica-Apollo-SM',
                                 manual_squares=True)

    assert drive(processor)[:6] == [
        'STATE WAITING_FOR_MONTAGE',
        'MODULE grid_squares filtering=default microscope=Arctica-Apollo-SM',
        'STATE WAITING_FOR_MANUAL_SQUARES',
        'MODULE manual_squares',
        'STATE WAITING_FOR_GRID_SQUARES',
        'MODULE eucentricity',
    ]


def test_filtering_is_passed_through(session):
    """--filtering 64 reaches grid_squares."""
    processor = TracingProcessor(working_directory=session,
                                 microscope='Arctica-Apollo-SM',
                                 filtering='64')

    assert drive(processor)[1] == (
        'MODULE grid_squares filtering=64 microscope=Arctica-Apollo-SM'
    )
