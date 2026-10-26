"""
Where the snapshot rings are drawn: about the direct beam, not the frame centre.

Three things are pinned here.

The estimator itself, on a synthetic centrosymmetric pattern with and without a
beam stop: if a change to it moves what it returns, this is where that shows.

The session rule -- the median of the frames that AGREE -- is what stands between
the estimator's bad single-frame answers and a picture. An empty frame still
returns a position, anywhere on the detector, so the rule is tested with exactly
that: good frames, plus answers from nowhere.

And the fallback is never silent. When no centre can be trusted the rings go
back to the middle of the frame, with the reason carried to the picture and the
sidecar; `frame` mode draws what 3.4.10 and earlier drew.
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pyREyes.lib.diffraction import beam_center_search as bcs
from pyREyes.lib.diffraction.beam_center import (
    BeamCenterEstimationError, find_beam_center)


# --------------------------------------------------------------------------- #
# Synthetic patterns
# --------------------------------------------------------------------------- #
def pattern(shape=(256, 320), centre=(153.4, 127.2), beam_stop=True, seed=12345,
            scale=1.0):
    """A centrosymmetric pattern about `centre`: two halo rings and Friedel pairs.

    `scale` grows the features with the frame so a larger frame is the same
    picture, not a sparser one.
    """
    height, width = shape
    cx, cy = centre
    yy, xx = np.indices(shape, dtype=np.float32)
    radius = np.hypot(xx - cx, yy - cy)
    rng = np.random.default_rng(seed)
    image = (100
             + 50 * np.exp(-((radius - 42 * scale) / (3 * scale)) ** 2)
             + 35 * np.exp(-((radius - 76 * scale) / (4 * scale)) ** 2))
    for _ in range(35):
        x = float(rng.uniform(25 * scale, width - 25 * scale))
        y = float(rng.uniform(25 * scale, height - 25 * scale))
        amplitude = float(rng.uniform(50, 150))
        sigma = float(rng.uniform(1.5, 3.0)) * scale
        for px, py in ((x, y), (2 * cx - x, 2 * cy - y)):
            image += amplitude * np.exp(
                -((xx - px) ** 2 + (yy - py) ** 2) / (2 * sigma ** 2))
    image += rng.normal(0, 1.5, image.shape)
    if beam_stop:
        stopped = (radius < 18 * scale) | ((xx > cx) & (np.abs(yy - cy) < 5 * scale))
        image[stopped] = 5 + rng.normal(0, 0.2, int(stopped.sum()))
    return image.astype(np.float32)


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('beam_stop', [False, True])
def test_estimator_finds_subpixel_centre_with_or_without_beam_stop(beam_stop):
    result = find_beam_center(pattern(beam_stop=beam_stop))

    assert result.x == pytest.approx(153.4, abs=0.75)
    assert result.y == pytest.approx(127.2, abs=0.75)
    assert result.method in {'inversion', 'isophote-fallback',
                             'masked-inversion-fallback'}
    assert 0 <= result.confidence <= 1


def test_estimator_refuses_what_is_not_a_pattern():
    with pytest.raises(ValueError, match='2-D diffraction image'):
        find_beam_center(np.zeros((4, 32, 32), dtype=np.float32))
    with pytest.raises(BeamCenterEstimationError, match='intensity variation'):
        find_beam_center(np.zeros((64, 64), dtype=np.float32))


# --------------------------------------------------------------------------- #
# One frame
# --------------------------------------------------------------------------- #
def test_estimate_completes_the_pixel_centre_mapping_on_a_reduced_frame():
    """A 1024 frame is analysed at 512 and comes back half a pixel low.

    The estimator scales its answer by the reduction factor alone; the
    wrapper adds the half reduced pixel back. Without it the error below is
    0.5 px on each axis, and 1.5 px on a 2048 frame.
    """
    frame = pattern(shape=(1024, 1024), centre=(471.3, 530.6), scale=4.0)

    raw = find_beam_center(frame)
    x, y, _ = bcs.estimate_frame(frame)

    assert (x - raw.x, y - raw.y) == pytest.approx((0.5, 0.5))
    assert x == pytest.approx(471.3, abs=0.3)
    assert y == pytest.approx(530.6, abs=0.3)


def test_estimate_leaves_a_frame_that_is_not_reduced_alone():
    frame = pattern()
    raw = find_beam_center(frame)
    assert bcs.estimate_frame(frame)[:2] == pytest.approx((raw.x, raw.y))


def test_estimate_returns_none_rather_than_raising():
    assert bcs.estimate_frame(np.zeros((64, 64), dtype=np.float32)) is None
    assert bcs.estimate_frame(np.zeros((3, 64, 64), dtype=np.float32)) is None


# --------------------------------------------------------------------------- #
# The session rule
# --------------------------------------------------------------------------- #
def test_the_centre_is_the_median_of_the_frames_that_agree():
    """Six frames on the beam and two answers from nowhere, as empty frames give."""
    good = [(913.5, 1027.5, 'masked-inversion-fallback'),
            (915.5, 1025.5, 'masked-inversion-fallback'),
            (911.5, 1029.5, 'masked-inversion-fallback'),
            (917.3, 1027.9, 'inversion'),
            (914.2, 1025.8, 'isophote-fallback'),
            (913.5, 1025.5, 'masked-inversion-fallback')]
    stray = [(1049.8, 938.1, 'inversion'), (1250.0, 700.0, 'inversion')]

    centre = bcs.summarise(good + stray, (2048, 2048), n_frames=10)

    assert centre.source == bcs.SOURCE_DETECTED and centre.trusted
    assert (centre.n_frames, centre.n_ok, centre.n_agree) == (10, 8, 6)
    assert (centre.x, centre.y) == pytest.approx((913.85, 1026.65))
    assert centre.spread_px < 5
    assert centre.methods['masked-inversion-fallback'] == 4


def test_frames_that_do_not_agree_are_not_trusted():
    scattered = [(900.0, 1000.0, 'inversion'), (1100.0, 950.0, 'inversion'),
                 (980.0, 1200.0, 'inversion'), (1010.0, 1010.0, 'inversion'),
                 (760.0, 1105.0, 'inversion')]
    centre = bcs.summarise(scattered, (2048, 2048), n_frames=5)
    assert not centre.trusted
    assert 'agree' in centre.note


def test_two_frames_cannot_outvote_a_bad_one():
    centre = bcs.summarise([(913.0, 1027.0, 'inversion'),
                            (914.0, 1026.0, 'inversion')], (2048, 2048), n_frames=2)
    assert not centre.trusted


def test_agreement_radius_scales_with_the_frame():
    """12 px apart agrees on a 2048 frame and does not on a 512 one."""
    estimates = [(250.0, 250.0, 'inversion'), (262.0, 250.0, 'inversion'),
                 (238.0, 250.0, 'inversion')]
    assert bcs.summarise(estimates, (2048, 2048), 3).trusted
    assert not bcs.summarise(estimates, (512, 512), 3).trusted


def test_no_estimate_at_all_is_the_frame_centre_and_says_why():
    centre = bcs.summarise([], (1024, 1024), n_frames=4)
    assert centre.source == bcs.SOURCE_FRAME and not centre.measured
    assert (centre.x, centre.y) == (512, 512)
    assert 'not found in 4 frame' in centre.note


def test_centres_are_kept_apart_by_frame_size():
    small = pattern()
    large = pattern(shape=(512, 640), centre=(306.8, 254.4), scale=2.0)
    centres = bcs.find_beam_centres([small] * 3 + [large] * 3)

    assert set(centres) == {(256, 320), (512, 640)}
    assert centres[(256, 320)].x == pytest.approx(153.4, abs=0.75)
    assert centres[(512, 640)].x == pytest.approx(306.8, abs=0.75)
    assert all(c.trusted for c in centres.values())


def test_an_untrusted_search_falls_back_to_the_frame_centre_with_the_reason():
    """One frame is not enough to trust, however good it looks."""
    centres = bcs.find_beam_centres([pattern()])
    centre = centres[(256, 320)]
    assert centre.source == bcs.SOURCE_FRAME
    assert (centre.x, centre.y) == (160, 128)
    assert 'not trusted' in centre.note


def test_blank_frames_fall_back_to_the_frame_centre():
    centres = bcs.find_beam_centres([np.zeros((128, 128), np.float32)] * 4)
    assert centres[(128, 128)].source == bcs.SOURCE_FRAME
    assert 'not found' in centres[(128, 128)].note


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def test_parse_mode():
    assert bcs.parse_mode('auto') == bcs.MODE_AUTO
    assert bcs.parse_mode('Frame') == bcs.MODE_FRAME
    assert bcs.parse_mode('913.5,1027.5') == (913.5, 1027.5)
    with pytest.raises(ValueError):
        bcs.parse_mode('middle')
    with pytest.raises(ValueError):
        bcs.parse_mode('913.5')


def test_resolve_follows_the_mode():
    detected = {(2048, 2048): bcs.summarise(
        [(913.0, 1027.0, 'inversion')] * 3, (2048, 2048), 3)}

    assert bcs.resolve(bcs.MODE_AUTO, (2048, 2048), detected).x == 913.0
    assert bcs.resolve(bcs.MODE_FRAME, (2048, 2048), detected).x == 1024
    given = bcs.resolve((900.0, 1000.0), (2048, 2048), detected)
    assert (given.x, given.y, given.source) == (900.0, 1000.0, bcs.SOURCE_GIVEN)

    # a size that was never searched is not given another size's centre
    other = bcs.resolve(bcs.MODE_AUTO, (1024, 1024), detected)
    assert other.source == bcs.SOURCE_FRAME and other.note


def test_frame_centre_is_the_geometry_the_rings_always_had():
    centre = bcs.frame_centre((2048, 2048))
    assert (centre.x, centre.y, centre.max_ring_radius()) == (1024, 1024, 1024)


def test_rings_about_a_measured_centre_stay_inside_the_frame():
    centre = bcs.given_centre((913.5, 1027.5), (2048, 2048))
    assert centre.max_ring_radius() == pytest.approx(913.5)


def test_sidecar_records_what_the_rings_were_drawn_about(tmp_path):
    centre = bcs.summarise([(913.0, 1027.0, 'inversion')] * 3, (2048, 2048), 3)
    path = bcs.write_sidecar(str(tmp_path), bcs.MODE_AUTO, [centre])

    record = json.loads(open(path).read())
    assert os.path.basename(path) == bcs.SIDECAR
    assert record['mode'] == 'auto'
    assert record['centres'][0]['x'] == 913.0
    assert record['centres'][0]['shape'] == [2048, 2048]
    assert record['centres'][0]['trusted'] is True
    assert not os.path.exists(path + '.tmp')


# --------------------------------------------------------------------------- #
# The pictures
# --------------------------------------------------------------------------- #
def ring_centres(ax):
    return {tuple(float(v) for v in patch.center) for patch in ax.patches}


def test_a_fallback_is_written_on_the_picture():
    fig, ax = plt.subplots()
    ax.imshow(np.zeros((64, 64)))
    bcs.annotate(ax, bcs.frame_centre((64, 64), note='beam centre not found in 3 frame(s)'))
    assert any('rings on frame centre' in t.get_text() for t in ax.texts)
    assert not ax.lines

    bcs.annotate(ax, bcs.given_centre((30.0, 31.0), (64, 64)))
    assert len(ax.lines) == 1 and ax.get_xlim() == (-0.5, 63.5)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Step-scan snapshots
# --------------------------------------------------------------------------- #
@pytest.fixture
def snapshot_processor(monkeypatch):
    """A DiffractionDataProcessor whose MRC reads come from a dict of arrays."""
    import pyREyes.dif_map as dm
    from pyREyes.lib.diffraction.DiffractionDataProcessor import (
        DiffractionDataProcessor)

    arrays = {}
    monkeypatch.setattr(dm, '_read_mrc', lambda path: arrays[path])
    rings = []
    monkeypatch.setattr(
        DiffractionDataProcessor, '_add_resolution_rings',
        lambda self, ax, cx, cy, max_radius, pixel_size: rings.append((cx, cy, max_radius)))

    def make(beam_center=bcs.MODE_AUTO):
        return DiffractionDataProcessor('Arctica-Apollo-SM', beam_center=beam_center)
    return make, arrays, rings


def test_snapshots_share_one_centre_found_before_plotting(snapshot_processor, tmp_path):
    make, arrays, rings = snapshot_processor
    for i in range(4):
        arrays[f'f{i}.mrc'] = pattern(seed=200 + i)
    processor = make()

    processor.find_beam_center(list(arrays), str(tmp_path))
    for name in arrays:
        processor.plot_diffraction_snapshot(name, str(tmp_path / (name + '.png')), 1, 10, 100)

    assert len(rings) == 4 and len(set(rings)) == 1
    assert rings[0][0] == pytest.approx(153.4, abs=0.75)     # x is the column
    assert rings[0][1] == pytest.approx(127.2, abs=0.75)
    assert json.loads((tmp_path / bcs.SIDECAR).read_text())['centres'][0]['trusted']
    assert (tmp_path / 'f0.mrc.png').exists()


def test_frame_mode_draws_about_the_middle(snapshot_processor, tmp_path):
    make, arrays, rings = snapshot_processor
    arrays['f.mrc'] = pattern()
    processor = make(bcs.MODE_FRAME)

    processor.find_beam_center(['f.mrc'], str(tmp_path))
    processor.plot_diffraction_snapshot('f.mrc', str(tmp_path / 'f.png'), 1, 10, 100)

    assert rings == [(160, 128, 128)]
    assert not (tmp_path / bcs.SIDECAR).exists()


def test_snapshot_of_a_frame_stack_is_plotted(snapshot_processor, tmp_path):
    """A two-sub-frame MRC used to reach imshow as a 3-D array and lose its picture."""
    make, arrays, rings = snapshot_processor
    half = pattern() / 2
    arrays['stack.mrc'] = np.stack([half, half])
    processor = make(bcs.MODE_FRAME)

    processor.plot_diffraction_snapshot('stack.mrc', str(tmp_path / 's.png'), 1, 10, 100)

    assert (tmp_path / 's.png').exists()
    assert len(rings) == 1


def test_append_targets_rings_follow_the_centre():
    from pyREyes import append_targets_nav as sta

    config = sta.MICROSCOPE_CONFIGS['Arctica-Apollo-SM']
    frame = pattern()

    fig = sta.plot_diffraction_pattern(frame, config, 10, 100, 'f.mrc')
    assert ring_centres(fig.axes[0]) == {(160.0, 128.0)}       # the middle of the frame
    plt.close(fig)

    centre = bcs.given_centre((153.4, 127.2), frame.shape)
    fig = sta.plot_diffraction_pattern(frame, config, 10, 100, 'f.mrc', centre)
    assert ring_centres(fig.axes[0]) == {(153.4, 127.2)}
    plt.close(fig)
