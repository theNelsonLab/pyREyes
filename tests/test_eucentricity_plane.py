"""
The eucentric plane, and the JSON that records it.

Two separate things are checked here, because they fail for different reasons:

  * that the regression still recovers a known plane, and now carries its own
    fit quality out with it;
  * that `trusted` gates on the fit against the raw fine-eucentricity points,
    not the tautological R^2 = 1 of a refit from a navigator whose Z values
    were computed from the plane.
"""

import json
import math

import pytest

from pyREyes.eucentricity import (
    PLANE_MIN_POINTS,
    PLANE_MIN_R2,
    RegressionResult,
    perform_regression,
    write_plane_json,
)
from pyREyes.lib.REyes_errors import ProcessingError


# A gently tilted plane, in microns.
PLANE_A, PLANE_B, PLANE_C = 0.004089, 0.007730, 49.7668


def points_on_plane(xy):
    return [[x, y, PLANE_A * x + PLANE_B * y + PLANE_C] for x, y in xy]


FIVE_POINTS = [(-400.0, -300.0), (-100.0, 250.0), (0.0, 0.0),
               (250.0, -150.0), (450.0, 380.0)]


def test_recovers_a_known_plane():
    result = perform_regression(points_on_plane(FIVE_POINTS))

    assert result.a == pytest.approx(PLANE_A, abs=1e-9)
    assert result.b == pytest.approx(PLANE_B, abs=1e-9)
    assert result.c == pytest.approx(PLANE_C, abs=1e-6)
    assert result.r2 == pytest.approx(1.0)
    assert result.n_points == 5
    assert result.trusted
    assert result.covariance is not None and len(result.covariance) == 3
    assert all(se is not None and se >= 0 for se in (result.se_a, result.se_b, result.se_c))


def test_three_points_fails_loudly():
    """Three points and three parameters leaves no residual degrees of freedom.

    Pre-existing behaviour, pinned here rather than changed: the residual
    variance divides by (n - 3), the standard errors come back as infinity,
    and formatting them raises, which perform_regression re-raises as
    ProcessingError. A session with three or fewer fine-eucentricity
    measurements fails the eucentricity stage outright rather than quietly
    producing a plane. PLANE_MIN_POINTS is therefore defensive: nothing with
    n < 4 reaches the trust gate through this function.
    """
    with pytest.raises(ProcessingError):
        perform_regression(points_on_plane([(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)]))


def test_standard_errors_are_never_reported_as_infinity(tmp_path):
    """JSON has no way to say inf, and a reader that found one would take the
    plane as infinitely uncertain rather than unmeasured. `allow_nan=False`
    makes that a write error instead of a silently invalid file."""
    result = RegressionResult(a=0.0, b=0.0, c=50.0, df=None,
                              se_a=float('inf'), r2=0.9, n_points=8)

    with pytest.raises(ValueError):
        write_plane_json(result, str(tmp_path))


@pytest.mark.parametrize('r2, n_points, expected', [
    (1.0, 16, True),
    (0.77, 5, True),          # a typical fit on five measurements: usable
    (PLANE_MIN_R2, 4, True),  # exactly on both thresholds
    (0.64, 16, False),        # good coverage, but the points are not planar
    (0.99, 3, False),         # perfect fit, but only because n == parameters
    (None, 16, False),
])
def test_trust_gate(r2, n_points, expected):
    result = RegressionResult(a=0.0, b=0.0, c=0.0, df=None,
                              r2=r2, n_points=n_points)
    assert result.trusted is expected


def test_writes_json_a_reader_can_act_on(tmp_path):
    path = write_plane_json(perform_regression(points_on_plane(FIVE_POINTS)),
                            str(tmp_path))

    plane = json.loads(open(path).read())

    assert path == str(tmp_path / 'eucentricity_plane.json')
    assert plane['trusted'] is True
    assert plane['n_points'] == 5
    assert plane['min_r2'] == PLANE_MIN_R2
    assert plane['min_points'] == PLANE_MIN_POINTS
    assert plane['equation'] == 'z = a*x + b*y + c'
    # Full precision, not the three decimals the log prints.
    assert plane['a'] == pytest.approx(PLANE_A, abs=1e-9)
    for key in ('a', 'b', 'c', 'r2'):
        assert math.isfinite(plane[key])
    assert len(plane['covariance']) == 3 and len(plane['covariance'][0]) == 3


def test_untrusted_plane_is_still_written(tmp_path):
    """A bad plane is recorded with trusted=False rather than withheld: a
    reader needs to know the fit happened and was rejected, and an absent
    file is indistinguishable from eucentricity never having run."""
    result = RegressionResult(a=1.0, b=2.0, c=3.0, df=None, r2=0.1, n_points=12)

    plane = json.loads(open(write_plane_json(result, str(tmp_path))).read())

    assert plane['trusted'] is False
    assert plane['a'] == 1.0
    assert plane['se_a'] is None and plane['covariance'] is None


def test_the_stage_writes_the_plane_beside_the_navigator(tmp_path, monkeypatch):
    """eucentricity-1 writes grid_squares/eucentricity_plane.json."""
    import pyREyes.eucentricity as euc

    monkeypatch.chdir(tmp_path)
    (tmp_path / 'grid_squares').mkdir()
    with open(tmp_path / 'g_grid_squares.log', 'w') as log:
        for x, y, z in points_on_plane(FIVE_POINTS):
            log.write(f'Stage {x:.3f} {y:.3f} {z:.3f}\n')
        log.write('Current number of items is 5\n')
    with open(tmp_path / 'grid_squares' / 'grid_squares.nav', 'w') as nav:
        nav.write('AdocVersion = 2.00\n\n')
        for i, (x, y) in enumerate(FIVE_POINTS, 1):
            nav.write(f'[Item = {i}]\nStageXYZ = {x:.3f} {y:.3f} 0.000\n\n')
    monkeypatch.setattr(euc, 'plot_results', lambda *a, **k: None)

    assert euc.main() == 0
    plane = json.loads((tmp_path / 'grid_squares' / 'eucentricity_plane.json').read_text())
    assert plane['trusted'] is True and plane['n_points'] == 5
