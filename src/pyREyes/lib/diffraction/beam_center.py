"""Beam-stop-aware direct-beam centre estimation for diffraction frames.

Estimates where the unscattered beam meets the detector without assuming that
it is the brightest feature: after identifying the dark beam stop, the
approximately centrosymmetric (Friedel-symmetric) diffraction pattern is
registered against its own 180-degree rotation. Independent halo-circle and
stop-masked registrations provide guarded fallbacks for weak beams and
unusually wide stops.

Keep this module self-contained: the session-level logic pyREyes needs on top
of it (sampling, agreement between frames, fallbacks, sidecars) lives in
``beam_center_search.py``.

Depends on numpy, scipy and scikit-image, all already pyREyes dependencies.

Public coordinates are zero-based pixel-centre coordinates with ``x`` along
columns and ``y`` along rows.

Three boundaries:

* ``confidence`` is a diagnostic indicator, not a calibrated probability and
  not an error bound.
* Failure is raised (``BeamCenterEstimationError``), never silently replaced by
  the detector midpoint. The caller decides.
* A single-frame estimate is not trustworthy on its own. Estimate over several
  frames and look at the spread: ``beam_center_search.find_beam_centres``.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage, optimize
from skimage import measure, morphology, registration, transform


ALGORITHM_VERSION = "reyes-beam-center/v1"

# Fixed RANSAC seed, so the halo-circle fit gives the same answer on every run.
_RANSAC_SEED = 12345


class BeamCenterEstimationError(RuntimeError):
    """Raised when no validated estimator can locate a direct-beam centre."""


@dataclass(frozen=True)
class CenterCandidate:
    method: str
    x: float
    y: float
    score: float
    diagnostics: dict[str, float | int]


@dataclass(frozen=True)
class BeamCenterResult:
    x: float
    y: float
    method: str
    confidence: float
    candidates: tuple[CenterCandidate, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class _PreparedImage:
    values: np.ndarray
    valid: np.ndarray
    dark_stop: np.ndarray
    scale_x: float
    scale_y: float
    prior_x: float
    prior_y: float
    search_radius: float


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected one 2-D diffraction image, got shape {array.shape}")
    if min(array.shape) < 32:
        raise ValueError("Diffraction image must be at least 32 x 32 pixels")
    finite = np.isfinite(array)
    if finite.sum() < array.size * 0.5:
        raise ValueError("At least half of the image pixels must be finite")
    return array.astype(np.float32, copy=False)


def _prepare_image(
    image: np.ndarray,
    *,
    max_analysis_size: int,
    search_radius_fraction: float,
    initial_center: tuple[float, float] | None,
) -> _PreparedImage:
    array = _validate_image(image)
    height, width = array.shape
    target_scale = min(1.0, max_analysis_size / max(height, width))
    analysis_height = max(32, round(height * target_scale))
    analysis_width = max(32, round(width * target_scale))
    if (analysis_height, analysis_width) != array.shape:
        reduced = transform.resize(
            array,
            (analysis_height, analysis_width),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32, copy=False)
    else:
        reduced = array.copy()

    finite = np.isfinite(reduced)
    finite_values = reduced[finite]
    low, middle, high = np.percentile(finite_values, (0.5, 50.0, 99.9))
    if high - low <= max(abs(float(high)), abs(float(low)), 1.0) * 1e-8:
        raise BeamCenterEstimationError(
            "Diffraction image does not contain enough intensity variation"
        )
    dynamic_scale = max(float(middle - low), float(high - low) / 20.0, 1e-6)
    normalized = np.arcsinh(np.maximum(reduced - low, 0.0) / dynamic_scale)
    upper = float(np.percentile(normalized[finite], 99.8))
    if upper > 0:
        normalized = np.clip(normalized / upper, 0.0, 1.0)
    normalized[~finite] = 0.0

    scale_x = width / analysis_width
    scale_y = height / analysis_height
    if initial_center is None:
        prior_x = (analysis_width - 1) / 2.0
        prior_y = (analysis_height - 1) / 2.0
    else:
        prior_x = float(initial_center[0]) / scale_x
        prior_y = float(initial_center[1]) / scale_y

    minimum_dimension = min(analysis_height, analysis_width)
    search_radius = max(2.0, minimum_dimension * search_radius_fraction)
    yy, xx = np.indices(normalized.shape, dtype=np.float32)
    central = np.hypot(xx - prior_x, yy - prior_y) <= minimum_dimension * 0.48

    # Grow only dark pixels connected to the centre neighbourhood so naturally
    # dark detector corners are not mistaken for the stop or its support arm.
    raw_dark_threshold = float(np.percentile(normalized[central & finite], 3.0))
    raw_dark = (normalized <= raw_dark_threshold + 1e-6) & central & finite
    labels, _ = ndimage.label(raw_dark)
    neighbourhood = (
        np.hypot(xx - prior_x, yy - prior_y) <= minimum_dimension * 0.10
    )
    touching_labels = np.unique(labels[neighbourhood & (labels > 0)])
    dark_stop = (
        np.isin(labels, touching_labels)
        if touching_labels.size
        else np.zeros_like(raw_dark)
    )
    stop_growth = max(2, round(minimum_dimension * 0.012))
    dark_stop = morphology.dilation(dark_stop, morphology.disk(stop_growth))

    return _PreparedImage(
        values=normalized,
        valid=finite & central & ~dark_stop,
        dark_stop=dark_stop,
        scale_x=scale_x,
        scale_y=scale_y,
        prior_x=prior_x,
        prior_y=prior_y,
        search_radius=search_radius,
    )


def _bounded(center: tuple[float, float], prepared: _PreparedImage) -> bool:
    return (
        abs(center[0] - prepared.prior_x) <= prepared.search_radius
        and abs(center[1] - prepared.prior_y) <= prepared.search_radius
    )


def _circle_parameters(model: measure.CircleModel) -> tuple[float, float, float]:
    if hasattr(model, "center") and hasattr(model, "radius"):
        center = model.center
        return float(center[0]), float(center[1]), float(model.radius)
    x, y, radius = model.params
    return float(x), float(y), float(radius)


def _isophote_candidate(prepared: _PreparedImage) -> CenterCandidate | None:
    """Fit circles to bright-halo level sets while rejecting stop edges."""
    values = prepared.values
    minimum_dimension = min(values.shape)
    smooth = ndimage.gaussian_filter(
        values,
        sigma=max(1.5, minimum_dimension * 0.010),
    )
    yy, xx = np.indices(values.shape, dtype=np.float32)
    roi = (
        np.hypot(xx - prepared.prior_x, yy - prepared.prior_y)
        <= minimum_dimension * 0.42
    )
    roi_values = smooth[roi & prepared.valid]
    if roi_values.size < 100:
        return None

    estimates: list[tuple[float, float, float, float, float]] = []
    for percentile in (80.0, 86.0, 90.0, 93.0, 95.0):
        level = float(np.percentile(roi_values, percentile))
        contour_points = []
        for contour in measure.find_contours(smooth, level):
            radius_from_prior = np.hypot(
                contour[:, 1] - prepared.prior_x,
                contour[:, 0] - prepared.prior_y,
            )
            selected = contour[
                (radius_from_prior >= minimum_dimension * 0.03)
                & (radius_from_prior <= minimum_dimension * 0.40)
            ]
            if len(selected) >= 10:
                contour_points.append(selected[:, [1, 0]])
        if not contour_points:
            continue
        points = np.concatenate(contour_points)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                model, inliers = measure.ransac(
                    points,
                    measure.CircleModel,
                    min_samples=3,
                    residual_threshold=1.5,
                    max_trials=300,
                    rng=_RANSAC_SEED,
                )
        except (TypeError, ValueError):
            continue
        if model is None or inliers is None:
            continue
        x, y, radius = _circle_parameters(model)
        inlier_fraction = float(np.mean(inliers))
        if (
            not _bounded((x, y), prepared)
            or not minimum_dimension * 0.04 <= radius <= minimum_dimension * 0.40
            or inlier_fraction < 0.10
        ):
            continue
        inlier_points = points[inliers]
        refined = optimize.least_squares(
            lambda parameters, selected_points=inlier_points: (
                np.hypot(
                    selected_points[:, 0] - parameters[0],
                    selected_points[:, 1] - parameters[1],
                )
                - parameters[2]
            ),
            np.asarray([x, y, radius]),
            loss="soft_l1",
            f_scale=0.75,
            max_nfev=100,
        )
        x, y, radius = (float(item) for item in refined.x)
        if _bounded((x, y), prepared):
            estimates.append((x, y, radius, inlier_fraction, percentile))

    if not estimates:
        return None
    points = np.asarray([(x, y) for x, y, _, _, _ in estimates], dtype=np.float64)
    weights = np.asarray(
        [
            inliers * (percentile / 100.0) ** 4
            for _, _, _, inliers, percentile in estimates
        ]
    )
    center = np.average(points, axis=0, weights=weights)
    spread = float(np.median(np.linalg.norm(points - center, axis=1)))
    best_inlier_fraction = max(item[3] for item in estimates)
    stability = float(
        np.exp(-spread / max(1.0, minimum_dimension * 0.015))
    )
    score = float(np.clip(best_inlier_fraction * 1.5, 0.0, 1.0) * stability)
    return CenterCandidate(
        method="isophote",
        x=float(center[0]),
        y=float(center[1]),
        score=score,
        diagnostics={
            "levels": len(estimates),
            "level_spread_px": spread,
            "best_inlier_fraction": best_inlier_fraction,
        },
    )


def _fill_stop(values: np.ndarray, stop: np.ndarray) -> np.ndarray:
    if not stop.any() or stop.all():
        return values.copy()
    indices = ndimage.distance_transform_edt(
        stop,
        return_distances=False,
        return_indices=True,
    )
    return values[tuple(indices)].astype(np.float32, copy=False)


def _phase_inversion_candidate(
    prepared: _PreparedImage,
    pattern: np.ndarray,
) -> CenterCandidate | None:
    pattern = pattern * np.hanning(pattern.shape[0])[:, None]
    pattern = pattern * np.hanning(pattern.shape[1])[None, :]
    try:
        shift, error, _ = registration.phase_cross_correlation(
            pattern,
            pattern[::-1, ::-1],
            upsample_factor=20,
            normalization=None,
        )
    except (FloatingPointError, ValueError):
        return None
    y = (pattern.shape[0] - 1 + float(shift[0])) / 2.0
    x = (pattern.shape[1] - 1 + float(shift[1])) / 2.0
    if not np.all(np.isfinite((x, y, error))) or not _bounded((x, y), prepared):
        return None
    return CenterCandidate(
        method="inversion",
        x=x,
        y=y,
        score=float(np.clip(1.0 - error, 0.0, 1.0)),
        diagnostics={"phase_correlation_error": float(error)},
    )


def _inversion_candidate(prepared: _PreparedImage) -> CenterCandidate | None:
    values = _fill_stop(prepared.values, prepared.dark_stop)
    minimum_dimension = min(values.shape)
    low_pass = ndimage.gaussian_filter(
        values,
        sigma=max(1.0, minimum_dimension * 0.006),
    )
    broad = ndimage.gaussian_filter(
        low_pass,
        sigma=max(4.0, minimum_dimension * 0.08),
    )
    return _phase_inversion_candidate(prepared, low_pass - broad)


def _masked_inversion_candidate(prepared: _PreparedImage) -> CenterCandidate | None:
    """Register inversion symmetry while omitting the stop from both images."""
    minimum_dimension = min(prepared.values.shape)
    low_pass = ndimage.gaussian_filter(
        prepared.values,
        sigma=max(1.0, minimum_dimension * 0.006),
    )
    broad = ndimage.gaussian_filter(
        low_pass,
        sigma=max(4.0, minimum_dimension * 0.08),
    )
    pattern = low_pass - broad
    try:
        shift, _, _ = registration.phase_cross_correlation(
            pattern,
            pattern[::-1, ::-1],
            reference_mask=prepared.valid,
            moving_mask=prepared.valid[::-1, ::-1],
            overlap_ratio=0.30,
        )
    except (FloatingPointError, ValueError):
        return None
    y = (pattern.shape[0] - 1 + float(shift[0])) / 2.0
    x = (pattern.shape[1] - 1 + float(shift[1])) / 2.0
    if not np.all(np.isfinite((x, y))) or not _bounded((x, y), prepared):
        return None

    integer_shift = np.rint(shift).astype(int)
    shifted_pattern = np.roll(pattern[::-1, ::-1], integer_shift, axis=(0, 1))
    shifted_valid = np.roll(
        prepared.valid[::-1, ::-1],
        integer_shift,
        axis=(0, 1),
    )
    overlap = prepared.valid & shifted_valid
    delta_y, delta_x = (int(value) for value in integer_shift)
    if delta_y > 0:
        overlap[:delta_y, :] = False
    elif delta_y < 0:
        overlap[delta_y:, :] = False
    if delta_x > 0:
        overlap[:, :delta_x] = False
    elif delta_x < 0:
        overlap[:, delta_x:] = False
    if overlap.sum() < 100:
        return None
    reference_samples = pattern[overlap].astype(np.float64)
    moving_samples = shifted_pattern[overlap].astype(np.float64)
    correlation = float(np.corrcoef(reference_samples, moving_samples)[0, 1])
    if not np.isfinite(correlation):
        return None
    return CenterCandidate(
        method="masked-inversion",
        x=x,
        y=y,
        score=float(np.clip(correlation, 0.0, 1.0)),
        diagnostics={
            "masked_correlation": correlation,
            "overlap_pixels": int(overlap.sum()),
        },
    )


def find_beam_center(
    image: np.ndarray,
    *,
    initial_center: tuple[float, float] | None = None,
    search_radius_fraction: float = 0.12,
    max_analysis_size: int = 512,
) -> BeamCenterResult:
    """Estimate the direct-beam centre in one diffraction image."""
    if not 0.01 <= search_radius_fraction <= 0.45:
        raise ValueError("search_radius_fraction must be between 0.01 and 0.45")
    if max_analysis_size < 64:
        raise ValueError("max_analysis_size must be at least 64")

    prepared = _prepare_image(
        image,
        max_analysis_size=max_analysis_size,
        search_radius_fraction=search_radius_fraction,
        initial_center=initial_center,
    )
    isophote = _isophote_candidate(prepared)
    inversion = _inversion_candidate(prepared)

    use_isophote = isophote is not None and (
        inversion is None or (inversion.score < 0.80 and isophote.score > 0.90)
    )
    masked_inversion = (
        _masked_inversion_candidate(prepared)
        if not use_isophote and (inversion is None or inversion.score < 0.70)
        else None
    )
    use_masked = (
        not use_isophote
        and masked_inversion is not None
        and (
            inversion is None
            or (inversion.score < 0.70 and masked_inversion.score > 0.70)
        )
    )
    chosen = (
        isophote
        if use_isophote
        else masked_inversion
        if use_masked
        else inversion
    )
    if chosen is None:
        raise BeamCenterEstimationError(
            "No direct-beam estimator produced a valid centre"
        )
    result_method = (
        "isophote-fallback"
        if use_isophote
        else "masked-inversion-fallback"
        if use_masked
        else "inversion"
    )

    agreement = 0.0
    if isophote is not None:
        distance = float(
            np.hypot(isophote.x - chosen.x, isophote.y - chosen.y)
        )
        agreement = float(
            np.exp(-distance / max(1.0, min(prepared.values.shape) * 0.03))
        )
    confidence = float(
        np.clip(0.75 * chosen.score + 0.25 * agreement, 0.0, 1.0)
    )

    candidates = tuple(
        CenterCandidate(
            method=candidate.method,
            x=candidate.x * prepared.scale_x,
            y=candidate.y * prepared.scale_y,
            score=candidate.score,
            diagnostics={
                **candidate.diagnostics,
                "analysis_x": candidate.x,
                "analysis_y": candidate.y,
            },
        )
        for candidate in (isophote, inversion, masked_inversion)
        if candidate is not None
    )
    return BeamCenterResult(
        x=chosen.x * prepared.scale_x,
        y=chosen.y * prepared.scale_y,
        method=result_method,
        confidence=confidence,
        candidates=candidates,
    )
