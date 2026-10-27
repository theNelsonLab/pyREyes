"""
Where the direct beam is on the detector, for the diffraction snapshots.

Up to 3.4.10 every snapshot drew its resolution rings about the middle of the
frame. The beam is rarely there: it sits wherever the diffraction alignment
left it, often tens of pixels away on a 2k frame, so each ring was labelled
with the resolution of a different circle. This module finds the centre from
the frames themselves and hands the plotting code a position to draw about.

ONE CENTRE PER SESSION, NOT PER FRAME
-------------------------------------
The estimator (`beam_center.find_beam_center`) is not to be believed on a
single frame: a frame with no pattern still returns a position, and the
estimator's `confidence` does not predict its error. What does separate a good
answer from a bad one is whether several frames AGREE. The beam does not move
between the frames of a session, so the frames about to be plotted are
sampled, each is estimated on its own, and the centre is the median of the
frames that agree. How many agree is the verdict. A centre is never carried
from one frame size to another.

WHAT HAPPENS WHEN IT CANNOT BE FOUND
------------------------------------
The rings go back to the middle of the frame and THE PICTURE SAYS SO. A ring
drawn about a guessed centre looks exactly like one drawn about a measured
centre, so the fallback is written on the snapshot, in the log and in the
sidecar; it is never silent. What WAS measured is kept in the sidecar
(`rejected`, `n_frames`, `n_agree`, ...), so an untrusted search can still be
audited.

Before falling back to the frame centre, a caller may offer a SESSION centre
(`sibling_centre`, via `find_beam_centres(fallback=session_fallback(...))`):
the trusted, detected centre of a sibling session in the same parent
directory, on frames of the SAME SHAPE: a session with only one or two
patterns has too few frames to outvote a bad answer on its own, and the beam
does not move between sessions collected together. A borrowed centre is labelled as
such on the picture, in the log and in the sidecar (`source = "session"`,
`borrowed_from`). A centre is never rescaled from one frame size to another.

THIS MOVES PICTURES, NOT NUMBERS
--------------------------------
Nothing here feeds a metric. Peak counts and quality classes keep their own
frame-centred exclusion zones, and no navigator, CSV or target selection
changes.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from pyREyes.lib.REyes_logging import log_print

__version__ = '3.6.1'

# How the centre of a snapshot is chosen.
MODE_AUTO = 'auto'      # search the frames; fall back to the frame centre, visibly
MODE_FRAME = 'frame'    # the middle of the frame: what 3.4.10 and earlier drew

SOURCE_DETECTED = 'detected'
SOURCE_GIVEN = 'given'
SOURCE_FRAME = 'frame'
SOURCE_SESSION = 'session'   # borrowed from a sibling session (same parent directory)

# Where a session's sidecars live, relative to its directory. Searched for a
# sibling's trusted centre (`sibling_centre`, `session_fallback`).
SIDECAR_LOCATIONS = (
    'targets/beam_center.json',
)

# Frames estimated per frame size. Each costs 0.2 to 0.4 s.
MAX_FRAMES = 25

# The verdict. Frames within AGREEMENT_PX_2K of the median agree with it; the
# centre is trusted when at least MIN_AGREEING of them do and they are at least
# MIN_AGREEING_FRACTION of the frames that gave an answer. The radius is in
# pixels of a 2048 frame and scales with the frame. Three is the fewest that
# can outvote one bad answer.
MIN_AGREEING = 3
MIN_AGREEING_FRACTION = 0.5
AGREEMENT_PX_2K = 15.0
REFERENCE_SIZE = 2048

SIDECAR = 'beam_center.json'

Shape = Tuple[int, int]
Mode = Union[str, Tuple[float, float]]


@dataclass(frozen=True)
class BeamCentre:
    """A position to draw the rings about, and how far to believe it.

    `x` runs along columns and `y` along rows, in pixels of a frame of `shape`,
    which is how matplotlib addresses an image shown with `imshow`.
    """
    x: float
    y: float
    shape: Shape
    source: str
    trusted: bool = False
    n_frames: int = 0
    n_ok: int = 0
    n_agree: int = 0
    spread_px: Optional[float] = None
    methods: Dict[str, int] = field(default_factory=dict)
    note: str = ''
    # An untrusted search's own answer, kept so the sidecar records what was
    # measured even when the rings are not drawn about it.
    rejected: Optional[Dict[str, float]] = None
    # For SOURCE_SESSION: the sibling session(s) the centre was borrowed from.
    borrowed_from: List[str] = field(default_factory=list)

    @property
    def measured(self) -> bool:
        """False when the rings are on the frame centre for want of anything better."""
        return self.source != SOURCE_FRAME

    def max_ring_radius(self) -> float:
        """The largest ring that stays inside the frame."""
        height, width = self.shape
        if self.source == SOURCE_FRAME:
            # exactly what the frame-centred rings always used
            return min(self.x, self.y)
        return max(1.0, min(self.x, self.y, width - 1 - self.x, height - 1 - self.y))

    def describe(self) -> str:
        height, width = self.shape
        where = f"({self.x:.1f}, {self.y:.1f}) px on {width}x{height}"
        if self.source == SOURCE_FRAME:
            return f"frame centre {where}" + (f" -- {self.note}" if self.note else '')
        if self.source == SOURCE_GIVEN:
            return f"given {where}"
        if self.source == SOURCE_SESSION:
            return (f"session centre {where}, borrowed from sibling session(s) "
                    f"{', '.join(self.borrowed_from)}")
        offset = np.hypot(self.x - (width - 1) / 2.0, self.y - (height - 1) / 2.0)
        return (f"detected {where}, {offset:.1f} px from the frame centre; "
                f"{self.n_agree} of {self.n_frames} frames agree to {self.spread_px:.1f} px")


def frame_centre(shape: Shape, note: str = '',
                 measured: Optional[BeamCentre] = None) -> BeamCentre:
    """The middle of the frame, as the rings were drawn before the centre was looked for.

    `measured` is an untrusted search result; its counts and its own answer are
    carried along (as `rejected`) so the sidecar still says what was measured.
    """
    height, width = int(shape[0]), int(shape[1])
    if measured is None:
        return BeamCentre(x=width / 2, y=height / 2, shape=(height, width),
                          source=SOURCE_FRAME, note=note)
    return BeamCentre(x=width / 2, y=height / 2, shape=(height, width),
                      source=SOURCE_FRAME, note=note,
                      n_frames=measured.n_frames, n_ok=measured.n_ok,
                      n_agree=measured.n_agree, spread_px=measured.spread_px,
                      methods=dict(measured.methods),
                      rejected=_rejected_of(measured))


def _rejected_of(found: BeamCentre) -> Optional[Dict[str, float]]:
    if found.source == SOURCE_DETECTED:
        return {'x': found.x, 'y': found.y}
    return found.rejected


def given_centre(xy: Tuple[float, float], shape: Shape) -> BeamCentre:
    height, width = int(shape[0]), int(shape[1])
    return BeamCentre(x=float(xy[0]), y=float(xy[1]), shape=(height, width),
                      source=SOURCE_GIVEN, trusted=True)


def parse_mode(text: str) -> Mode:
    """`auto`, `frame`, or `X,Y` in pixels of the plotted frame. For argparse."""
    value = str(text).strip().lower()
    if value in (MODE_AUTO, MODE_FRAME):
        return value
    try:
        x, y = (float(part) for part in value.split(','))
    except ValueError:
        raise ValueError(
            f"beam centre must be 'auto', 'frame' or X,Y in pixels, not {text!r}")
    return (x, y)


ARGUMENT_HELP = (
    "where the resolution rings are centred: 'auto' finds the direct beam from "
    "the frames being plotted (default), 'frame' uses the middle of the frame as "
    "3.4.10 and earlier did, or X,Y gives it in pixels of the plotted frame "
    "(x along columns, from zero)")


def sample(items: Sequence, limit: int = MAX_FRAMES) -> List:
    """Up to `limit` items, evenly spread and in order."""
    items = list(items)
    if len(items) <= limit:
        return items
    picks = sorted({int(round(v)) for v in np.linspace(0, len(items) - 1, limit)})
    return [items[i] for i in picks]


def estimate_frame(frame: np.ndarray) -> Optional[Tuple[float, float, str]]:
    """One frame's centre as (x, y, method), or None if the estimator gave up.

    The estimator works on a copy reduced to 512 px and scales its answer back
    by the reduction factor alone, which places it half a reduced pixel low:
    1.5 px on a 2048 frame, 3.5 px on 4096, nothing at 512. The pixel-centre
    mapping is completed here, so the estimator itself stays unchanged.
    """
    from pyREyes.lib.diffraction.beam_center import (
        BeamCenterEstimationError, find_beam_center)

    frame = np.asarray(frame)
    if frame.ndim != 2:
        return None
    try:
        result = find_beam_center(frame)
    except (BeamCenterEstimationError, ValueError):
        return None
    except Exception as exc:
        # Anything else is a fault in the estimator, and it is still only a
        # picture: the stage that draws it also publishes the target
        # candidates, and must not be stopped by where a ring goes.
        log_print(f"  beam centre: estimator failed on a frame "
                  f"({type(exc).__name__}: {exc})", logging.WARNING)
        return None

    height, width = frame.shape
    reduction = min(1.0, 512 / max(height, width))
    scale_x = width / max(32, int(round(width * reduction)))
    scale_y = height / max(32, int(round(height * reduction)))
    return (result.x + (scale_x - 1) / 2.0,
            result.y + (scale_y - 1) / 2.0,
            result.method)


def summarise(estimates: Sequence[Tuple[float, float, str]], shape: Shape,
              n_frames: int) -> BeamCentre:
    """The median of the frames that agree, and whether enough of them do."""
    height, width = int(shape[0]), int(shape[1])
    if not estimates:
        return frame_centre(shape, note=f"beam centre not found in {n_frames} frame(s)")

    xs = np.array([e[0] for e in estimates], dtype=float)
    ys = np.array([e[1] for e in estimates], dtype=float)
    radius = AGREEMENT_PX_2K * max(height, width) / REFERENCE_SIZE
    agree = np.hypot(xs - np.median(xs), ys - np.median(ys)) <= radius
    n_agree = int(agree.sum())

    # the answer is taken from the frames that agree, not dragged by the rest;
    # when none do there is no answer, and the spread of all of them says so
    kept = agree if n_agree else np.ones_like(agree)
    x, y = float(np.median(xs[kept])), float(np.median(ys[kept]))
    spread = float(np.max(np.hypot(xs[kept] - x, ys[kept] - y)))

    methods: Dict[str, int] = {}
    for _, _, method in estimates:
        methods[method] = methods.get(method, 0) + 1

    reasons = []
    if n_agree < MIN_AGREEING:
        reasons.append(f"only {n_agree} frame(s) agree on a centre")
    elif n_agree < MIN_AGREEING_FRACTION * len(estimates):
        reasons.append(f"only {n_agree} of {len(estimates)} frames agree "
                       f"to {radius:.0f} px")

    return BeamCentre(x=x, y=y, shape=(height, width), source=SOURCE_DETECTED,
                      trusted=not reasons, n_frames=n_frames, n_ok=len(estimates),
                      n_agree=n_agree, spread_px=spread, methods=methods,
                      note='; '.join(reasons))


def find_beam_centres(frames: Iterable[np.ndarray],
                      fallback: Optional[Callable[[Shape], Optional[BeamCentre]]] = None,
                      ) -> Dict[Shape, BeamCentre]:
    """One centre for each frame size among `frames`.

    A session can hold more than one binning, and a centre in pixels means
    nothing without the frame it was measured on, so sizes are kept apart.
    An untrusted result is replaced by `fallback(shape)` when one is given and
    finds something (for example `session_fallback`), otherwise by the frame
    centre -- either way carrying the reason and what was measured.
    """
    by_shape: Dict[Shape, List[np.ndarray]] = {}
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim == 2:
            by_shape.setdefault((frame.shape[0], frame.shape[1]), []).append(frame)

    centres: Dict[Shape, BeamCentre] = {}
    for shape, group in by_shape.items():
        estimates = [e for e in (estimate_frame(f) for f in group) if e is not None]
        found = summarise(estimates, shape, len(group))
        if found.source == SOURCE_DETECTED and found.trusted:
            log_print(f"  beam centre: {found.describe()}")
            centres[shape] = found
            continue
        if found.source == SOURCE_DETECTED:
            reason = f"beam centre not trusted: {found.note}"
            log_print(f"  beam centre: not trusted on {shape[1]}x{shape[0]} "
                      f"({found.note}; measured {found.x:.1f}, {found.y:.1f})",
                      logging.WARNING)
        else:
            reason = found.note
            log_print(f"  beam centre: {found.note}", logging.WARNING)
        centres[shape] = untrusted(shape, reason, found, fallback)
    return centres


def untrusted(shape: Shape, reason: str, found: Optional[BeamCentre],
              fallback: Optional[Callable[[Shape], Optional[BeamCentre]]] = None
              ) -> BeamCentre:
    """What to draw about when the search of these frames did not settle it.

    A centre from `fallback` if it offers one, else the frame centre. Either
    way the measurement that was not trusted travels with it.
    """
    shape = (int(shape[0]), int(shape[1]))
    borrowed = None
    if fallback is not None:
        try:
            borrowed = fallback(shape)
        except Exception as exc:     # a picture must never stop a stage
            log_print(f"  beam centre: fallback failed "
                      f"({type(exc).__name__}: {exc})", logging.WARNING)
            borrowed = None
    if borrowed is not None:
        log_print(f"  beam centre: {borrowed.describe()} ({reason})",
                  logging.WARNING)
        if found is None:
            return replace(borrowed, note=reason)
        return replace(borrowed, note=reason, n_frames=found.n_frames,
                       n_ok=found.n_ok, n_agree=found.n_agree,
                       spread_px=found.spread_px, methods=dict(found.methods),
                       rejected=_rejected_of(found))
    log_print(f"  beam centre: no trusted centre from these frames; frame "
              f"centre {shape[1]}x{shape[0]}", logging.WARNING)
    if found is None or found.source == SOURCE_FRAME and found.rejected is None \
            and found.n_frames == 0:
        return frame_centre(shape, note=reason)
    return frame_centre(shape, note=reason, measured=found)


def sibling_centre(session_root, shape: Shape,
                   locations: Sequence[str] = SIDECAR_LOCATIONS
                   ) -> Optional[BeamCentre]:
    """A trusted centre measured in a sibling session, or None.

    Siblings are the other directories under the parent of `session_root`,
    each searched at `locations` for a sidecar. Only a centre that was
    DETECTED and TRUSTED on frames of exactly `shape` is taken -- never a
    borrowed one (no chains), never a frame centre, never a rescaled one.
    Several siblings give their median.
    """
    from pathlib import Path

    shape = (int(shape[0]), int(shape[1]))
    root = Path(session_root).resolve()
    parent = root.parent
    found: List[Tuple[str, float, float]] = []
    try:
        siblings = sorted(p for p in parent.iterdir() if p.is_dir())
    except OSError:
        return None
    for sibling in siblings:
        if sibling.resolve() == root:
            continue
        for rel in locations:
            path = sibling / rel
            if not path.is_file():
                continue
            try:
                with open(path) as handle:
                    record = json.load(handle)
            except (OSError, ValueError):
                continue
            for c in record.get('centres', []):
                try:
                    c_shape = (int(c['shape'][0]), int(c['shape'][1]))
                except (KeyError, TypeError, IndexError, ValueError):
                    continue
                if (c_shape == shape and c.get('source') == SOURCE_DETECTED
                        and c.get('trusted')):
                    found.append((sibling.name, float(c['x']), float(c['y'])))
                    break
            else:
                continue
            break       # one sidecar per sibling is enough
    if not found:
        return None
    names = [name for name, _, _ in found]
    x = float(np.median([f[1] for f in found]))
    y = float(np.median([f[2] for f in found]))
    return BeamCentre(x=x, y=y, shape=shape, source=SOURCE_SESSION, trusted=True,
                      borrowed_from=names)


def session_fallback(session_root, locations: Sequence[str] = SIDECAR_LOCATIONS):
    """`fallback` for `find_beam_centres`: the sibling sessions' centre."""
    def fallback(shape: Shape) -> Optional[BeamCentre]:
        return sibling_centre(session_root, shape, locations)
    return fallback


def resolve(mode: Mode, shape: Shape,
            detected: Optional[Dict[Shape, BeamCentre]] = None) -> BeamCentre:
    """The centre to draw a frame of `shape` about, under `mode`."""
    shape = (int(shape[0]), int(shape[1]))
    if isinstance(mode, tuple):
        return given_centre(mode, shape)
    if mode == MODE_FRAME:
        return frame_centre(shape)
    if detected and shape in detected:
        return detected[shape]
    return frame_centre(shape, note="beam centre not searched for this frame size")


def _note(ax, text: str, colour: str) -> None:
    """A wrapped note inside the bottom of the image, so a tight bbox stays square."""
    import textwrap
    ax.text(0.01, 0.01, textwrap.fill(text, 75), transform=ax.transAxes,
            color=colour, fontsize=7.5, ha='left', va='bottom', clip_on=True,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5))


def annotate(ax, centre: BeamCentre) -> None:
    """Mark a measured centre; say so on the picture when there is none."""
    if centre.measured:
        ax.plot([centre.x], [centre.y], marker='+', color='dimgrey',
                markersize=12, markeredgewidth=1.0, linestyle='none')
        ax.set_xlim(-0.5, centre.shape[1] - 0.5)
        ax.set_ylim(centre.shape[0] - 0.5, -0.5)
        if centre.source == SOURCE_SESSION:
            names = ', '.join(n[:-len('-reyes')] if n.endswith('-reyes') else n
                              for n in centre.borrowed_from)
            _note(ax, f"rings on SESSION beam centre from sibling session(s) "
                      f"{names}; this session: {centre.note or 'not searched'}",
                  'darkorange')
        elif centre.source == SOURCE_DETECTED and centre.note:
            # e.g. found on the wider sample: say where it came from
            _note(ax, f"beam centre {centre.note}", 'dimgrey')
    elif centre.note:
        ax.text(0.01, 0.01, f"rings on frame centre: {centre.note}",
                transform=ax.transAxes, color='firebrick', fontsize=8,
                ha='left', va='bottom')


def write_sidecar(out_dir: str, mode: Mode,
                  centres: Iterable[BeamCentre]) -> Optional[str]:
    """Record beside the snapshots what their rings were drawn about."""
    from pyREyes.lib.diffraction.beam_center import ALGORITHM_VERSION

    # `borrowed_from` is written only for a borrowed centre, so the sidecar of
    # a session that borrowed nothing reads exactly as before.
    def entry(centre: BeamCentre) -> dict:
        values = asdict(centre)
        if not values.get('borrowed_from'):
            values.pop('borrowed_from', None)
        return values

    record = {
        'mode': list(mode) if isinstance(mode, tuple) else mode,
        'algorithm': ALGORITHM_VERSION,
        'convention': 'x along columns, y along rows, zero-based pixel centres',
        'centres': [entry(c) for c in centres],
    }
    path = os.path.join(out_dir, SIDECAR)
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(path + '.tmp', 'w') as handle:
            json.dump(record, handle, indent=2)
        os.replace(path + '.tmp', path)
    except OSError as exc:
        log_print(f"  beam centre: could not write {path} ({exc})", logging.WARNING)
        return None
    return path


def read_sidecar(path) -> Dict[Shape, BeamCentre]:
    """The centres a sidecar recorded, by frame shape ({} if unreadable)."""
    try:
        with open(path) as handle:
            record = json.load(handle)
    except (OSError, ValueError):
        return {}
    known = set(BeamCentre.__dataclass_fields__)
    out: Dict[Shape, BeamCentre] = {}
    for c in record.get('centres', []):
        try:
            values = {k: v for k, v in c.items() if k in known}
            values['shape'] = (int(c['shape'][0]), int(c['shape'][1]))
            centre = BeamCentre(**values)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        out[centre.shape] = centre
    return out
