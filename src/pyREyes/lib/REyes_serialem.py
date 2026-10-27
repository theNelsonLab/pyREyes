"""
What the SerialEM logs say about blocks and items that will never arrive.

A monitor that learns about the microscope only from success -- a block's
completion phrase, a movie's `Acquired movie:` line -- cannot tell any other
outcome from "not yet", and waits for "not yet" for ever when SerialEM has
abandoned a grid square or a whole grid. This module reads the other outcomes.
Four things, all of them pure reads of log text:

1. ORDERING THE LOGS OF ONE STAGE. A re-entered stage writes `<name>-1.log`,
   `-2`, ... beside `<name>.log`. A hyphen sorts before a dot, so a plain
   `sorted()` puts the resume log FIRST; `ordered_logs` puts it after the
   original, which is what "the latest record wins" needs.

2. SET-ASIDE RECORDS written by the REyes SerialEM macros (older macro
   versions write the VALVE_CLOSED form, which is still accepted):

       BLOCK_SKIPPED | stage=STPSCN | square=<pos> | item=<posct> | reason=<...>
       ITEM_SKIPPED  | stage=DFMV | item=<n> | reason=<...>
       VALVE_CLOSED  | stage=STPSCN | square=<pos> | ...
       VALVE_CLOSED  | stage=<...> | item=<n> | ...
       STPSCN_SUMMARY | ... | set_aside=<n>
       STPSCN_RESULT  | result=<complete|partial|failed> | collected=<n> | set_aside=<m>
       WAIT_POLL_ERROR | wait=<...> | errors=<n>                 (informational)

   Matched by EXACT PREFIX on the stripped line (`NAME | `), never by substring:
   these names must not be found inside an ordinary Echo. Logs without any of
   these records are read exactly as before.

3. SERIALEM ABORTED THIS GRID. START_REYES's Catch prints a banner into
   whichever log is open when the pipeline dies --

       !!!!! ERROR DURING PIPELINE EXECUTION !!!!!
       Grid: sample-reyes (Slot 1)

   -- and, for a multi-grid run, the run summary records
   `Slot 1 (sample-reyes): FAILED - <reason>`, e.g. `All_blocks_failed`.
   `serialem_abort()` finds these, attributes them to a grid by the `Grid:`
   line, and refuses a banner that a later log of the same grid has superseded
   (a resumed or recovered grid must not be stopped by the abort it is
   recovering from).

4. THE SET-ASIDE LIST. Blocks and movies set aside are written to
   `targets/set_aside_items.json`, one entry per (stage, item), so the losses
   of a session can be counted and revisited later instead of disappearing
   into a log.
"""

import json
import os
import re
from datetime import datetime, timezone

__version__ = '3.6.2'

# --------------------------------------------------------------------------- #
# 1. Ordering
# --------------------------------------------------------------------------- #
_SUFFIX_RE = re.compile(r'^(?P<stem>.+?)(?:-(?P<n>\d+))?\.log$', re.IGNORECASE)


def log_order_key(path):
    """(stem, resume number, name): `x.log` < `x-1.log` < `x-2.log` < `x-10.log`.

    Logs of different stems keep their alphabetical order, so a session with
    no resume logs is read in exactly the order `sorted()` gave before.
    """
    name = os.path.basename(str(path))
    m = _SUFFIX_RE.match(name)
    if not m:
        return (name, 0, name)
    return (m.group('stem'), int(m.group('n') or 0), name)


def ordered_logs(paths):
    """The given logs, originals before their resume logs (see module note)."""
    return sorted(set(str(p) for p in paths), key=log_order_key)


# --------------------------------------------------------------------------- #
# 2. Records
# --------------------------------------------------------------------------- #
RECORD_NAMES = ('BLOCK_SKIPPED', 'ITEM_SKIPPED', 'VALVE_CLOSED',
                'STPSCN_SUMMARY', 'STPSCN_RESULT', 'WAIT_POLL_ERROR')
SEP = ' | '

# Macros without STPSCN_RESULT say that a step-scan ended short only in prose.
STPSCN_INCOMPLETE = 'StepScan incomplete'


def parse_record(line, names=RECORD_NAMES):
    """`(NAME, {field: value})` for a contract record, else None.

    Exact prefix: the stripped line must START with `NAME | `. Values stay
    strings; `item` is the caller's to convert.
    """
    text = line.strip()
    for name in names:
        if text.startswith(name + SEP):
            fields = {}
            for chunk in text[len(name) + len(SEP):].split(SEP):
                key, sep, value = chunk.partition('=')
                if sep and key.strip():
                    fields[key.strip()] = value.strip()
            return name, fields
    return None


def read_text(path):
    """A log's text, or '' if it cannot be read. Logs are latin-1 (Windows)."""
    try:
        with open(path, encoding='latin-1') as handle:
            return handle.read()
    except OSError:
        return ''


def records_in_text(text, names=RECORD_NAMES):
    """Every contract record in `text`, in line order."""
    out = []
    for line in text.splitlines():
        parsed = parse_record(line, names)
        if parsed is not None:
            out.append(parsed)
    return out


def _as_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def skipped_items(log_paths, stages):
    """{item: {...}} for every item SerialEM set aside in one of `stages`.

    `ITEM_SKIPPED` and the older `VALVE_CLOSED` both count; logs are read in
    `ordered_logs` order and the latest record for an item wins. Stage names
    are compared upper-case. An item that was set aside and later collected
    still appears here -- the caller decides by its done flag, which is the
    authority for "collected".
    """
    wanted = {s.upper() for s in stages}
    found = {}
    for path in ordered_logs(log_paths):
        for name, fields in records_in_text(read_text(path),
                                            ('ITEM_SKIPPED', 'VALVE_CLOSED')):
            if fields.get('stage', '').upper() not in wanted:
                continue
            item = _as_int(fields.get('item'))
            if item is None:
                continue
            reason = fields.get('reason') or (
                'valve_closed' if name == 'VALVE_CLOSED' else 'unspecified')
            found[item] = {'stage': fields['stage'].upper(), 'item': item,
                           'reason': reason, 'record': name,
                           'log': os.path.basename(path)}
    return found


def block_skips(text):
    """{square: (reason, record)} for every step-scan block set aside.

    `BLOCK_SKIPPED | stage=STPSCN | square=<pos> | item=<posct> | reason=...`
    or the older `VALVE_CLOSED | stage=STPSCN | square=<pos> | ...`. Keyed on
    the `square=` field, NOT on the file: when a square fails before it has
    opened its own log, BLOCK_SKIPPED is written into the PREVIOUS square's
    log, and that square then has no log at all. For a closed valve both
    records may be written; BLOCK_SKIPPED's specific reason wins.
    """
    found = {}
    for name, fields in records_in_text(text, ('BLOCK_SKIPPED', 'VALVE_CLOSED')):
        if fields.get('stage', '').upper() != 'STPSCN':
            continue
        square = _as_int(fields.get('square'))
        if square is None:
            continue
        if name == 'BLOCK_SKIPPED':
            found[square] = (fields.get('reason') or 'unspecified', name)
        elif square not in found or found[square][1] != 'BLOCK_SKIPPED':
            found[square] = ('valve_closed', name)
    return found


def block_skip(text, square):
    """`(reason, record)` if `square` was set aside according to `text`, else None."""
    return block_skips(text).get(int(square))


def stpscn_result(text):
    """How the step-scan stage ended, from one log's text, or None.

    'complete' | 'partial' | 'failed' from `STPSCN_RESULT`. For macros without
    that record, `STPSCN_SUMMARY` followed by the prose line
    `StepScan incomplete` means the stage ended short: 'partial' if it says it
    collected anything, else 'failed'. None means the stage has not ended in
    this log (or ended normally, which the all-squares phrase says).
    """
    result = None
    summary = None
    for name, fields in records_in_text(text, ('STPSCN_RESULT', 'STPSCN_SUMMARY')):
        if name == 'STPSCN_RESULT':
            result = fields.get('result', '').lower() or None
        else:
            summary = fields
    if result:
        return result
    if summary is not None and any(line.strip().startswith(STPSCN_INCOMPLETE)
                                   for line in text.splitlines()):
        collected = _as_int(summary.get('collected'))
        return 'partial' if collected else 'failed'
    return None


# --------------------------------------------------------------------------- #
# 3. Has SerialEM abandoned this grid?
# --------------------------------------------------------------------------- #
ABORT_BANNERS = ('!!!!! ERROR DURING PIPELINE EXECUTION !!!!!',
                 '!!!!! ERROR DURING MONTAGE ACQUISITION !!!!!')
ALL_BLOCKS_FAILED = 'All_blocks_failed'
_GRID_RE = re.compile(r'^\s*Grid:\s*(\S+)')
_SUMMARY_RE = re.compile(r'^\s*Slot\s+\d+\s+\((?P<grid>[^)]+)\):\s*'
                         r'(?P<result>[A-Z]+)\s*-\s*(?P<reason>\S+)')
SUMMARY_SUFFIX = '_summary.log'
# A log holding only SerialEM's date line -- the stub the NEXT grid's first
# SaveLogOpenNew leaves in this grid's folder, or a summary log's opener -- is
# not activity of this grid and must not supersede a banner.
_STUB_LINES = 2
# Same-second writes (the banner's log and the next grid's stub) must not be
# ordered by a clock that resolves to the second on some filesystems.
_MTIME_SLACK = 2.0


def _substantive(text):
    return sum(1 for line in text.splitlines() if line.strip()) > _STUB_LINES


def _banner_in(text, grid_name):
    """(line number, banner) of the last banner attributed to `grid_name`."""
    lines = text.splitlines()
    hit = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped not in ABORT_BANNERS:
            continue
        named = None
        for follow in lines[i + 1:i + 4]:
            m = _GRID_RE.match(follow)
            if m:
                named = m.group(1)
                break
        if grid_name and named and named != grid_name:
            continue            # another grid's banner
        hit = (i + 1, stripped)
    return hit


class AbortWatch:
    """Reads a grid folder's logs for SerialEM's abort, caching by mtime/size.

    `check()` returns None, or a dict saying where the abort was found and why
    it counts. Rules:

    - Only top-level `*.log` of the grid folder (and `*_summary.log` of its
      parent): `failed/` subfolders are an operator's archive.
    - A banner counts only if no SUBSTANTIVE log of this folder is newer than
      the one holding it -- a resumed or recovered grid writes new logs, and
      must not be halted by the abort it is recovering from. The next grid's
      stub log (only SerialEM's date line) and summary logs are not
      substantive.
    - A banner names its grid (`Grid: <folder>`); one naming another folder is
      ignored. One with no Grid line is taken as this folder's.
    - `All_blocks_failed` anywhere in the newest substantive log counts.
    - A run summary line `Slot N (<this folder>): FAILED - <reason>` counts if
      the summary is at least as new as the grid's newest substantive log.
    - Evidence is never discarded for being old: only a newer substantive log
      of the grid (SerialEM resumed after the abort) supersedes it.
    """

    def __init__(self, root, grid_name=None, include_parent=True):
        self.root = os.path.abspath(str(root))
        self.grid_name = grid_name or os.path.basename(self.root)
        self.include_parent = include_parent
        self._cache = {}

    def _text(self, path, stat):
        key = (stat.st_mtime_ns, stat.st_size)
        cached = self._cache.get(path)
        if cached and cached[0] == key:
            return cached[1]
        text = read_text(path)
        self._cache[path] = (key, text)
        return text

    def _logs(self, folder):
        out = []
        try:
            entries = list(os.scandir(folder))
        except OSError:
            return out
        for entry in entries:
            if not entry.name.lower().endswith('.log'):
                continue
            try:
                if not entry.is_file():
                    continue
                stat = entry.stat()
            except OSError:
                continue
            out.append((entry.path, stat))
        return out

    def check(self):
        own = []          # (mtime, path, text) of substantive, non-summary logs
        summaries = []
        for path, stat in self._logs(self.root):
            text = self._text(path, stat)
            if os.path.basename(path).endswith(SUMMARY_SUFFIX):
                summaries.append((stat.st_mtime, path, text))
            elif _substantive(text):
                own.append((stat.st_mtime, path, text))
        if self.include_parent:
            parent = os.path.dirname(self.root)
            for path, stat in self._logs(parent):
                if os.path.basename(path).endswith(SUMMARY_SUFFIX):
                    summaries.append((stat.st_mtime, path,
                                      self._text(path, stat)))
        if not own:
            return None

        newest = max(m for m, _, _ in own)
        for mtime, path, text in sorted(own, key=lambda t: (t[0], log_order_key(t[1]))):
            if mtime < newest - _MTIME_SLACK:
                continue
            hit = _banner_in(text, self.grid_name)
            if hit:
                return {'reason': 'serialem_aborted', 'log': os.path.basename(path),
                        'line': hit[0], 'text': hit[1]}
            if any(ALL_BLOCKS_FAILED in line for line in text.splitlines()):
                return {'reason': ALL_BLOCKS_FAILED, 'log': os.path.basename(path),
                        'line': None, 'text': ALL_BLOCKS_FAILED}

        for mtime, path, text in summaries:
            if mtime < newest - _MTIME_SLACK:
                continue
            for line in text.splitlines():
                m = _SUMMARY_RE.match(line)
                if (m and m.group('grid') == self.grid_name
                        and m.group('result') == 'FAILED'):
                    return {'reason': m.group('reason'),
                            'log': os.path.basename(path), 'line': None,
                            'text': line.strip()}
        return None


def serialem_abort(root, grid_name=None):
    """One-shot `AbortWatch(root, grid_name).check()`."""
    return AbortWatch(root, grid_name).check()


def describe_abort(found):
    """One line for a log or a session status."""
    where = found['log'] + (f":{found['line']}" if found.get('line') else '')
    return f"SerialEM abandoned this grid ({found['reason']}; {where}: {found['text']})"


# --------------------------------------------------------------------------- #
# 4. The set-aside list
# --------------------------------------------------------------------------- #
SET_ASIDE_NAME = 'set_aside_items.json'
SET_ASIDE_SCHEMA = 1


def _utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def make_entry(stage, item, reason, source, record, detail='', log=None):
    """One set-aside item. `source` is 'serialem' or 'pyreyes'."""
    entry = {'stage': str(stage).upper(), 'item': int(item), 'reason': reason,
             'source': source, 'record': record, 'detail': detail or '',
             'utc': _utc()}
    if log:
        entry['log'] = log
    return entry


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='\n') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)
    return path


def set_aside_path(targets_dir):
    return os.path.join(str(targets_dir), SET_ASIDE_NAME)


def load_set_aside(targets_dir):
    """The entries of `targets/set_aside_items.json`, or []."""
    try:
        with open(set_aside_path(targets_dir), encoding='utf-8') as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    items = data.get('items') if isinstance(data, dict) else None
    return list(items) if isinstance(items, list) else []


def record_set_aside(targets_dir, entries):
    """Merge `entries` into the list, one per (stage, item), newest winning.

    Unchanged entries keep their first timestamp, so a monitor that re-reads
    the same SerialEM record every tick does not rewrite the file every tick.
    Returns the entries that were new or changed. SINGLE WRITER: called by the
    monitor only.
    """
    entries = list(entries)
    if not entries:
        return []
    current = {(e.get('stage'), e.get('item')): e for e in load_set_aside(targets_dir)}
    changed = []
    for entry in entries:
        key = (entry['stage'], entry['item'])
        old = current.get(key)
        same = old is not None and all(
            old.get(f) == entry.get(f) for f in ('reason', 'source', 'record',
                                                 'detail'))
        if same:
            continue
        current[key] = entry
        changed.append(entry)
    if changed:
        items = sorted(current.values(), key=lambda e: (e['stage'], e['item']))
        _atomic_json(set_aside_path(targets_dir),
                     {'schema': SET_ASIDE_SCHEMA, 'updated_utc': _utc(),
                      'items': items})
    return changed
