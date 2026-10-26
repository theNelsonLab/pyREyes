"""The SerialEM bundles in docs/scripts, checked as text.

SerialEM reports most macro errors only at run time, often hours into a grid,
so the checks that can be made on the text are made here: every Call target
exists, no variable is assigned with two lifetimes, no macro has gone missing,
and the bundles still write every phrase and record reyes-monitor reads.
"""

import re
from pathlib import Path

import pytest

BUNDLES = Path(__file__).resolve().parent.parent / 'docs' / 'scripts'
APOLLO = BUNDLES / 'SerialEMsettings-scripts_ARCTICA_DE_Apollo_REyes.txt'
CETA = BUNDLES / 'SerialEMsettings-scripts_ARCTICA_CETA_REyes.txt'
ALL = [APOLLO, CETA]

pytestmark = pytest.mark.skipif(not all(p.is_file() for p in ALL),
                                reason='docs/scripts not present (installed package)')

# Macro counts as docs/scripts/README.md gives them, so a dropped macro cannot
# pass unnoticed.
EXPECTED_MACROS = {APOLLO.name: 28, CETA.name: 28}

# Trigger phrases reyes-monitor matches (case-insensitive) in every bundle.
PHRASES = [
    'LM Montage data collection',
    'Completed fine eucentricity for all grid squares',
    'Completed StepScan for current grid square',
    'Completed StepScan for all grid squares',
    'Movie folder:',
    'Acquired movie:',
    'Completed movie acquisition',
    'Completed acquiring tilt movies for all crystals!',
    '!!!!! ERROR DURING PIPELINE EXECUTION !!!!!',
]

# Set-aside and wait records both bundles write; REyes_serialem parses them.
RECORDS = [
    'BLOCK_SKIPPED | stage=STPSCN | square=',
    'STPSCN_SUMMARY | ',
    'STPSCN_RESULT | result=',
    'ITEM_SKIPPED | stage=DFMV | item=',
    'VALVE_CLOSED | stage=',
    'WAIT_POLL_ERROR | wait=',
    'All_blocks_failed',
    'Blocks_set_aside_',
    'Items_set_aside_',
]

PERSISTENT = (':=', ':@=')
REGULAR = ('=', '@=')
ASSIGN = re.compile(r'^(\w+)(\[[^\]]*\])?\s*(:@=|:=|@=|=)(?!=)')
NOT_A_VARIABLE = {'if', 'elseif', 'echo', 'loop'}


def read(path):
    return path.read_text(errors='replace')


def assignments(text):
    """name -> set of operators it is assigned with, comments excluded."""
    found = {}
    for line in text.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        match = ASSIGN.match(stripped)
        if match and match.group(1).lower() not in NOT_A_VARIABLE:
            found.setdefault(match.group(1), set()).add(match.group(3))
    return found


@pytest.mark.parametrize('path', ALL, ids=lambda p: p.name)
def test_every_call_target_exists(path):
    text = read(path)
    defined = set(re.findall(r'^(?:MacroName|ScriptName)\s+(\S+)', text,
                             re.MULTILINE))
    called = set(re.findall(r'^\s*Call\s+(\S+)', text, re.MULTILINE))
    assert called - defined == set(), f'undefined Call target(s) in {path.name}'


@pytest.mark.parametrize('path', ALL, ids=lambda p: p.name)
def test_no_variable_mixes_persistent_and_regular_lifetimes(path):
    """SerialEM refuses, at run time, to assign a variable that already exists
    with the other lifetime."""
    offenders = {name: sorted(ops) for name, ops in assignments(read(path)).items()
                 if any(o in PERSISTENT for o in ops)
                 and any(o in REGULAR for o in ops)}
    assert not offenders, f'{path.name}: {offenders}'


@pytest.mark.parametrize('path', ALL, ids=lambda p: p.name)
def test_the_macro_inventory_is_intact(path):
    text = read(path)
    n = len(re.findall(r'^(?:MacroName|ScriptName)\s+\S+', text, re.MULTILINE))
    assert n == EXPECTED_MACROS[path.name]


@pytest.mark.parametrize('path', ALL, ids=lambda p: p.name)
@pytest.mark.parametrize('phrase', PHRASES)
def test_every_trigger_phrase_is_written(path, phrase):
    assert phrase.lower() in read(path).lower()


@pytest.mark.parametrize('path', ALL, ids=lambda p: p.name)
@pytest.mark.parametrize('record', RECORDS)
def test_every_bundle_writes_the_records_the_monitor_reads(path, record):
    assert record in read(path)
