"""The --good-rule override and its precedence.

One value, resolved the same way everywhere: the command-line option, else
REYES_GOOD_RULE in the environment, else the microscope configuration. The
tests pin the helper, the option on `dif-map-2` and `reyes-monitor`, the
classification it changes, the `classification` key it adds to a block's
`reyes.json`, and that without the option nothing changes: no value is
forwarded to dif-map-2 and the environment is left alone.
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import replace

import pandas as pd
import pytest

from conftest import TracingProcessor, drive
import pyREyes.dif_map as dm
from pyREyes.lib.REyes_rules import (GOOD_RULE_ENV, add_good_rule_argument,
                                     export_good_rule, good_rule_source,
                                     resolve_good_rule)


CETA = 'Arctica-CETA'


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No REYES_GOOD_RULE from the outside, and whatever a test sets is undone."""
    monkeypatch.delenv(GOOD_RULE_ENV, raising=False)


# --------------------------------------------------------------------------- #
# The helper
# --------------------------------------------------------------------------- #
def test_precedence_cli_over_env_over_default():
    assert resolve_good_rule(None, 3, env={}) == 3.0
    assert resolve_good_rule(None, 3, env={GOOD_RULE_ENV: '1.5'}) == 1.5
    assert resolve_good_rule(2, 3, env={GOOD_RULE_ENV: '1.5'}) == 2.0
    assert isinstance(resolve_good_rule(None, 3, env={}), float)

    assert good_rule_source(None, env={}) == 'default'
    assert good_rule_source(None, env={}, default_label='microscope configuration') == \
        'microscope configuration'
    assert good_rule_source(None, env={GOOD_RULE_ENV: '1.5'}) == GOOD_RULE_ENV
    assert good_rule_source(2, env={GOOD_RULE_ENV: '1.5'}) == 'command line'


def test_the_environment_is_read_when_env_is_not_given(monkeypatch):
    monkeypatch.setenv(GOOD_RULE_ENV, '2.5')
    assert resolve_good_rule(None, 3) == 2.5
    assert good_rule_source(None) == GOOD_RULE_ENV


def test_a_malformed_env_value_is_ignored_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_good_rule(None, 3, env={GOOD_RULE_ENV: 'high'}) == 3.0
    assert any(GOOD_RULE_ENV in r.message and 'high' in r.message
               and r.levelno == logging.WARNING for r in caplog.records)
    assert good_rule_source(None, env={GOOD_RULE_ENV: 'high'}) == 'default'
    # an empty value counts as unset too
    assert resolve_good_rule(None, 3, env={GOOD_RULE_ENV: ''}) == 3.0


def test_add_good_rule_argument_parses_a_float_and_defaults_to_none():
    parser = argparse.ArgumentParser()
    add_good_rule_argument(parser)
    assert parser.parse_args([]).good_rule is None
    assert parser.parse_args(['--good-rule', '2']).good_rule == 2.0
    with pytest.raises(SystemExit):
        parser.parse_args(['--good-rule', 'high'])

    # on an argument group, with the default named in the help text
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('rules')
    action = add_good_rule_argument(group, help_default='the preset')
    assert action.metavar == 'DQI'
    assert GOOD_RULE_ENV in action.help and 'the preset' in action.help
    assert parser.parse_args(['--good-rule', '1.5']).good_rule == 1.5


def test_export_good_rule_sets_the_variable_for_children_and_none_leaves_it():
    env = {}
    export_good_rule(None, env=env)
    assert env == {}
    export_good_rule(2, env=env)
    assert env == {GOOD_RULE_ENV: '2.0'}
    assert resolve_good_rule(None, 3, env=env) == 2.0

    # inherited value kept when nothing is exported
    env = {GOOD_RULE_ENV: '1.5'}
    export_good_rule(None, env=env)
    assert env == {GOOD_RULE_ENV: '1.5'}

    assert GOOD_RULE_ENV not in os.environ
    export_good_rule(2.5)
    assert os.environ[GOOD_RULE_ENV] == '2.5'


# --------------------------------------------------------------------------- #
# dif-map-2: classification
# --------------------------------------------------------------------------- #
def analyzer(config):
    return dm.DiffractionAnalyzer(config, dm.ImageProcessor(config))


def test_ceta_frame_with_dqi_2_5_is_bad_at_rule_3_and_good_at_rule_2():
    config = dm.MICROSCOPE_CONFIGS[CETA]
    assert config.good_rule == 3

    good = dm.DiffractionQuality.GOOD_DIFFRACTION
    bad = dm.DiffractionQuality.BAD_DIFFRACTION
    assert analyzer(config)._classify_diffraction(20, 50) is bad
    assert analyzer(replace(config, good_rule=2.0))._classify_diffraction(20, 50) is good
    # the shared configuration is not changed by the override
    assert dm.MICROSCOPE_CONFIGS[CETA].good_rule == 3


def frames():
    # equal sums: no row falls under the Grid threshold (mean / grid_rule)
    return pd.DataFrame({
        'Sum': [1000.0, 1000.0],
        'FilteredPeaks': [20, 20],
        'FTPeaks': [50, 70],
        'DifQuality': ['', ''],
    })


def test_update_diffraction_quality_follows_the_rule_in_force():
    config = dm.MICROSCOPE_CONFIGS[CETA]

    df = frames()
    analyzer(config).update_diffraction_quality(df)
    assert list(df['DifQuality']) == ['Bad diffraction', 'Good diffraction']

    df = frames()
    analyzer(replace(config, good_rule=2.0)).update_diffraction_quality(df)
    assert list(df['DifQuality']) == ['Good diffraction', 'Good diffraction']


def test_dif_map_parser_has_the_option():
    args = dm.create_parser().parse_args(['--good-rule', '2'])
    assert args.good_rule == 2.0
    assert dm.create_parser().parse_args([]).good_rule is None


def run_dif_map(tmp_path, monkeypatch, argv):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['dif-map-2', '--microscope', CETA] + argv)
    code = dm.main()
    log = (tmp_path / 'REyes_logs' / 'dmp_processing.log').read_text()
    return code, log


def test_dif_map_logs_the_rule_and_its_source(tmp_path, monkeypatch):
    code, log = run_dif_map(tmp_path, monkeypatch, [])
    assert code == 0
    assert 'good rule: 3 (microscope configuration)' in log

    monkeypatch.setenv(GOOD_RULE_ENV, '1.5')
    code, log = run_dif_map(tmp_path, monkeypatch, [])
    assert code == 0
    assert f'good rule: 1.5 ({GOOD_RULE_ENV})' in log

    code, log = run_dif_map(tmp_path, monkeypatch, ['--good-rule', '2'])
    assert code == 0
    assert 'good rule: 2 (command line)' in log
    assert dm.MICROSCOPE_CONFIGS[CETA].good_rule == 3


# --------------------------------------------------------------------------- #
# dif-map-2: reyes.json
# --------------------------------------------------------------------------- #
def test_reyes_json_records_the_classification_rules(tmp_path):
    (tmp_path / 'REyes_logs').mkdir()
    folder = tmp_path / 'b_1'
    folder.mkdir()
    for i in range(4):
        (folder / f'20251107_{i:05d}_integrated_movie.mrc').touch()

    config = replace(dm.MICROSCOPE_CONFIGS[CETA], good_rule=2.0)
    handler = dm.FileHandler(str(tmp_path), config)
    indices = handler.determine_indices(folder)
    handler.create_reyes_json(folder, indices)
    data = json.loads((folder / 'reyes.json').read_text())

    assert data['classification'] == {'good_rule': 2.0,
                                      'grid_rule': config.grid_rule}
    # every 3.6.1 key is still there
    assert set(data) >= {'version', 'min_id', 'max_id', 'format', 'uid', 'files'}
    assert data['version'] == 1


# --------------------------------------------------------------------------- #
# reyes-monitor
# --------------------------------------------------------------------------- #
def dif_map_calls(trace):
    return [t for t in trace if t.startswith('MODULE dif_map')]


def test_the_monitor_forwards_the_rule_to_dif_map_and_exports_it(session):
    processor = TracingProcessor(working_directory=session, good_rule=2.5)
    assert os.environ[GOOD_RULE_ENV] == '2.5'

    calls = dif_map_calls(drive(processor))
    assert len(calls) == 2
    for call in calls:
        assert 'good_rule=2.5' in call.split()
        assert 'microscope=Arctica-CETA' in call.split()


def test_without_the_option_nothing_is_forwarded_and_the_env_is_untouched(session):
    processor = TracingProcessor(working_directory=session)
    assert processor.good_rule is None
    assert GOOD_RULE_ENV not in os.environ

    calls = dif_map_calls(drive(processor))
    assert len(calls) == 2
    assert not any('good_rule' in call for call in calls)
    assert GOOD_RULE_ENV not in os.environ


def test_an_inherited_value_is_left_for_the_stages(session, monkeypatch):
    monkeypatch.setenv(GOOD_RULE_ENV, '1.5')
    processor = TracingProcessor(working_directory=session)
    assert os.environ[GOOD_RULE_ENV] == '1.5'
    assert not any('good_rule' in call for call in dif_map_calls(drive(processor)))
