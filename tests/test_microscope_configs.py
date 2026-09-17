"""
Loading microscope_configs.json into MicroscopeConfig objects.
"""

import json
import logging
from dataclasses import fields

import pytest

from pyREyes.lib.REyes_microscope_configurations import (
    CONFIG_FILE,
    MicroscopeConfig,
    load_microscope_configs,
)


def base_params():
    with open(CONFIG_FILE) as handle:
        return json.load(handle)['Arctica-CETA']


def write_configs(tmp_path, data):
    path = tmp_path / 'microscope_configs.json'
    path.write_text(json.dumps(data))
    return path


def test_packaged_configs_load_without_warnings(caplog):
    with caplog.at_level(logging.WARNING):
        configs = load_microscope_configs()

    assert 'Arctica-CETA' in configs
    assert caplog.records == []


def test_unknown_keys_are_ignored_with_one_warning_per_key(tmp_path, caplog):
    params = dict(base_params(), dark_threshold=0.07)
    path = write_configs(tmp_path, {'A': params, 'B': dict(params)})

    with caplog.at_level(logging.WARNING):
        configs = load_microscope_configs(path)

    assert set(configs) == {'A', 'B'}
    assert not hasattr(configs['A'], 'dark_threshold')
    assert configs['A'].camera_length == params['camera_length']
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert 'dark_threshold' in warnings[0]


def test_missing_field_raises(tmp_path):
    params = base_params()
    del params['camera_length']
    path = write_configs(tmp_path, {'A': params})

    with pytest.raises(ValueError, match='camera_length'):
        load_microscope_configs(path)


def test_every_field_is_present_in_the_packaged_configs():
    names = {f.name for f in fields(MicroscopeConfig)}
    with open(CONFIG_FILE) as handle:
        for params in json.load(handle).values():
            assert names <= params.keys()


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_microscope_configs(tmp_path / 'absent.json')
