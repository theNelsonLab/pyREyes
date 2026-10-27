"""
Loading microscope_configs.json into MicroscopeConfig objects.
"""

import json
import logging
from dataclasses import MISSING, fields

import pytest

from pyREyes.lib.REyes_microscope_configurations import (
    CONFIG_FILE,
    MicroscopeConfig,
    load_microscope_configs,
)


def required_field_names():
    """Fields with no default. The optional ones (autoprocess_config,
    camera_length_calibrated) mean "no value for this instrument" when
    absent, so a configuration need not carry them."""
    return {f.name for f in fields(MicroscopeConfig)
            if f.default is MISSING and f.default_factory is MISSING}


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


def test_every_required_field_is_present_in_the_packaged_configs():
    names = required_field_names()
    with open(CONFIG_FILE) as handle:
        for params in json.load(handle).values():
            assert names <= params.keys()


def test_an_optional_field_may_be_absent(tmp_path):
    params = base_params()
    params.pop('autoprocess_config', None)
    params.pop('camera_length_calibrated', None)
    path = write_configs(tmp_path, {'A': params})

    configs = load_microscope_configs(path)

    assert configs['A'].autoprocess_config is None
    assert configs['A'].camera_length_calibrated is None


def test_an_optional_field_is_read_when_present(tmp_path):
    params = dict(base_params(), autoprocess_config='Other-Name',
                  camera_length_calibrated=1280.0)
    path = write_configs(tmp_path, {'A': params})

    configs = load_microscope_configs(path)

    assert configs['A'].autoprocess_config == 'Other-Name'
    assert configs['A'].camera_length_calibrated == 1280.0
    assert configs['A'].camera_length == params['camera_length']


def test_the_packaged_configs_set_the_optional_fields_only_where_known():
    """Only the Arctica DE Apollo configuration names a pyautoprocess
    configuration (Talos-Apollo-SM, the same camera geometry); no packaged
    configuration carries a calibrated camera length."""
    configs = load_microscope_configs()
    assert configs['Arctica-Apollo-SM'].autoprocess_config == 'Talos-Apollo-SM'
    for name, config in configs.items():
        if name != 'Arctica-Apollo-SM':
            assert config.autoprocess_config is None, name
        assert config.camera_length_calibrated is None, name


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_microscope_configs(tmp_path / 'absent.json')
