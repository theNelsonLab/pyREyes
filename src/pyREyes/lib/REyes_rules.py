"""
Classification rules that can be overridden from the command line.

The "good rule" is the minimum DQI (Diffraction Quality Index: lattice-quality
peaks per diffraction peak) for a frame to count as "Good diffraction" in
``dif-map-2``. Each microscope configuration carries its own value; this
module gives every command of the package one way to override it:

    command-line value  >  REYES_GOOD_RULE in the environment  >  the default

The environment variable is how a value reaches a stage that runs in another
process: a program that starts ``dif-map-2`` (the monitor, or any other
pipeline built on this package) calls `export_good_rule` once and every
process it starts, directly or indirectly, resolves to the same value. A
command that also receives ``--good-rule`` explicitly uses that value.
"""

import argparse
import logging
import os
from typing import Mapping, Optional

__version__ = '3.6.2'

# Environment variable read by `resolve_good_rule` and written by
# `export_good_rule`.
GOOD_RULE_ENV = 'REYES_GOOD_RULE'

# Source labels returned by `good_rule_source`.
SOURCE_CLI = 'command line'
SOURCE_ENV = GOOD_RULE_ENV

_logger = logging.getLogger(__name__)


def _env_value(env: Optional[Mapping[str, str]], warn: bool = True) -> Optional[float]:
    """The good rule in `env` (os.environ when None), or None if unset or malformed."""
    env = os.environ if env is None else env
    raw = env.get(GOOD_RULE_ENV)
    if raw is None or raw.strip() == '':
        return None
    try:
        return float(raw)
    except ValueError:
        if warn:
            _logger.warning(f"Ignoring {GOOD_RULE_ENV}={raw!r}: not a number")
        return None


def resolve_good_rule(cli_value: Optional[float], default: float,
                      env: Optional[Mapping[str, str]] = None) -> float:
    """The good rule in force: `cli_value`, else the environment, else `default`.

    Args:
        cli_value: The value given on the command line (None when not given)
        default: The value to fall back on, normally the microscope
            configuration's `good_rule`
        env: The environment to read `REYES_GOOD_RULE` from; os.environ when None

    Returns:
        The value as a float. A `REYES_GOOD_RULE` that is not a number is
        ignored with a warning, as if it were unset.
    """
    if cli_value is not None:
        return float(cli_value)
    from_env = _env_value(env)
    if from_env is not None:
        return from_env
    return float(default)


def good_rule_source(cli_value: Optional[float],
                     env: Optional[Mapping[str, str]] = None,
                     default_label: str = 'default') -> str:
    """Where `resolve_good_rule` takes its value from, for a log line.

    Returns 'command line', 'REYES_GOOD_RULE' or `default_label`, following
    the same precedence as `resolve_good_rule` (a malformed environment value
    counts as unset).
    """
    if cli_value is not None:
        return SOURCE_CLI
    if _env_value(env, warn=False) is not None:
        return SOURCE_ENV
    return default_label


def add_good_rule_argument(parser: argparse.ArgumentParser,
                           help_default: str = 'the microscope configuration') -> argparse.Action:
    """Add `--good-rule DQI` to `parser` (or an argument group).

    The option defaults to None so that `resolve_good_rule` can tell "not
    given" from any numeric value. `help_default` names where the value comes
    from when neither the option nor `REYES_GOOD_RULE` is set.
    """
    return parser.add_argument(
        '--good-rule',
        type=float,
        default=None,
        metavar='DQI',
        help=f'Minimum DQI (lattice-quality peaks per diffraction peak) for a frame '
             f'to count as good diffraction. Precedence: this option, then the '
             f'{GOOD_RULE_ENV} environment variable, then {help_default}')


def export_good_rule(value: Optional[float],
                     env: Optional[Mapping[str, str]] = None) -> None:
    """Set `REYES_GOOD_RULE` for the processes started from here on.

    Does nothing when `value` is None, so an unset option leaves the
    environment -- and any value inherited from a parent process -- untouched.
    `env` is the mapping to write to; os.environ when None.
    """
    if value is None:
        return
    env = os.environ if env is None else env
    env[GOOD_RULE_ENV] = repr(float(value))
