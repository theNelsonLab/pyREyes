"""Package hygiene: what is shipped imports, runs and is self-contained.

- every module under pyREyes imports;
- every console script in pyproject.toml resolves to a callable;
- every `pyREyes.*` import in the sources names a module that is in the tree;
- every module's __version__ matches the package version;
- no source file carries an absolute path from a developer's machine.
"""

import ast
import importlib
import pkgutil
import re
import sys
from pathlib import Path

import pytest

import pyREyes

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'src' / 'pyREyes'
PYPROJECT = ROOT / 'pyproject.toml'

pytestmark = pytest.mark.skipif(not (SRC.is_dir() and PYPROJECT.is_file()),
                                reason='needs the source tree')


def pyproject():
    if sys.version_info >= (3, 11):
        import tomllib
    else:                                              # pragma: no cover
        tomllib = pytest.importorskip('tomli')
    with open(PYPROJECT, 'rb') as handle:
        return tomllib.load(handle)


def source_modules():
    names = []
    for path in sorted(SRC.rglob('*.py')):
        rel = path.relative_to(SRC.parent).with_suffix('')
        parts = list(rel.parts)
        if parts[-1] == '__init__':
            parts = parts[:-1]
        names.append('.'.join(parts))
    return names


def test_the_tests_run_against_this_tree():
    assert any(Path(p).resolve() == SRC.resolve() for p in pyREyes.__path__)


@pytest.mark.parametrize('module', source_modules())
def test_every_module_imports(module):
    importlib.import_module(module)


def test_walking_the_package_finds_only_source_modules():
    walked = {m.name for m in pkgutil.walk_packages(pyREyes.__path__, 'pyREyes.')}
    assert walked <= set(source_modules())


def test_every_console_script_resolves():
    scripts = pyproject()['project']['scripts']
    assert scripts
    for name, target in scripts.items():
        module, _, attr = target.partition(':')
        assert callable(getattr(importlib.import_module(module), attr)), name


def test_every_internal_import_is_in_the_tree():
    known = set(source_modules()) | {'pyREyes'}
    missing = []
    for path in sorted(SRC.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and \
                    node.module.split('.')[0] == 'pyREyes':
                candidates = [node.module] + [f'{node.module}.{a.name}'
                                              for a in node.names]
                if not any(c in known for c in candidates):
                    missing.append(f'{path.name}: from {node.module} import ...')
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] == 'pyREyes' and \
                            alias.name not in known:
                        missing.append(f'{path.name}: import {alias.name}')
    assert not missing, missing


def test_module_versions_match_the_package():
    version = pyproject()['project']['version']
    pattern = re.compile(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
    wrong = {}
    for path in sorted(SRC.rglob('*.py')):
        for found in pattern.findall(path.read_text(encoding='utf-8')):
            if found != version:
                wrong[str(path.relative_to(SRC))] = found
    assert not wrong, f'package is {version}: {wrong}'


# Absolute paths that only exist on a particular machine.
MACHINE_PATHS = re.compile(
    r'(/Users/[^/\s]+/|/home/[^/\s]+/|[A-Za-z]:\\\\?Users\\\\?[^\\\s]+)',
    re.IGNORECASE)


@pytest.mark.parametrize('path', sorted(p for p in SRC.rglob('*')
                                        if p.is_file() and p.suffix in
                                        {'.py', '.json', '.md', '.txt'}),
                         ids=lambda p: str(p.relative_to(SRC)))
def test_no_machine_specific_paths(path):
    hits = [line.strip() for line in path.read_text(encoding='utf-8',
                                                    errors='replace').splitlines()
            if MACHINE_PATHS.search(line)]
    assert not hits, hits
