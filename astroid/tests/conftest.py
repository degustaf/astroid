# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

import sys

import pytest

from astroid import MANAGER
from astroid.bases import BUILTINS
from .resources import build_file, find


@pytest.fixture
def sys_path_setup():
    sys.path.insert(0, find(''))
    yield

    del sys.path[0]
    datadir = find('')
    for key in list(sys.path_importer_cache):
        if key.startswith(datadir):
            del sys.path_importer_cache[key]


@pytest.fixture(scope='module')
def astroid_cache_setup():
    """Mixin for handling the astroid cache problems.

    When clearing the astroid cache, some tests fails due to
    cache inconsistencies, where some objects had a different
    builtins object referenced.
    This saves the builtins module and makes sure to add it
    back to the astroid_cache after the tests finishes.
    The builtins module is special, since some of the
    transforms for a couple of its objects (str, bytes etc)
    are executed only once, so astroid_bootstrapping will be
    useless for retrieving the original builtins module.
    """
    _builtins = MANAGER.astroid_cache.get(BUILTINS)
    yield _builtins

    if _builtins:
        MANAGER.astroid_cache[BUILTINS] = _builtins


@pytest.fixture(scope='session')
@pytest.mark.usefixture('sys_path_setup')
def module():
    return build_file('data/module.py', 'data.module')


@pytest.fixture
@pytest.mark.usefixture('sys_path_setup')
def module2():
    return build_file('data/module2.py', 'data.module2')


@pytest.fixture
@pytest.mark.usefixture('sys_path_setup')
def nonregr():
    return build_file('data/nonregr.py', 'data.nonregr')


@pytest.fixture
def YO_cls(module):
    return module['YO']


@pytest.fixture
def YOUPI_cls(module):
    return module['YOUPI']
