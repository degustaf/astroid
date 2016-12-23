# -*- coding: utf-8 -*-
# Copyright (c) 2014-2016 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2015 Florian Bruhin <me@the-compiler.org>
# Copyright (c) 2015 Radosław Ganczarek <radoslaw@ganczarek.in>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

"""
unit tests for module modutils (module manipulation utilities)
"""
import email
import os
import sys
from xml import etree

import astroid
from astroid.interpreter._import import spec
from astroid import modutils
from astroid.tests import resources
import pytest


def _get_file_from_object(obj):
    return modutils._path_from_filename(obj.__file__)


@pytest.fixture
def package():
    yield "mypypa"

    for k in list(sys.path_importer_cache):
        if 'MyPyPa' in k:
            del sys.path_importer_cache[k]


def test_find_zipped_module(package):
    found_spec = spec.find_spec(
        [package], [resources.find('data/MyPyPa-0.1.0-py2.5.zip')])
    assert found_spec.type == spec.ModuleType.PY_ZIPMODULE
    assert found_spec.location.split(os.sep)[-3:] == \
        ["data", "MyPyPa-0.1.0-py2.5.zip", package]


def test_find_egg_module(package):
    found_spec = spec.find_spec(
        [package], [resources.find('data/MyPyPa-0.1.0-py2.5.egg')])
    assert found_spec.type == spec.ModuleType.PY_ZIPMODULE
    assert found_spec.location.split(os.sep)[-3:] == \
        ["data", "MyPyPa-0.1.0-py2.5.egg", package]


def test_knownValues_load_module_from_name_1():
    assert modutils.load_module_from_name('sys') == sys


def test_knownValues_load_module_from_name_2():
    assert modutils.load_module_from_name('os.path') == os.path


def test_raise_load_module_from_name_1():
    with pytest.raises(ImportError):
        modutils.load_module_from_name('os.path', use_sys=0)


def test_knownValues_get_module_part_1():
    assert modutils.get_module_part('astroid.modutils') == 'astroid.modutils'


def test_knownValues_get_module_part_2():
    assert modutils.get_module_part('astroid.modutils.get_module_part') == \
        'astroid.modutils'


def test_knownValues_get_module_part_3():
    """relative import from given file"""
    assert modutils.get_module_part('node_classes.AssName',
                                    modutils.__file__) == 'node_classes'


def test_knownValues_get_compiled_module_part():
    assert modutils.get_module_part('math.log10') == 'math'
    assert modutils.get_module_part('math.log10', __file__) == 'math'


def test_knownValues_get_builtin_module_part():
    assert modutils.get_module_part('sys.path') == 'sys'
    assert modutils.get_module_part('sys.path', '__file__') == 'sys'


def test_get_module_part_exception():
    with pytest.raises(ImportError):
        modutils.get_module_part('unknown.module', modutils.__file__)


def test_knownValues_modpath_from_file_1():
    from xml.etree import ElementTree
    assert modutils.modpath_from_file(ElementTree.__file__) == \
        ['xml', 'etree', 'ElementTree']


def test_knownValues_modpath_from_file_2():
    assert modutils.modpath_from_file('unittest_modutils.py',
                                      {os.getcwd(): 'arbitrary.pkg'}) == \
        ['arbitrary', 'pkg', 'unittest_modutils']


def test_raise_modpath_from_file_Exception():
    with pytest.raises(Exception):
        modutils.modpath_from_file('/turlututu')


@pytest.mark.skip(reason="Why does this fail in pytest.")
@pytest.mark.usefixture('sys_path_setup')
def test_do_not_load_twice():
    modutils.load_module_from_modpath(['data', 'lmfp', 'foo'])
    modutils.load_module_from_modpath(['data', 'lmfp'])
    # pylint: disable=no-member; just-once is added by a test file dynamically.
    assert len(sys.just_once) == 1
    del sys.just_once


@pytest.mark.usefixture('sys_path_setup')
def test_site_packages():
    filename = _get_file_from_object(modutils)
    result = modutils.file_from_modpath(['astroid', 'modutils'])
    assert os.path.realpath(result) == os.path.realpath(filename)


@pytest.mark.usefixture('sys_path_setup')
def test_std_lib():
    path = modutils.file_from_modpath(['os', 'path']).replace('.pyc', '.py')
    assert os.path.realpath(path) == \
        os.path.realpath(os.path.__file__.replace('.pyc', '.py'))


@pytest.mark.usefixture('sys_path_setup')
def test_builtin():
    assert modutils.file_from_modpath(['sys']) is None


@pytest.mark.usefixture('sys_path_setup')
def test_unexisting():
    with pytest.raises(ImportError):
        modutils.file_from_modpath(['turlututu'])


@pytest.mark.skip(reason="Why does this fail in pytest")
@pytest.mark.usefixture('sys_path_setup')
def test_unicode_in_package_init():
    # file_from_modpath should not crash when reading an __init__
    # file with unicode characters.
    modutils.file_from_modpath(["data", "unicode_package", "core"])


def test():
    filename = _get_file_from_object(os.path)
    assert modutils.get_source_file(os.path.__file__) == \
        os.path.normpath(filename)


def test_raise():
    with pytest.raises(modutils.NoSourceFile):
        modutils.get_source_file('whatever')


# class StandardLibModuleTest(resources.SysPathSetup, unittest.TestCase):
#     """
#     return true if the module may be considered as a module from the standard
#     library
#     """

@pytest.mark.usefixture('sys_path_setup')
def test_datetime():
    # This is an interesting example, since datetime, on pypy,
    # is under lib_pypy, rather than the usual Lib directory.
    assert modutils.is_standard_module('datetime')


@pytest.mark.usefixture('sys_path_setup')
def test_builtins():
    if sys.version_info < (3, 0):
        assert modutils.is_standard_module('__builtin__')
        assert not modutils.is_standard_module('builtins')
    else:
        assert not modutils.is_standard_module('__builtin__')
        assert modutils.is_standard_module('builtins')


@pytest.mark.usefixture('sys_path_setup')
def test_builtin2():
    assert modutils.is_standard_module('sys')
    assert modutils.is_standard_module('marshal')


@pytest.mark.usefixture('sys_path_setup')
def test_nonstandard():
    assert not modutils.is_standard_module('astroid')


@pytest.mark.usefixture('sys_path_setup')
def test_unknown():
    assert not modutils.is_standard_module('unknown')


@pytest.mark.usefixture('sys_path_setup')
def test_4():
    assert modutils.is_standard_module('hashlib')
    assert modutils.is_standard_module('pickle')
    assert modutils.is_standard_module('email')
    assert modutils.is_standard_module('io') == (sys.version_info >= (2, 6))
    assert modutils.is_standard_module('StringIO') == (sys.version_info < (3, 0))
    assert modutils.is_standard_module('unicodedata')


@pytest.mark.skip(reason="why does this fail under pytest")
@pytest.mark.usefixture('sys_path_setup')
def test_custom_path():
    datadir = resources.find('')
    if datadir.startswith(modutils.EXT_LIB_DIR):
        pytest.mark.skip('known breakage of is_standard_module on installed package')

    assert modutils.is_standard_module('data.module', (datadir,))
    assert modutils.is_standard_module('data.module', (os.path.abspath(datadir),))


@pytest.mark.usefixture('sys_path_setup')
def test_failing_edge_cases():
    # using a subpackage/submodule path as std_path argument
    assert not modutils.is_standard_module('xml.etree', etree.__path__)
    # using a module + object name as modname argument
    assert modutils.is_standard_module('sys.path')
    # this is because only the first package/module is considered
    assert modutils.is_standard_module('sys.whatever')
    assert not modutils.is_standard_module('xml.whatever', etree.__path__)


def test_knownValues_is_relative_1():
    assert modutils.is_relative('utils', email.__path__[0])


def test_knownValues_is_relative_2():
    assert modutils.is_relative('ElementPath', etree.ElementTree.__file__)


def test_knownValues_is_relative_3():
    assert not modutils.is_relative('astroid', astroid.__path__[0])


def test_get_module_files_1():
    package = resources.find('data/find_test')
    modules = set(modutils.get_module_files(package, []))
    expected = ['__init__.py', 'module.py', 'module2.py',
                'noendingnewline.py', 'nonregr.py']
    assert modules == {os.path.join(package, x) for x in expected}


def test_get_all_files():
    """test that list_all returns all Python files from given location
    """
    non_package = resources.find('data/notamodule')
    modules = modutils.get_module_files(non_package, [], list_all=True)
    assert modules == [os.path.join(non_package, 'file.py')]


def test_load_module_set_attribute():
    import xml.etree.ElementTree
    import xml
    del xml.etree.ElementTree
    del sys.modules['xml.etree.ElementTree']
    m = modutils.load_module_from_modpath(['xml', 'etree', 'ElementTree'])
    assert hasattr(xml, 'etree')
    assert hasattr(xml.etree, 'ElementTree')
    assert m is xml.etree.ElementTree
