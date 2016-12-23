# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2014-2016 Claudiu Popa <pcmanticore@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

import inspect
import os

from six.moves import builtins
import pytest

from astroid.builder import AstroidBuilder, extract_node
from astroid.raw_building import (
    attach_dummy_node, build_module,
    build_class, build_function, build_from_import
)
from astroid import test_utils
from astroid import nodes
from astroid.bases import BUILTINS


def test_attach_dummy_node():
    node = build_module('MyModule')
    attach_dummy_node(node, 'DummyNode')
    assert 1 == len(list(node.get_children()))


def test_build_module():
    node = build_module('MyModule')
    assert node.name == 'MyModule'
    assert not node.pure_python
    assert not node.package
    assert node.parent is None


def test_build_class():
    node = build_class('MyClass')
    assert node.name == 'MyClass'
    assert node.doc is None


def test_build_function():
    node = build_function('MyFunction')
    assert node.name == 'MyFunction'
    assert node.doc is None


def test_build_function_args():
    args = ['myArgs1', 'myArgs2']
    # pylint: disable=no-member; not aware of postinit
    node = build_function('MyFunction', args)
    assert 'myArgs1' == node.args.args[0].name
    assert 'myArgs2' == node.args.args[1].name
    assert 2 == len(node.args.args)


def test_build_function_defaults():
    # pylint: disable=no-member; not aware of postinit
    defaults = ['defaults1', 'defaults2']
    node = build_function('MyFunction', None, defaults)
    assert 2 == len(node.args.defaults)


def test_build_from_import():
    names = ['exceptions, inference, inspector']
    node = build_from_import('astroid', names)
    assert len(names) == len(node.names)


@test_utils.require_version(minver='3.0')
def test_io_is__io():
    # _io module calls itself io. This leads
    # to cyclic dependencies when astroid tries to resolve
    # what io.BufferedReader is. The code that handles this
    # is in astroid.raw_building.imported_member, which verifies
    # the true name of the module.
    import _io

    builder = AstroidBuilder()
    module = builder.inspect_build(_io)
    buffered_reader = module.getattr('BufferedReader')[0]
    assert buffered_reader.root().name == 'io'


@pytest.mark.skipif(os.name != 'java', reason='Requires Jython')
def test_open_is_inferred_correctly():
    # Lot of Jython builtins don't have a __module__ attribute.
    for name, _ in inspect.getmembers(builtins, predicate=inspect.isbuiltin):
        if name == 'print':
            continue
        node = extract_node('{0} #@'.format(name))
        inferred = next(node.infer())
        assert isinstance(inferred, nodes.FunctionDef), name
        assert inferred.root().name == BUILTINS, name
