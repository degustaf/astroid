# Copyright (c) 2015-2016 Claudiu Popa <pcmanticore@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER


"""Tests for the astroid AST peephole optimizer."""

import ast
import textwrap

import pytest

import astroid
from astroid import astpeephole
from astroid import builder
from astroid import manager
from astroid import test_utils
from astroid.tests import resources


@pytest.fixture(autouse=True)
def _manager():
    _manager = manager.AstroidManager()
    _manager.optimize_ast = True
    yield _manager

    _manager.optimize_ast = False


@pytest.fixture
def _optimizer():
    return astpeephole.ASTPeepholeOptimizer()


def _get_binops(code):
    module = ast.parse(textwrap.dedent(code))
    return [node.value for node in module.body
            if isinstance(node, ast.Expr)]


@test_utils.require_version(maxver='3.0')
def test_optimize_binop_unicode(_optimizer):
    nodes = _get_binops("""
    u"a" + u"b" + u"c"

    u"a" + "c" + "b"
    u"a" + b"c"
    """)

    result = _optimizer.optimize_binop(nodes[0])
    assert isinstance(result, astroid.Const)
    assert result.value == u"abc"

    assert _optimizer.optimize_binop(nodes[1]) is None
    assert _optimizer.optimize_binop(nodes[2]) is None


def test_optimize_binop(_optimizer):
    nodes = _get_binops("""
    "a" + "b" + "c" + "d"
    b"a" + b"b" + b"c" + b"d"
    "a" + "b"

    "a" + "b" + 1 + object
    var = 4
    "a" + "b" + var + "c"
    "a" + "b" + "c" - "4"
    "a" + "b" + "c" + "d".format()
    "a" - "b"
    "a"
    1 + 4 + 5 + 6
    """)

    result = _optimizer.optimize_binop(nodes[0])
    assert isinstance(result, astroid.Const)
    assert result.value == "abcd"

    result = _optimizer.optimize_binop(nodes[1])
    assert isinstance(result, astroid.Const)
    assert result.value == b"abcd"

    for node in nodes[2:]:
        assert _optimizer.optimize_binop(node) is None


def test_big_binop_crash():
    # Test that we don't fail on a lot of joined strings
    # through the addition operator.
    module = resources.build_file('data/joined_strings.py')
    element = next(module['x'].infer())
    assert isinstance(element, astroid.Const)
    assert len(element.value) == 61660


def test_optimisation_disabled(_manager):
    try:
        _manager.optimize_ast = False
        module = builder.parse("""
        '1' + '2' + '3'
        """)
        assert isinstance(module.body[0], astroid.Expr)
        assert isinstance(module.body[0].value, astroid.BinOp)
        assert isinstance(module.body[0].value.left, astroid.BinOp)
        assert isinstance(module.body[0].value.left.left, astroid.Const)
    finally:
        _manager.optimize_ast = True
