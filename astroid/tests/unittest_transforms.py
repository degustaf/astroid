# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>
# Copyright (c) 2015-2016 Claudiu Popa <pcmanticore@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER


from __future__ import print_function

import contextlib
import time

import pytest

from astroid import builder
from astroid import nodes
from astroid import parse
from astroid import transforms


@contextlib.contextmanager
def add_transform(manager, node, transform, predicate=None):
    manager.register_transform(node, transform, predicate)
    try:
        yield
    finally:
        manager.unregister_transform(node, transform, predicate)


@pytest.fixture
def transformer():
    return transforms.TransformVisitor()


def parse_transform(transformer, code):
    module = parse(code, apply_transforms=False)
    return transformer.visit(module)


def test_function_inlining_transform(transformer):
    def transform_call(node):
        # Let's do some function inlining
        inferred = next(node.infer())
        return inferred

    transformer.register_transform(nodes.Call, transform_call)

    module = parse_transform(transformer, '''
    def test(): return 42
    test() #@
    ''')

    assert isinstance(module.body[1], nodes.Expr)
    assert isinstance(module.body[1].value, nodes.Const)
    assert module.body[1].value.value == 42


def test_recursive_transforms_into_astroid_fields(transformer):
    # Test that the transformer walks properly the tree
    # by going recursively into the _astroid_fields per each node.
    def transform_compare(node):
        # Let's check the values of the ops
        _, right = node.ops[0]
        # Assume they are Consts and they were transformed before
        # us.
        return nodes.const_factory(node.left.value < right.value)

    def transform_name(node):
        # Should be Consts
        return next(node.infer())

    transformer.register_transform(nodes.Compare, transform_compare)
    transformer.register_transform(nodes.Name, transform_name)

    module = parse_transform(transformer, '''
    a = 42
    b = 24
    a < b
    ''')

    assert isinstance(module.body[2], nodes.Expr)
    assert isinstance(module.body[2].value, nodes.Const)
    assert not module.body[2].value.value


def test_transform_patches_locals(transformer):
    def transform_function(node):
        assign = nodes.Assign()
        name = nodes.AssignName()
        name.name = 'value'
        assign.targets = [name]
        assign.value = nodes.const_factory(42)
        node.body.append(assign)

    transformer.register_transform(nodes.FunctionDef, transform_function)

    module = parse_transform(transformer, '''
    def test():
        pass
    ''')

    func = module.body[0]
    assert len(func.body) == 2
    assert isinstance(func.body[1], nodes.Assign)
    assert func.body[1].as_string() == 'value = 42'


def test_predicates(transformer):
    def transform_call(node):
        inferred = next(node.infer())
        return inferred

    def should_inline(node):
        return node.func.name.startswith('inlineme')

    transformer.register_transform(nodes.Call, transform_call, should_inline)

    module = parse_transform(transformer, '''
    def inlineme_1():
        return 24
    def dont_inline_me():
        return 42
    def inlineme_2():
        return 2
    inlineme_1()
    dont_inline_me()
    inlineme_2()
    ''')
    values = module.body[-3:]
    assert isinstance(values[0], nodes.Expr)
    assert isinstance(values[0].value, nodes.Const)
    assert values[0].value.value == 24
    assert isinstance(values[1], nodes.Expr)
    assert isinstance(values[1].value, nodes.Call)
    assert isinstance(values[2], nodes.Expr)
    assert isinstance(values[2].value, nodes.Const)
    assert values[2].value.value == 2


def test_transforms_are_separated():
    # Test that the transforming is done at a separate
    # step, which means that we are not doing inference
    # on a partially constructed tree anymore, which was the
    # source of crashes in the past when certain inference rules
    # were used in a transform.
    def transform_function(node):
        if node.decorators:
            for decorator in node.decorators.nodes:
                inferred = next(decorator.infer())
                if inferred.qname() == 'abc.abstractmethod':
                    return next(node.infer_call_result(node))

    manager = builder.MANAGER
    with add_transform(manager, nodes.FunctionDef, transform_function):
        module = builder.parse('''
        import abc
        from abc import abstractmethod

        class A(object):
            @abc.abstractmethod
            def ala(self):
                return 24

            @abstractmethod
            def bala(self):
                return 42
        ''')

    cls = module['A']
    ala = cls.body[0]
    bala = cls.body[1]
    assert isinstance(ala, nodes.Const)
    assert ala.value == 24
    assert isinstance(bala, nodes.Const)
    assert bala.value == 42


def test_transforms_are_called_for_builtin_modules():
    # Test that transforms are called for builtin modules.
    def transform_function(node):
        name = nodes.AssignName()
        name.name = 'value'
        node.args.args = [name]
        return node

    manager = builder.MANAGER
    predicate = lambda node: node.root().name == 'time'
    with add_transform(manager, nodes.FunctionDef,
                       transform_function, predicate):
        builder_instance = builder.AstroidBuilder()
        module = builder_instance.module_build(time)

    asctime = module['asctime']
    assert len(asctime.args.args) == 1
    assert isinstance(asctime.args.args[0], nodes.AssignName)
    assert asctime.args.args[0].name == 'value'


def test_builder_apply_transforms():
    def transform_function(node):
        return nodes.const_factory(42)

    manager = builder.MANAGER
    with add_transform(manager, nodes.FunctionDef, transform_function):
        astroid_builder = builder.AstroidBuilder(apply_transforms=False)
        module = astroid_builder.string_build('''def test(): pass''')

    # The transform wasn't applied.
    assert isinstance(module.body[0], nodes.FunctionDef)


def test_transform_crashes_on_is_subtype_of(transformer):
    # Test that we don't crash when having is_subtype_of
    # in a transform, as per issue #188. This happened
    # before, when the transforms weren't in their own step.
    def transform_class(cls):
        if cls.is_subtype_of('django.db.models.base.Model'):
            return cls
        return cls

    transformer.register_transform(nodes.ClassDef, transform_class)

    parse_transform(transformer, '''
        # Change environ to automatically call putenv() if it exists
        import os
        putenv = os.putenv
        try:
            # This will fail if there's no putenv
            putenv
        except NameError:
            pass
        else:
            import UserDict
    ''')
