# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>
# Copyright (c) 2015-2016 Claudiu Popa <pcmanticore@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER


import six
from six.moves import builtins

from astroid import builder
from astroid import exceptions
from astroid import helpers
from astroid import manager
from astroid import raw_building
from astroid import test_utils
from astroid import util
import pytest


BUILTINS = manager.AstroidManager().astroid_cache[builtins.__name__]


def _extract(obj_name):
    return BUILTINS.getattr(obj_name)[0]


def _build_custom_builtin(obj_name):
    proxy = raw_building.build_class(obj_name)
    proxy.parent = BUILTINS
    return proxy


def assert_classes_equal(cls, other):
    assert cls.name == other.name
    assert cls.parent == other.parent
    assert cls.qname() == other.qname()


@pytest.mark.parametrize("code,expected", [
    ('1', _extract('int')),
    ('[]', _extract('list')),
    ('{1, 2, 3}', _extract('set')),
    ('{1:2, 4:3}', _extract('dict')),
    ('type', _extract('type')),
    ('object', _extract('type')),
    ('object()', _extract('object')),
    ('lambda: None', _build_custom_builtin('function')),
    ('len', _build_custom_builtin('builtin_function_or_method')),
    ('None', _build_custom_builtin('NoneType')),
    ('import sys\nsys#@', _build_custom_builtin('module')),
])
def test_object_type(code, expected):
    node = builder.extract_node(code)
    objtype = helpers.object_type(node)
    assert_classes_equal(objtype, expected)


def test_object_type_classes_and_functions():
    ast_nodes = builder.extract_node('''
    def generator():
        yield

    class A(object):
        def test(self):
            self #@
        @classmethod
        def cls_method(cls): pass
        @staticmethod
        def static_method(): pass
    A #@
    A() #@
    A.test #@
    A().test #@
    A.cls_method #@
    A().cls_method #@
    A.static_method #@
    A().static_method #@
    generator() #@
    ''')

    from_self = helpers.object_type(ast_nodes[0])
    cls = next(ast_nodes[1].infer())
    assert_classes_equal(from_self, cls)

    cls_type = helpers.object_type(ast_nodes[1])
    assert_classes_equal(cls_type, _extract('type'))

    instance_type = helpers.object_type(ast_nodes[2])
    cls = next(ast_nodes[2].infer())._proxied
    assert_classes_equal(instance_type, cls)

    expected_method_types = [
        (ast_nodes[3], 'instancemethod' if six.PY2 else 'function'),
        (ast_nodes[4], 'instancemethod' if six.PY2 else 'method'),
        (ast_nodes[5], 'instancemethod' if six.PY2 else 'method'),
        (ast_nodes[6], 'instancemethod' if six.PY2 else 'method'),
        (ast_nodes[7], 'function'),
        (ast_nodes[8], 'function'),
        (ast_nodes[9], 'generator'),
    ]
    for node, expected in expected_method_types:
        node_type = helpers.object_type(node)
        expected_type = _build_custom_builtin(expected)
        assert_classes_equal(node_type, expected_type)


@test_utils.require_version(minver='3.0')
def test_object_type_metaclasses():
    module = builder.parse('''
    import abc
    class Meta(metaclass=abc.ABCMeta):
        pass
    meta_instance = Meta()
    ''')
    meta_type = helpers.object_type(module['Meta'])
    assert_classes_equal(meta_type, module['Meta'].metaclass())

    meta_instance = next(module['meta_instance'].infer())
    instance_type = helpers.object_type(meta_instance)
    assert_classes_equal(instance_type, module['Meta'])


@test_utils.require_version(minver='3.0')
def test_object_type_most_derived():
    node = builder.extract_node('''
    class A(type):
        def __new__(*args, **kwargs):
             return type.__new__(*args, **kwargs)
    class B(object): pass
    class C(object, metaclass=A): pass

    # The most derived metaclass of D is A rather than type.
    class D(B , C): #@
        pass
    ''')
    metaclass = node.metaclass()
    assert metaclass.name == 'A'
    obj_type = helpers.object_type(node)
    assert metaclass == obj_type


def test_inference_errors():
    node = builder.extract_node('''
    from unknown import Unknown
    u = Unknown #@
    ''')
    assert helpers.object_type(node) == util.Uninferable


def test_object_type_too_many_types():
    node = builder.extract_node('''
    from unknown import Unknown
    def test(x):
        if x:
            return lambda: None
        else:
            return 1
    test(Unknown) #@
    ''')
    assert helpers.object_type(node) == util.Uninferable


def test_is_subtype():
    ast_nodes = builder.extract_node('''
    class int_subclass(int):
        pass
    class A(object): pass #@
    class B(A): pass #@
    class C(A): pass #@
    int_subclass() #@
    ''')
    cls_a = ast_nodes[0]
    cls_b = ast_nodes[1]
    cls_c = ast_nodes[2]
    int_subclass = ast_nodes[3]
    int_subclass = helpers.object_type(next(int_subclass.infer()))
    base_int = _extract('int')
    assert helpers.is_subtype(int_subclass, base_int)
    assert helpers.is_supertype(base_int, int_subclass)

    assert helpers.is_supertype(cls_a, cls_b)
    assert helpers.is_supertype(cls_a, cls_c)
    assert helpers.is_subtype(cls_b, cls_a)
    assert helpers.is_subtype(cls_c, cls_a)
    assert not helpers.is_subtype(cls_a, cls_b)
    assert not helpers.is_subtype(cls_a, cls_b)


@test_utils.require_version(maxver='3.0')
def test_is_subtype_supertype_old_style_classes():
    cls_a, cls_b = builder.extract_node('''
    class A: #@
        pass
    class B(A): #@
        pass
    ''')
    assert not helpers.is_subtype(cls_a, cls_b)
    assert not helpers.is_subtype(cls_b, cls_a)
    assert not helpers.is_supertype(cls_a, cls_b)
    assert not helpers.is_supertype(cls_b, cls_a)


def test_is_subtype_supertype_mro_error():
    cls_e, cls_f = builder.extract_node('''
    class A(object): pass
    class B(A): pass
    class C(A): pass
    class D(B, C): pass
    class E(C, B): pass #@
    class F(D, E): pass #@
    ''')
    assert not helpers.is_subtype(cls_e, cls_f)
    assert not helpers.is_subtype(cls_e, cls_f)
    with pytest.raises(exceptions._NonDeducibleTypeHierarchy):
        helpers.is_subtype(cls_f, cls_e)
    assert not helpers.is_supertype(cls_f, cls_e)


def test_is_subtype_supertype_unknown_bases():
    cls_a, cls_b = builder.extract_node('''
    from unknown import Unknown
    class A(Unknown): pass #@
    class B(A): pass #@
    ''')
    with pytest.raises(exceptions._NonDeducibleTypeHierarchy):
        helpers.is_subtype(cls_a, cls_b)
    with pytest.raises(exceptions._NonDeducibleTypeHierarchy):
        helpers.is_supertype(cls_a, cls_b)


def test_is_subtype_supertype_unrelated_classes():
    cls_a, cls_b = builder.extract_node('''
    class A(object): pass #@
    class B(object): pass #@
    ''')
    assert not helpers.is_subtype(cls_a, cls_b)
    assert not helpers.is_subtype(cls_b, cls_a)
    assert not helpers.is_supertype(cls_a, cls_b)
    assert not helpers.is_supertype(cls_b, cls_a)


def test_is_subtype_supertype_classes_no_type_ancestor():
    cls_a = builder.extract_node('''
    class A(object): #@
        pass
    ''')
    builtin_type = _extract('type')
    assert not helpers.is_supertype(builtin_type, cls_a)
    assert not helpers.is_subtype(cls_a, builtin_type)


def test_is_subtype_supertype_classes_metaclasses():
    cls_a = builder.extract_node('''
    class A(type): #@
        pass
    ''')
    builtin_type = _extract('type')
    assert helpers.is_supertype(builtin_type, cls_a)
    assert helpers.is_subtype(cls_a, builtin_type)
