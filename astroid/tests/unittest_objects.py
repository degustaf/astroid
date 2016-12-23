# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>
# Copyright (c) 2015-2016 Claudiu Popa <pcmanticore@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER


from astroid import bases
from astroid import builder
from astroid import exceptions
from astroid import nodes
from astroid import objects
from astroid import test_utils

import pytest


def assertEqualMro(klass, expected_mro):
    assert [member.name for member in klass.super_mro()] == \
        expected_mro


def test_frozenset():
    node = builder.extract_node("""
    frozenset({1: 2, 2: 3}) #@
    """)
    inferred = next(node.infer())
    assert isinstance(inferred, objects.FrozenSet)

    assert inferred.pytype() == "%s.frozenset" % bases.BUILTINS

    itered = inferred.itered()
    assert len(itered) == 2
    assert isinstance(itered[0], nodes.Const)
    assert [const.value for const in itered] == [1, 2]

    proxied = inferred._proxied
    assert inferred.qname() == "%s.frozenset" % bases.BUILTINS
    assert isinstance(proxied, nodes.ClassDef)


def test_inferring_super_outside_methods():
    ast_nodes = builder.extract_node('''
    class Module(object):
        pass
    class StaticMethod(object):
        @staticmethod
        def static():
            # valid, but we don't bother with it.
            return super(StaticMethod, StaticMethod) #@
    # super outside methods aren't inferred
    super(Module, Module) #@
    # no argument super is not recognised outside methods as well.
    super() #@
    ''')
    in_static = next(ast_nodes[0].value.infer())
    assert isinstance(in_static, bases.Instance)
    assert in_static.qname() == "%s.super" % bases.BUILTINS

    module_level = next(ast_nodes[1].infer())
    assert isinstance(module_level, bases.Instance)
    assert in_static.qname() == "%s.super" % bases.BUILTINS

    no_arguments = next(ast_nodes[2].infer())
    assert isinstance(no_arguments, bases.Instance)
    assert no_arguments.qname() == "%s.super" % bases.BUILTINS


def test_inferring_unbound_super_doesnt_work():
    node = builder.extract_node('''
    class Test(object):
        def __init__(self):
            super(Test) #@
    ''')
    unbounded = next(node.infer())
    assert isinstance(unbounded, bases.Instance)
    assert unbounded.qname() == "%s.super" % bases.BUILTINS


def test_use_default_inference_on_not_inferring_args():
    ast_nodes = builder.extract_node('''
    class Test(object):
        def __init__(self):
            super(Lala, self) #@
            super(Test, lala) #@
    ''')
    first = next(ast_nodes[0].infer())
    assert isinstance(first, bases.Instance)
    assert first.qname() == "%s.super" % bases.BUILTINS

    second = next(ast_nodes[1].infer())
    assert isinstance(second, bases.Instance)
    assert second.qname() == "%s.super" % bases.BUILTINS


@test_utils.require_version(maxver='3.0')
def test_super_on_old_style_class():
    # super doesn't work on old style class, but leave
    # that as an error for pylint. We'll infer Super objects,
    # but every call will result in a failure at some point.
    node = builder.extract_node('''
    class OldStyle:
        def __init__(self):
            super(OldStyle, self) #@
    ''')
    old = next(node.infer())
    assert isinstance(old, objects.Super)
    assert isinstance(old.mro_pointer, nodes.ClassDef)
    assert old.mro_pointer.name == 'OldStyle'
    with pytest.raises(exceptions.SuperError) as cm:
        old.super_mro()
    assert str(cm.value) == "Unable to call super on old-style classes."


@test_utils.require_version(minver='3.0')
def test_no_arguments_super():
    ast_nodes = builder.extract_node('''
    class First(object): pass
    class Second(First):
        def test(self):
            super() #@
        @classmethod
        def test_classmethod(cls):
            super() #@
    ''')
    first = next(ast_nodes[0].infer())
    assert isinstance(first, objects.Super)
    assert isinstance(first.type, bases.Instance)
    assert first.type.name == 'Second'
    assert isinstance(first.mro_pointer, nodes.ClassDef)
    assert first.mro_pointer.name == 'Second'

    second = next(ast_nodes[1].infer())
    assert isinstance(second, objects.Super)
    assert isinstance(second.type, nodes.ClassDef)
    assert second.type.name == 'Second'
    assert isinstance(second.mro_pointer, nodes.ClassDef)
    assert second.mro_pointer.name == 'Second'


def test_super_simple_cases():
    ast_nodes = builder.extract_node('''
    class First(object): pass
    class Second(First): pass
    class Third(First):
        def test(self):
            super(Third, self) #@
            super(Second, self) #@

            # mro position and the type
            super(Third, Third) #@
            super(Third, Second) #@
            super(Fourth, Fourth) #@

    class Fourth(Third):
        pass
    ''')

    # .type is the object which provides the mro.
    # .mro_pointer is the position in the mro from where
    # the lookup should be done.

    # super(Third, self)
    first = next(ast_nodes[0].infer())
    assert isinstance(first, objects.Super)
    assert isinstance(first.type, bases.Instance)
    assert first.type.name == 'Third'
    assert isinstance(first.mro_pointer, nodes.ClassDef)
    assert first.mro_pointer.name == 'Third'

    # super(Second, self)
    second = next(ast_nodes[1].infer())
    assert isinstance(second, objects.Super)
    assert isinstance(second.type, bases.Instance)
    assert second.type.name == 'Third'
    assert isinstance(first.mro_pointer, nodes.ClassDef)
    assert second.mro_pointer.name == 'Second'

    # super(Third, Third)
    third = next(ast_nodes[2].infer())
    assert isinstance(third, objects.Super)
    assert isinstance(third.type, nodes.ClassDef)
    assert third.type.name == 'Third'
    assert isinstance(third.mro_pointer, nodes.ClassDef)
    assert third.mro_pointer.name == 'Third'

    # super(Third, second)
    fourth = next(ast_nodes[3].infer())
    assert isinstance(fourth, objects.Super)
    assert isinstance(fourth.type, nodes.ClassDef)
    assert fourth.type.name == 'Second'
    assert isinstance(fourth.mro_pointer, nodes.ClassDef)
    assert fourth.mro_pointer.name == 'Third'

    # Super(Fourth, Fourth)
    fifth = next(ast_nodes[4].infer())
    assert isinstance(fifth, objects.Super)
    assert isinstance(fifth.type, nodes.ClassDef)
    assert fifth.type.name == 'Fourth'
    assert isinstance(fifth.mro_pointer, nodes.ClassDef)
    assert fifth.mro_pointer.name == 'Fourth'


def test_super_infer():
    node = builder.extract_node('''
    class Super(object):
        def __init__(self):
            super(Super, self) #@
    ''')
    inferred = next(node.infer())
    assert isinstance(inferred, objects.Super)
    reinferred = next(inferred.infer())
    assert isinstance(reinferred, objects.Super)
    assert inferred is reinferred


def test_inferring_invalid_supers():
    ast_nodes = builder.extract_node('''
    class Super(object):
        def __init__(self):
            # MRO pointer is not a type
            super(1, self) #@
            # MRO type is not a subtype
            super(Super, 1) #@
            # self is not a subtype of Bupper
            super(Bupper, self) #@
    class Bupper(Super):
        pass
    ''')
    first = next(ast_nodes[0].infer())
    assert isinstance(first, objects.Super)
    with pytest.raises(exceptions.SuperError) as cm:
        first.super_mro()
    assert isinstance(cm.value.super_.mro_pointer, nodes.Const)
    assert cm.value.super_.mro_pointer.value == 1
    for node, invalid_type in zip(ast_nodes[1:],
                                  (nodes.Const, bases.Instance)):
        inferred = next(node.infer())
        assert isinstance(inferred, objects.Super), node
        with pytest.raises(exceptions.SuperError) as cm:
            inferred.super_mro()
        assert isinstance(cm.value.super_.type, invalid_type)


def test_proxied():
    node = builder.extract_node('''
    class Super(object):
        def __init__(self):
            super(Super, self) #@
    ''')
    inferred = next(node.infer())
    proxied = inferred._proxied
    assert proxied.qname() == "%s.super" % bases.BUILTINS
    assert isinstance(proxied, nodes.ClassDef)


def test_super_bound_model():
    ast_nodes = builder.extract_node('''
    class First(object):
        def method(self):
            pass
        @classmethod
        def class_method(cls):
            pass
    class Super_Type_Type(First):
        def method(self):
            super(Super_Type_Type, Super_Type_Type).method #@
            super(Super_Type_Type, Super_Type_Type).class_method #@
        @classmethod
        def class_method(cls):
            super(Super_Type_Type, Super_Type_Type).method #@
            super(Super_Type_Type, Super_Type_Type).class_method #@

    class Super_Type_Object(First):
        def method(self):
            super(Super_Type_Object, self).method #@
            super(Super_Type_Object, self).class_method #@
    ''')
    # Super(type, type) is the same for both functions and classmethods.
    first = next(ast_nodes[0].infer())
    assert isinstance(first, nodes.FunctionDef)
    assert first.name == 'method'

    second = next(ast_nodes[1].infer())
    assert isinstance(second, bases.BoundMethod)
    assert second.bound.name == 'First'
    assert second.type == 'classmethod'

    third = next(ast_nodes[2].infer())
    assert isinstance(third, nodes.FunctionDef)
    assert third.name == 'method'

    fourth = next(ast_nodes[3].infer())
    assert isinstance(fourth, bases.BoundMethod)
    assert fourth.bound.name == 'First'
    assert fourth.type == 'classmethod'

    # Super(type, obj) can lead to different attribute bindings
    # depending on the type of the place where super was called.
    fifth = next(ast_nodes[4].infer())
    assert isinstance(fifth, bases.BoundMethod)
    assert fifth.bound.name == 'First'
    assert fifth.type == 'method'

    sixth = next(ast_nodes[5].infer())
    assert isinstance(sixth, bases.BoundMethod)
    assert sixth.bound.name == 'First'
    assert sixth.type == 'classmethod'


def test_super_getattr_single_inheritance():
    ast_nodes = builder.extract_node('''
    class First(object):
        def test(self): pass
    class Second(First):
        def test2(self): pass
    class Third(Second):
        test3 = 42
        def __init__(self):
            super(Third, self).test2 #@
            super(Third, self).test #@
            # test3 is local, no MRO lookup is done.
            super(Third, self).test3 #@
            super(Third, self) #@

            # Unbounds.
            super(Third, Third).test2 #@
            super(Third, Third).test #@

    ''')
    first = next(ast_nodes[0].infer())
    assert isinstance(first, bases.BoundMethod)
    assert first.bound.name == 'Second'

    second = next(ast_nodes[1].infer())
    assert isinstance(second, bases.BoundMethod)
    assert second.bound.name == 'First'

    with pytest.raises(exceptions.InferenceError):
        next(ast_nodes[2].infer())
    fourth = next(ast_nodes[3].infer())
    with pytest.raises(exceptions.AttributeInferenceError):
        fourth.getattr('test3')
    with pytest.raises(exceptions.AttributeInferenceError):
        next(fourth.igetattr('test3'))

    first_unbound = next(ast_nodes[4].infer())
    assert isinstance(first_unbound, nodes.FunctionDef)
    assert first_unbound.name == 'test2'
    assert first_unbound.parent.name == 'Second'

    second_unbound = next(ast_nodes[5].infer())
    assert isinstance(second_unbound, nodes.FunctionDef)
    assert second_unbound.name == 'test'
    assert second_unbound.parent.name == 'First'


def test_super_invalid_mro():
    node = builder.extract_node('''
    class A(object):
       test = 42
    class Super(A, A):
       def __init__(self):
           super(Super, self) #@
    ''')
    inferred = next(node.infer())
    with pytest.raises(exceptions.AttributeInferenceError):
        next(inferred.getattr('test'))


def test_super_complex_mro():
    ast_nodes = builder.extract_node('''
    class A(object):
        def spam(self): return "A"
        def foo(self): return "A"
        @staticmethod
        def static(self): pass
    class B(A):
        def boo(self): return "B"
        def spam(self): return "B"
    class C(A):
        def boo(self): return "C"
    class E(C, B):
        def __init__(self):
            super(E, self).boo #@
            super(C, self).boo #@
            super(E, self).spam #@
            super(E, self).foo #@
            super(E, self).static #@
    ''')
    first = next(ast_nodes[0].infer())
    assert isinstance(first, bases.BoundMethod)
    assert first.bound.name == 'C'
    second = next(ast_nodes[1].infer())
    assert isinstance(second, bases.BoundMethod)
    assert second.bound.name == 'B'
    third = next(ast_nodes[2].infer())
    assert isinstance(third, bases.BoundMethod)
    assert third.bound.name == 'B'
    fourth = next(ast_nodes[3].infer())
    assert fourth.bound.name == 'A'
    static = next(ast_nodes[4].infer())
    assert isinstance(static, nodes.FunctionDef)
    assert static.parent.scope().name == 'A'


def test_super_data_model():
    ast_nodes = builder.extract_node('''
    class X(object): pass
    class A(X):
        def __init__(self):
            super(A, self) #@
            super(A, A) #@
            super(X, A) #@
    ''')
    first = next(ast_nodes[0].infer())
    thisclass = first.getattr('__thisclass__')[0]
    assert isinstance(thisclass, nodes.ClassDef)
    assert thisclass.name == 'A'
    selfclass = first.getattr('__self_class__')[0]
    assert isinstance(selfclass, nodes.ClassDef)
    assert selfclass.name == 'A'
    self_ = first.getattr('__self__')[0]
    assert isinstance(self_, bases.Instance)
    assert self_.name == 'A'
    cls = first.getattr('__class__')[0]
    assert cls == first._proxied

    second = next(ast_nodes[1].infer())
    thisclass = second.getattr('__thisclass__')[0]
    assert thisclass.name == 'A'
    self_ = second.getattr('__self__')[0]
    assert isinstance(self_, nodes.ClassDef)
    assert self_.name == 'A'

    third = next(ast_nodes[2].infer())
    thisclass = third.getattr('__thisclass__')[0]
    assert thisclass.name == 'X'
    selfclass = third.getattr('__self_class__')[0]
    assert selfclass.name == 'A'


def test_super_mro():
    ast_nodes = builder.extract_node('''
    class A(object): pass
    class B(A): pass
    class C(A): pass
    class E(C, B):
        def __init__(self):
            super(E, self) #@
            super(C, self) #@
            super(B, self) #@

            super(B, 1) #@
            super(1, B) #@
    ''')
    first = next(ast_nodes[0].infer())
    assertEqualMro(first, ['C', 'B', 'A', 'object'])
    second = next(ast_nodes[1].infer())
    assertEqualMro(second, ['B', 'A', 'object'])
    third = next(ast_nodes[2].infer())
    assertEqualMro(third, ['A', 'object'])

    fourth = next(ast_nodes[3].infer())
    with pytest.raises(exceptions.SuperError):
        fourth.super_mro()
    fifth = next(ast_nodes[4].infer())
    with pytest.raises(exceptions.SuperError):
        fifth.super_mro()


def test_super_yes_objects():
    ast_nodes = builder.extract_node('''
    from collections import Missing
    class A(object):
        def __init__(self):
            super(Missing, self) #@
            super(A, Missing) #@
    ''')
    first = next(ast_nodes[0].infer())
    assert isinstance(first, bases.Instance)
    second = next(ast_nodes[1].infer())
    assert isinstance(second, bases.Instance)


def test_super_invalid_types():
    node = builder.extract_node('''
    import collections
    class A(object):
        def __init__(self):
            super(A, collections) #@
    ''')
    inferred = next(node.infer())
    with pytest.raises(exceptions.SuperError):
        inferred.super_mro()
    with pytest.raises(exceptions.SuperError):
        inferred.super_mro()


def test_super_properties():
    node = builder.extract_node('''
    class Foo(object):
        @property
        def dict(self):
            return 42

    class Bar(Foo):
        @property
        def dict(self):
            return super(Bar, self).dict

    Bar().dict
    ''')
    inferred = next(node.infer())
    assert isinstance(inferred, nodes.Const)
    assert inferred.value == 42
