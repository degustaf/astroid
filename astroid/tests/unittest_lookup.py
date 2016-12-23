# Copyright (c) 2007-2013 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2014-2016 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

"""tests for the astroid variable lookup capabilities
"""
import functools
import sys

from astroid import builder
from astroid import exceptions
from astroid import nodes
from astroid import scoped_nodes
from astroid import util
import pytest


def test_limit():
    code = '''
        l = [a
             for a,b in list]

        a = 1
        b = a
        a = None

        def func():
            c = 1
    '''
    astroid = builder.parse(code, __name__)
    # a & b
    a = next(astroid.nodes_of_class(nodes.Name))
    assert a.lineno == 2
    assert len(astroid.lookup('b')[1]) == 1
    assert len(astroid.lookup('a')[1]) == 1
    b = astroid.locals['b'][1 if sys.version_info < (3, 0) else 0]
    stmts = a.lookup('a')[1]
    assert len(stmts) == 1
    assert b.lineno == 6
    b_infer = b.infer()
    b_value = next(b_infer)
    assert b_value.value == 1
    # c
    with pytest.raises(StopIteration):
        functools.partial(next, b_infer)()
    func = astroid.locals['func'][0]
    assert len(func.lookup('c')[1]) == 1


def test_module(nonregr):
    astroid = builder.parse('pass', __name__)
    # built-in objects
    none = next(astroid.ilookup('None'))
    assert none.value is None
    obj = next(astroid.ilookup('object'))
    assert isinstance(obj, nodes.ClassDef)
    assert obj.name == 'object'
    with pytest.raises(exceptions.InferenceError):
        functools.partial(next, astroid.ilookup('YOAA'))()

    # XXX
    assert len(list(nonregr.ilookup('enumerate'))) == 2


def test_class_ancestor_name():
    code = '''
        class A:
            pass

        class A(A):
            pass
    '''
    astroid = builder.parse(code, __name__)
    cls1 = astroid.locals['A'][0]
    cls2 = astroid.locals['A'][1]
    name = next(cls2.nodes_of_class(nodes.Name))
    assert next(name.infer()) == cls1


### backport those test to inline code
def test_method(YOUPI_cls):
    method = YOUPI_cls['method']
    my_dict = next(method.ilookup('MY_DICT'))
    assert isinstance(my_dict, nodes.Dict), my_dict
    none = next(method.ilookup('None'))
    assert none.value is None
    with pytest.raises(exceptions.InferenceError):
        functools.partial(next, method.ilookup('YOAA'))()


@pytest.mark.skip(reason="TODO Why isn't this passing under pytest")
def test_function_argument_with_default(module2):
    make_class = module2['make_class']
    base = next(make_class.ilookup('base'))
    assert isinstance(base, nodes.ClassDef), base.__class__
    assert base.name == 'YO'
    assert base.root().name == 'data.module'


def test_class(YOUPI_cls):
    my_dict = next(YOUPI_cls.ilookup('MY_DICT'))
    assert isinstance(my_dict, nodes.Dict)
    none = next(YOUPI_cls.ilookup('None'))
    assert none.value is None
    obj = next(YOUPI_cls.ilookup('object'))
    assert isinstance(obj, nodes.ClassDef)
    assert obj.name == 'object'
    with pytest.raises(exceptions.InferenceError):
        functools.partial(next, YOUPI_cls.ilookup('YOAA'))()


def test_inner_classes(nonregr):
    ddd = list(nonregr['Ccc'].ilookup('Ddd'))
    assert ddd[0].name == 'Ddd'


def test_loopvar_hiding():
    astroid = builder.parse("""
        x = 10
        for x in range(5):
            print (x)

        if x > 0:
            print ('#' * x)
    """, __name__)
    xnames = [n for n in astroid.nodes_of_class(nodes.Name) if n.name == 'x']
    # inside the loop, only one possible assignment
    assert len(xnames[0].lookup('x')[1]) == 1
    # outside the loop, two possible assignments
    assert len(xnames[1].lookup('x')[1]) == 2
    assert len(xnames[2].lookup('x')[1]) == 2


def test_list_comps():
    astroid = builder.parse("""
        print ([ i for i in range(10) ])
        print ([ i for i in range(10) ])
        print ( list( i for i in range(10) ) )
    """, __name__)
    xnames = [n for n in astroid.nodes_of_class(nodes.Name) if n.name == 'i']
    assert len(xnames[0].lookup('i')[1]) == 1
    assert xnames[0].lookup('i')[1][0].lineno == 2
    assert len(xnames[1].lookup('i')[1]) == 1
    assert xnames[1].lookup('i')[1][0].lineno == 3
    assert len(xnames[2].lookup('i')[1]) == 1
    assert xnames[2].lookup('i')[1][0].lineno == 4


def test_list_comp_target():
    """test the list comprehension target"""
    astroid = builder.parse("""
        ten = [ var for var in range(10) ]
        var
    """)
    var = astroid.body[1].value
    if sys.version_info < (3, 0):
        assert var.inferred() == [util.Uninferable]
    else:
        with pytest.raises(exceptions.NameInferenceError):
            var.inferred()


def test_dict_comps():
    astroid = builder.parse("""
        print ({ i: j for i in range(10) for j in range(10) })
        print ({ i: j for i in range(10) for j in range(10) })
    """, __name__)
    xnames = [n for n in astroid.nodes_of_class(nodes.Name) if n.name == 'i']
    assert len(xnames[0].lookup('i')[1]) == 1
    assert xnames[0].lookup('i')[1][0].lineno == 2
    assert len(xnames[1].lookup('i')[1]) == 1
    assert xnames[1].lookup('i')[1][0].lineno == 3

    xnames = [n for n in astroid.nodes_of_class(nodes.Name) if n.name == 'j']
    assert len(xnames[0].lookup('i')[1]) == 1
    assert xnames[0].lookup('i')[1][0].lineno == 2
    assert len(xnames[1].lookup('i')[1]) == 1
    assert xnames[1].lookup('i')[1][0].lineno == 3


def test_set_comps():
    astroid = builder.parse("""
        print ({ i for i in range(10) })
        print ({ i for i in range(10) })
    """, __name__)
    xnames = [n for n in astroid.nodes_of_class(nodes.Name) if n.name == 'i']
    assert len(xnames[0].lookup('i')[1]) == 1
    assert xnames[0].lookup('i')[1][0].lineno == 2
    assert len(xnames[1].lookup('i')[1]) == 1
    assert xnames[1].lookup('i')[1][0].lineno == 3


def test_set_comp_closure():
    astroid = builder.parse("""
        ten = { var for var in range(10) }
        var
    """)
    var = astroid.body[1].value
    with pytest.raises(exceptions.NameInferenceError):
        var.inferred()


def test_generator_attributes():
    tree = builder.parse("""
        def count():
            "test"
            yield 0

        iterer = count()
        num = iterer.next()
    """)
    next_node = tree.body[2].value.func
    gener = next_node.expr.inferred()[0]
    if sys.version_info < (3, 0):
        assert isinstance(gener.getattr('next')[0], nodes.FunctionDef)
    else:
        assert isinstance(gener.getattr('__next__')[0], nodes.FunctionDef)
    assert isinstance(gener.getattr('send')[0], nodes.FunctionDef)
    assert isinstance(gener.getattr('throw')[0], nodes.FunctionDef)
    assert isinstance(gener.getattr('close')[0], nodes.FunctionDef)


def test_explicit___name__():
    code = '''
        class Pouet:
            __name__ = "pouet"
        p1 = Pouet()

        class PouetPouet(Pouet): pass
        p2 = Pouet()

        class NoName: pass
        p3 = NoName()
    '''
    astroid = builder.parse(code, __name__)
    p1 = next(astroid['p1'].infer())
    assert p1.getattr('__name__')
    p2 = next(astroid['p2'].infer())
    assert p2.getattr('__name__')
    assert astroid['NoName'].getattr('__name__')
    p3 = next(astroid['p3'].infer())
    with pytest.raises(exceptions.AttributeInferenceError):
        p3.getattr('__name__')


def test_function_module_special():
    astroid = builder.parse('''
    def initialize(linter):
        """initialize linter with checkers in this package """
        package_load(linter, __path__[0])
    ''', 'data.__init__')
    path = [n for n in astroid.nodes_of_class(nodes.Name) if n.name == '__path__'][0]
    assert len(path.lookup('__path__')[1]) == 1


def test_builtin_lookup():
    assert scoped_nodes.builtin_lookup('__dict__')[1] == ()
    intstmts = scoped_nodes.builtin_lookup('int')[1]
    assert len(intstmts) == 1
    assert isinstance(intstmts[0], nodes.ClassDef)
    assert intstmts[0].name == 'int'
    # pylint: disable=no-member; union type in const_factory, this shouldn't happen
    assert intstmts[0] is nodes.const_factory(1)._proxied


def test_decorator_arguments_lookup():
    code = '''
        def decorator(value):
            def wrapper(function):
                return function
            return wrapper

        class foo:
            member = 10  #@

            @decorator(member) #This will cause pylint to complain
            def test(self):
                pass
    '''
    member = builder.extract_node(code, __name__).targets[0]
    it = member.infer()
    obj = next(it)
    assert isinstance(obj, nodes.Const)
    assert obj.value == 10
    with pytest.raises(StopIteration):
        functools.partial(next, it)()


def test_inner_decorator_member_lookup():
    code = '''
        class FileA:
            def decorator(bla):
                return bla

            @__(decorator)
            def funcA():
                return 4
    '''
    decname = builder.extract_node(code, __name__)
    it = decname.infer()
    obj = next(it)
    assert isinstance(obj, nodes.FunctionDef)
    with pytest.raises(StopIteration):
        functools.partial(next, it)()


def test_static_method_lookup():
    code = '''
        class FileA:
            @staticmethod
            def funcA():
                return 4


        class Test:
            FileA = [1,2,3]

            def __init__(self):
                print (FileA.funcA())
    '''
    astroid = builder.parse(code, __name__)
    it = astroid['Test']['__init__'].ilookup('FileA')
    obj = next(it)
    assert isinstance(obj, nodes.ClassDef)
    with pytest.raises(StopIteration):
        functools.partial(next, it)()


def test_global_delete():
    code = '''
        def run2():
            f = Frobble()

        class Frobble:
            pass
        Frobble.mumble = True

        del Frobble

        def run1():
            f = Frobble()
    '''
    astroid = builder.parse(code, __name__)
    stmts = astroid['run2'].lookup('Frobbel')[1]
    assert len(stmts) == 0
    stmts = astroid['run1'].lookup('Frobbel')[1]
    assert len(stmts) == 0
