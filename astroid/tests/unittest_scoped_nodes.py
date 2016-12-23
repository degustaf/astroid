# Copyright (c) 2006-2014 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2011, 2013-2015 Google, Inc.
# Copyright (c) 2013-2016 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>
# Copyright (c) 2015 Philip Lorenz <philip@bithub.de>
# Copyright (c) 2015 Rene Zhang <rz99@cornell.edu>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

"""tests for specific behaviour of astroid scoped nodes (i.e. module, class and
function)
"""
import os
import sys
from functools import partial
import warnings

from astroid import builder
from astroid import nodes
from astroid import scoped_nodes
from astroid import util
from astroid.exceptions import (
    InferenceError, AttributeInferenceError,
    NoDefault, ResolveError, MroError,
    InconsistentMroError, DuplicateBasesError,
    TooManyLevelsError,
)
from astroid.bases import (
    BUILTINS, Instance,
    BoundMethod, UnboundMethod, Generator
)
from astroid import __pkginfo__
from astroid import test_utils
from astroid.tests import resources
import pytest
from .conftest import module, module2, YO_cls


@pytest.fixture
@pytest.mark.usefixture('sys_path_setup')
def pack():
    return resources.build_file('data/__init__.py', 'data')


@pytest.fixture
def four_args_func(module):
    return module['four_args']


@pytest.fixture
def make_class_func(module2):
    return module2['make_class']


@pytest.fixture
def specialization_cls(module2):
    return module2['Specialization']


@pytest.mark.parametrize("node,test_attr", [
    (module(), 'YO'),
    (module()['global_access'], 'local'),
    (module()['YOUPI'], 'method'),
])
def test_dict_interface(node, test_attr):
    assert node[test_attr] is node[test_attr]
    assert test_attr in node
    node.keys()
    node.values()
    node.items()
    iter(node)


@pytest.mark.parametrize("node,name,value", [
    (module(), '__name__', 'data.module'),
    (module(), '__doc__', 'test module for astroid\n'),
    (module(), '__file__', os.path.abspath(resources.find('data/module.py'))),
    (make_class_func(module2()), '__name__', 'make_class'),
    (make_class_func(module2()), '__doc__', 'check base is correctly resolved to Concrete0'),
    (YO_cls(module()), '__name__', 'YO'),
    (YO_cls(module()), '__module__', 'data.module'),
    (YO_cls(module()), '__doc__', 'hehe'),
])
def test_special_attributes_const(node, name, value):
    attr = node.getattr(name)
    assert len(attr) == 1
    assert isinstance(attr[0], nodes.Const)
    assert attr[0].value == value


@pytest.mark.parametrize("node,name,_type", [
    (module(), '__dict__', nodes.Dict),
    (pack(), '__path__', nodes.List),
])
def test_special_attributes2(node, name, _type):
    attr = node.getattr(name)
    assert len(attr) == 1
    assert isinstance(attr[0], _type)


def test_special_attributes_error(module):
    with pytest.raises(AttributeInferenceError):
        module.getattr('__path__')


@pytest.mark.parametrize("node,name,_type", [
    (module().getattr('YO')[0], 'YO', nodes.ClassDef),
    (next(module().igetattr('redirect')), 'four_args',
        nodes.FunctionDef),
    (next(module().igetattr('NameNode')), 'Name', nodes.ClassDef),
])
def test_getattr(node, name, _type):
    assert node.name == name
    assert isinstance(node, _type)


@pytest.mark.skip(reason="Find out why this fails under pytest")
def test_getattr2():
    # resolve packageredirection
    mod = resources.build_file('data/appl/myConnection.py',
                               'data.appl.myConnection')
    ssl = next(mod.igetattr('SSL1'))
    cnx = next(ssl.igetattr('Connection'))
    assert cnx.__class__ == nodes.ClassDef
    assert cnx.name == 'Connection'
    assert cnx.root().name == 'data.SSL1.Connection1'


def test_getattr3(nonregr):
    assert len(nonregr.getattr('enumerate')) == 2
    with pytest.raises(InferenceError):
        nonregr.igetattr('YOAA')


@pytest.mark.usefixture('sys_path_setup')
@pytest.mark.parametrize("_file,module_name,names", [
    ('data/all.py', 'all', ['Aaa', '_bla', 'name']),
    ('data/notall.py', 'notall', ['Aaa', 'func', 'name', 'other']),

])
def test_wildcard_import_names(_file, module_name, names):
    m = resources.build_file(_file, module_name)
    res = sorted(m.wildcard_import_names())
    assert res == names


@pytest.mark.parametrize("code,expected", [
    ('''
    name = 'a'
    _bla = 2
    other = 'o'
    class Aaa: pass
    def func(): print('yo')
    __all__ = 'Aaa', '_bla', 'name'
    ''', ['Aaa', 'func', 'name', 'other']),

    ('''
    name = 'a'
    _bla = 2
    other = 'o'
    class Aaa: pass

    def func(): return 'yo'
    ''', ['Aaa', 'func', 'name', 'other']),

    ('''
        from missing import tzop
        trop = "test"
        __all__ = (trop, "test1", tzop, 42)
    ''', ["trop", "tzop"]),

    ('''
        test = tzop = 42
        __all__ = ('test', ) + ('tzop', )
    ''', ['test', 'tzop']),

])
def test_public_names(code, expected):
    m = builder.parse(code)
    res = sorted(m.public_names())
    assert res == expected


def test_module_getattr():
    data = '''
        appli = application
        appli += 2
        del appli
    '''
    astroid = builder.parse(data, __name__)
    # test del statement not returned by getattr
    assert len(astroid.getattr('appli')) == 2


def mod1():
    mod = nodes.Module('very.multi.package', 'doc')
    mod.package = True
    return mod


def mod2():
    mod = nodes.Module('very.multi.module', 'doc')
    mod.package = False
    return mod


@pytest.mark.parametrize("mod,rel_name,i,expected", [
    # package
    (mod1(), 'utils', 0, 'very.multi.package.utils'),
    (mod1(), 'utils', 1, 'very.multi.package.utils'),
    (mod1(), 'utils', 2, 'very.multi.utils'),
    (mod1(), '', 1, 'very.multi.package'),
    # non package
    (mod2(), 'utils', 0, 'very.multi.utils'),
    (mod2(), 'utils', 1, 'very.multi.utils'),
    (mod2(), 'utils', 2, 'very.utils'),
    (mod2(), '', 1, 'very.multi'),
])
def test_relative_to_absolute_name(mod, rel_name, i, expected):
    assert mod.relative_to_absolute_name(rel_name, i) == expected


def test_relative_to_absolute_name_beyond_top_level():
    mod = nodes.Module('a.b.c', '')
    mod.package = True
    for level in (5, 4):
        with pytest.raises(TooManyLevelsError) as cm:
            mod.relative_to_absolute_name('test', level)

        expected = ("Relative import with too many levels "
                    "({level}) for module {name!r}".format(
                        level=level - 1, name=mod.name))
        assert expected == str(cm.value)


@pytest.mark.usefixture('sys_path_setup')
@pytest.mark.parametrize("data,name", [
    ('''from . import subpackage''', 'subpackage'),
    ('''from . import subpackage as pouet''', 'pouet'),

])
def test_import(data, name):
    sys.path.insert(0, resources.find('data'))
    astroid = builder.parse(data, 'package', 'data/package/__init__.py')
    try:
        m = astroid.import_module('', level=1)
        assert m.name == 'package'
        inferred = list(astroid.igetattr(name))
        assert len(inferred) == 1
        assert inferred[0].name == 'package.subpackage'
    finally:
        del sys.path[0]


def test_file_stream_in_memory():
    data = '''irrelevant_variable is irrelevant'''
    astroid = builder.parse(data, 'in_memory')
    with warnings.catch_warnings(record=True):
        assert astroid.file_stream.read().decode() == data


@pytest.mark.usefixture('sys_path_setup')
def test_file_stream_physical():
    path = resources.find('data/all.py')
    astroid = builder.AstroidBuilder().file_build(path, 'all')
    with open(path, 'rb') as file_io:
        with warnings.catch_warnings(record=True):
            assert astroid.file_stream.read() == file_io.read()


@pytest.mark.usefixture('sys_path_setup')
def test_file_stream_api():
    path = resources.find('data/all.py')
    astroid = builder.AstroidBuilder().file_build(path, 'all')
    if __pkginfo__.numversion >= (1, 6):
        # file_stream is slated for removal in astroid 1.6.
        with pytest.raises(AttributeError):
            # pylint: disable=pointless-statement
            astroid.file_stream
    else:
        # Until astroid 1.6, Module.file_stream will emit
        # PendingDeprecationWarning in 1.4, DeprecationWarning
        # in 1.5 and finally it will be removed in 1.6, leaving
        # only Module.stream as the recommended way to retrieve
        # its file stream.
        with warnings.catch_warnings(record=True) as cm:
            with test_utils.enable_warning(PendingDeprecationWarning):
                assert astroid.file_stream is not astroid.file_stream
        assert len(cm) > 1
        assert cm[0].category == PendingDeprecationWarning


@pytest.mark.usefixture('sys_path_setup')
def test_stream_api():
    path = resources.find('data/all.py')
    astroid = builder.AstroidBuilder().file_build(path, 'all')
    stream = astroid.stream()
    assert hasattr(stream, 'close')
    with stream:
        with open(path, 'rb') as file_io:
            assert stream.read() == file_io.read()


def test_default_value_type(make_class_func):
    assert isinstance(make_class_func.args.default_value('base'), nodes.Attribute)


@pytest.mark.parametrize("attr", ['args', 'kwargs', 'any'])
def test_default_value(make_class_func, attr):
    with pytest.raises(NoDefault):
        make_class_func.args.default_value('any')


def test_navigation(module):
    function = module['global_access']
    assert function.statement() == function
    l_sibling = function.previous_sibling()
    # check taking parent if child is not a stmt
    assert isinstance(l_sibling, nodes.Assign)
    child = function.args.args[0]
    assert l_sibling is child.previous_sibling()
    r_sibling = function.next_sibling()
    assert isinstance(r_sibling, nodes.ClassDef)
    assert r_sibling.name == 'YO'
    assert r_sibling is child.next_sibling()
    last = r_sibling.next_sibling().next_sibling().next_sibling()
    assert isinstance(last, nodes.Assign)
    assert last.next_sibling() is None
    first = l_sibling.root().body[0]
    assert first.previous_sibling() is None


@test_utils.require_version(maxver='3.0')
def test_nested_args():
    code = '''
        def nested_args(a, (b, c, d)):
            "nested arguments test"
    '''
    tree = builder.parse(code)
    func = tree['nested_args']
    assert sorted(func.locals) == ['a', 'b', 'c', 'd']
    assert func.args.format_args() == 'a, (b, c, d)'


def test_four_args(four_args_func):
    assert sorted(four_args_func.keys()) == ['a', 'b', 'c', 'd']
    assert four_args_func.type == 'function'


def test_format_args(make_class_func):
    assert make_class_func.args.format_args() == \
        'any, base=data.module.YO, *args, **kwargs'


def test_format_args2(four_args_func):
    assert four_args_func.args.format_args() == 'a, b, c, d'


def test_is_generator(module2):
    assert module2['generator'].is_generator()


@pytest.mark.parametrize("node", [
    ('not_a_generator'),
    ('make_class'),
])
def test_is_not_a_generator(module2, node):
    assert not module2[node].is_generator()


def test_is_abstract(module2):
    method = module2['AbstractClass']['to_override']
    assert method.is_abstract(pass_is_abstract=False)
    assert method.qname() == 'data.module2.AbstractClass.to_override'
    assert method.pytype() == '%s.instancemethod' % BUILTINS
    method = module2['AbstractClass']['return_something']
    assert not method.is_abstract(pass_is_abstract=False)
    # non regression : test raise "string" doesn't cause an exception in is_abstract
    func = module2['raise_string']
    assert not func.is_abstract(pass_is_abstract=False)


def test_is_abstract_decorated():
    methods = builder.extract_node("""
        import abc

        class Klass(object):
            @abc.abstractproperty
            def prop(self):  #@
               pass

            @abc.abstractmethod
            def method1(self):  #@
               pass

            some_other_decorator = lambda x: x
            @some_other_decorator
            def method2(self):  #@
               pass
     """)
    assert methods[0].is_abstract(pass_is_abstract=False)
    assert methods[1].is_abstract(pass_is_abstract=False)
    assert not methods[2].is_abstract(pass_is_abstract=False)


def test_lambda_pytype():
    data = '''
        def f():
            g = lambda: None
    '''
    astroid = builder.parse(data)
    g = list(astroid['f'].ilookup('g'))[0]
    assert g.pytype() == '%s.function' % BUILTINS


def test_lambda_qname():
    astroid = builder.parse('lmbd = lambda: None', __name__)
    assert '%s.<lambda>' % __name__ == astroid['lmbd'].parent.value.qname()


def test_is_method():
    data = '''
        class A:
            def meth1(self):
                return 1
            @classmethod
            def meth2(cls):
                return 2
            @staticmethod
            def meth3():
                return 3

        def function():
            return 0

        @staticmethod
        def sfunction():
            return -1
    '''
    astroid = builder.parse(data)
    assert astroid['A']['meth1'].is_method()
    assert astroid['A']['meth2'].is_method()
    assert astroid['A']['meth3'].is_method()
    assert not astroid['function'].is_method()
    assert not astroid['sfunction'].is_method()


def test_argnames():
    if sys.version_info < (3, 0):
        code = 'def f(a, (b, c), *args, **kwargs): pass'
    else:
        code = 'def f(a, b, c, *args, **kwargs): pass'
    astroid = builder.parse(code, __name__)
    assert astroid['f'].argnames() == ['a', 'b', 'c', 'args', 'kwargs']


def test_return_nothing():
    """test inferred value on a function with empty return"""
    data = '''
        def func():
            return

        a = func()
    '''
    astroid = builder.parse(data)
    call = astroid.body[1].value
    func_vals = call.inferred()
    assert len(func_vals) == 1
    assert isinstance(func_vals[0], nodes.Const)
    assert func_vals[0].value is None


def test_func_instance_attr():
    """test instance attributes for functions"""
    data = """
        def test():
            print(test.bar)

        test.bar = 1
        test()
    """
    astroid = builder.parse(data, 'mod')
    func = astroid.body[2].value.func.inferred()[0]
    assert isinstance(func, nodes.FunctionDef)
    assert func.name == 'test'
    one = func.getattr('bar')[0].inferred()[0]
    assert isinstance(one, nodes.Const)
    assert one.value == 1


def test_type_builtin_descriptor_subclasses():
    astroid = builder.parse("""
        class classonlymethod(classmethod):
            pass
        class staticonlymethod(staticmethod):
            pass

        class Node:
            @classonlymethod
            def clsmethod_subclass(cls):
                pass
            @classmethod
            def clsmethod(cls):
                pass
            @staticonlymethod
            def staticmethod_subclass(cls):
                pass
            @staticmethod
            def stcmethod(cls):
                pass
    """)
    node = astroid.locals['Node'][0]
    assert node.locals['clsmethod_subclass'][0].type == 'classmethod'
    assert node.locals['clsmethod'][0].type == 'classmethod'
    assert node.locals['staticmethod_subclass'][0].type == 'staticmethod'
    assert node.locals['stcmethod'][0].type == 'staticmethod'


@pytest.mark.usefixture('sys_path_setup')
def test_decorator_builtin_descriptors():
    astroid = builder.parse("""
        def static_decorator(platform=None, order=50):
            def wrapper(f):
                f.cgm_module = True
                f.cgm_module_order = order
                f.cgm_module_platform = platform
                return staticmethod(f)
            return wrapper

        def long_classmethod_decorator(platform=None, order=50):
            def wrapper(f):
                def wrapper2(f):
                    def wrapper3(f):
                        f.cgm_module = True
                        f.cgm_module_order = order
                        f.cgm_module_platform = platform
                        return classmethod(f)
                    return wrapper3(f)
                return wrapper2(f)
            return wrapper

        def classmethod_decorator(platform=None):
            def wrapper(f):
                f.platform = platform
                return classmethod(f)
            return wrapper

        def classmethod_wrapper(fn):
            def wrapper(cls, *args, **kwargs):
                result = fn(cls, *args, **kwargs)
                return result

            return classmethod(wrapper)

        def staticmethod_wrapper(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return staticmethod(wrapper)

        class SomeClass(object):
            @static_decorator()
            def static(node, cfg):
                pass
            @classmethod_decorator()
            def classmethod(cls):
                pass
            @static_decorator
            def not_so_static(node):
                pass
            @classmethod_decorator
            def not_so_classmethod(node):
                pass
            @classmethod_wrapper
            def classmethod_wrapped(cls):
                pass
            @staticmethod_wrapper
            def staticmethod_wrapped():
                pass
            @long_classmethod_decorator()
            def long_classmethod(cls):
                pass
    """)
    node = astroid.locals['SomeClass'][0]
    assert node.locals['static'][0].type == 'staticmethod'
    assert node.locals['classmethod'][0].type == 'classmethod'
    assert node.locals['not_so_static'][0].type == 'method'
    assert node.locals['not_so_classmethod'][0].type == 'method'
    assert node.locals['classmethod_wrapped'][0].type == 'classmethod'
    assert node.locals['staticmethod_wrapped'][0].type == 'staticmethod'
    assert node.locals['long_classmethod'][0].type == 'classmethod'


@pytest.mark.usefixture('sys_path_setup')
def test_igetattr():
    func = builder.extract_node('''
    def test():
        pass
    ''')
    func.instance_attrs['value'] = [nodes.Const(42)]
    value = func.getattr('value')
    assert len(value) == 1
    assert isinstance(value[0], nodes.Const)
    assert value[0].value == 42
    inferred = next(func.igetattr('value'))
    assert isinstance(inferred, nodes.Const)
    assert inferred.value == 42


@test_utils.require_version(minver='3.0')
def test_return_annotation_is_not_the_last():
    func = builder.extract_node('''
    def test() -> bytes:
        pass
        pass
        return
    ''')
    last_child = func.last_child()
    assert isinstance(last_child, nodes.Return)
    assert func.tolineno == 5


def test_cls_special_attributes_1(YO_cls):
    if not YO_cls.newstyle:
        with pytest.raises(AttributeInferenceError):
            YO_cls.getattr('__mro__')


@pytest.mark.parametrize("cls", [
    nodes.List._proxied, nodes.Const(1)._proxied, module()['YO']
])
@pytest.mark.parametrize("attr", [
    '__bases__', '__name__', '__doc__', '__module__', '__dict__', '__mro__'
])
def test_cls_special_attributes_3(cls, attr):
        if sys.version_info[0] == 2 and cls.name == 'YO' and attr == '__mro__':
            pytest.skip()
        assert len(cls.getattr(attr)) == 1


@pytest.mark.parametrize("cls", [
    nodes.List._proxied, nodes.Const(1)._proxied, module()['YO']
])
def test_cls_special_attributes_4(cls):
        assert cls.getattr('__doc__')[0].value == cls.doc


def test__mro__attribute():
    node = builder.extract_node('''
    class A(object): pass
    class B(object): pass
    class C(A, B): pass
    ''')
    mro = node.getattr('__mro__')[0]
    assert isinstance(mro, nodes.Tuple)
    assert mro.elts == node.mro()


def test__bases__attribute():
    node = builder.extract_node('''
    class A(object): pass
    class B(object): pass
    class C(A, B): pass
    class D(C): pass
    ''')
    bases = node.getattr('__bases__')[0]
    assert isinstance(bases, nodes.Tuple)
    assert len(bases.elts) == 1
    assert isinstance(bases.elts[0], nodes.ClassDef)
    assert bases.elts[0].name == 'C'


def test_cls_special_attributes_2():
    astroid = builder.parse('''
        class A(object): pass
        class B(object): pass

        A.__bases__ += (B,)
    ''', __name__)
    assert len(astroid['A'].getattr('__bases__')) == 2
    assert isinstance(astroid['A'].getattr('__bases__')[1], nodes.Tuple)
    assert isinstance(astroid['A'].getattr('__bases__')[0], nodes.AssignAttr)


@pytest.mark.parametrize("inst", [Instance(YO_cls(module())),
                                  nodes.List(), nodes.Const(1)])
@pytest.mark.parametrize("attr", ['__mro__', '__bases__', '__name__'])
def test_instance_special_attributes_error(inst, attr):
    with pytest.raises(AttributeInferenceError):
        inst.getattr(attr)


@pytest.mark.parametrize("inst", [Instance(YO_cls(module())),
                                  nodes.List(), nodes.Const(1)])
@pytest.mark.parametrize("attr", ['__dict__', '__doc__'])
def test_instance_special_attributes(inst, attr):
    assert len(inst.getattr(attr)) == 1


def test_navigation2(YO_cls):
    assert YO_cls.statement() == YO_cls
    l_sibling = YO_cls.previous_sibling()
    assert isinstance(l_sibling, nodes.FunctionDef)
    assert l_sibling.name == 'global_access'
    r_sibling = YO_cls.next_sibling()
    assert isinstance(r_sibling, nodes.ClassDef)
    assert r_sibling.name == 'YOUPI'


def test_local_attr_ancestors():
    module = builder.parse('''
    class A():
        def __init__(self): pass
    class B(A): pass
    class C(B): pass
    class D(object): pass
    class F(): pass
    class E(F, D): pass
    ''')
    # Test old-style (Python 2) / new-style (Python 3+) ancestors lookups
    klass2 = module['C']
    it = klass2.local_attr_ancestors('__init__')
    anc_klass = next(it)
    assert isinstance(anc_klass, nodes.ClassDef)
    assert anc_klass.name == 'A'
    if sys.version_info[0] == 2:
        with pytest.raises(StopIteration):
            partial(next, it)()
    else:
        anc_klass = next(it)
        assert isinstance(anc_klass, nodes.ClassDef)
        assert anc_klass.name == 'object'
        with pytest.raises(StopIteration):
            partial(next, it)()

    it = klass2.local_attr_ancestors('method')
    with pytest.raises(StopIteration):
        partial(next, it)()

    # Test mixed-style ancestor lookups
    klass2 = module['E']
    it = klass2.local_attr_ancestors('__init__')
    anc_klass = next(it)
    assert isinstance(anc_klass, nodes.ClassDef)
    assert anc_klass.name == 'object'
    with pytest.raises(StopIteration):
        partial(next, it)()


def test_local_attr_mro():
    module = builder.parse('''
    class A(object):
        def __init__(self): pass
    class B(A):
        def __init__(self, arg, arg2): pass
    class C(A): pass
    class D(C, B): pass
    ''')
    dclass = module['D']
    init = dclass.local_attr('__init__')[0]
    assert isinstance(init, nodes.FunctionDef)
    assert init.parent.name == 'B'

    cclass = module['C']
    init = cclass.local_attr('__init__')[0]
    assert isinstance(init, nodes.FunctionDef)
    assert init.parent.name == 'A'

    ancestors = list(dclass.local_attr_ancestors('__init__'))
    assert [node.name for node in ancestors] == ['B', 'A', 'object']


def test_instance_attr_ancestors(YOUPI_cls):
    it = YOUPI_cls.instance_attr_ancestors('yo')
    anc_klass = next(it)
    assert isinstance(anc_klass, nodes.ClassDef)
    assert anc_klass.name == 'YO'
    with pytest.raises(StopIteration):
        partial(next, it)()

    it = YOUPI_cls.instance_attr_ancestors('member')
    with pytest.raises(StopIteration):
        partial(next, it)()


def test_methods(YOUPI_cls):
    expected_methods = {'__init__', 'class_method', 'method', 'static_method'}
    methods = {m.name for m in YOUPI_cls.methods()}
    assert methods.issuperset(expected_methods)
    methods = {m.name for m in YOUPI_cls.mymethods()}
    assert expected_methods == methods


def test_methods2(specialization_cls):
    expected_methods = {'__init__', 'class_method', 'method', 'static_method'}
    methods = {m.name for m in specialization_cls.mymethods()}
    assert set([]) == methods
    method_locals = specialization_cls.local_attr('method')
    assert len(method_locals) == 1
    assert method_locals[0].name == 'method'
    with pytest.raises(AttributeInferenceError):
        specialization_cls.local_attr('nonexistent')
    methods = {m.name for m in specialization_cls.methods()}
    assert methods.issuperset(expected_methods)


@test_utils.require_version(maxver='3.0')
def test_ancestors(YOUPI_cls, specialization_cls):
    assert ['YO'] == [a.name for a in YOUPI_cls.ancestors()]
    assert ['YOUPI', 'YO'] == [a.name for a in specialization_cls.ancestors()]


@test_utils.require_version(minver='3.0')
def test_ancestors_py3(YOUPI_cls, specialization_cls):
    assert ['YO', 'object'] == [a.name for a in YOUPI_cls.ancestors()]
    assert ['YOUPI', 'YO', 'object'] == [a.name for a in specialization_cls.ancestors()]


def test_type(YOUPI_cls, module2):
    assert YOUPI_cls.type == 'class'
    klass = module2['Metaclass']
    assert klass.type == 'metaclass'
    klass = module2['MyException']
    assert klass.type == 'exception'
    klass = module2['MyError']
    assert klass.type == 'exception'
    # the following class used to be detected as a metaclass
    # after the fix which used instance._proxied in .ancestors(),
    # when in fact it is a normal class
    klass = module2['NotMetaclass']
    assert klass.type == 'class'


def test_inner_classes(nonregr):
    eee = nonregr['Ccc']['Eee']
    assert [n.name for n in eee.ancestors()] == ['Ddd', 'Aaa', 'object']


def test_classmethod_attributes():
    data = '''
        class WebAppObject(object):
            def registered(cls, application):
                cls.appli = application
                cls.schema = application.schema
                cls.config = application.config
                return cls
            registered = classmethod(registered)
    '''
    astroid = builder.parse(data, __name__)
    cls = astroid['WebAppObject']
    assert sorted(cls.locals.keys()) == \
        ['appli', 'config', 'registered', 'schema']


def test_class_getattr():
    data = '''
        class WebAppObject(object):
            appli = application
            appli += 2
            del self.appli
    '''
    astroid = builder.parse(data, __name__)
    cls = astroid['WebAppObject']
    # test del statement not returned by getattr
    assert len(cls.getattr('appli')) == 2


def test_instance_getattr():
    data = '''
        class WebAppObject(object):
            def __init__(self, application):
                self.appli = application
                self.appli += 2
                del self.appli
     '''
    astroid = builder.parse(data)
    inst = Instance(astroid['WebAppObject'])
    # test del statement not returned by getattr
    assert len(inst.getattr('appli')) == 2


def test_instance_getattr_with_class_attr():
    data = '''
        class Parent:
            aa = 1
            cc = 1

        class Klass(Parent):
            aa = 0
            bb = 0

            def incr(self, val):
                self.cc = self.aa
                if val > self.aa:
                    val = self.aa
                if val < self.bb:
                    val = self.bb
                self.aa += val
    '''
    astroid = builder.parse(data)
    inst = Instance(astroid['Klass'])
    assert len(inst.getattr('aa')) == 3
    assert len(inst.getattr('bb')) == 1
    assert len(inst.getattr('cc')) == 2


def test_getattr_method_transform():
    data = '''
        class Clazz(object):

            def m1(self, value):
                self.value = value
            m2 = m1

        def func(arg1, arg2):
            "function that will be used as a method"
            return arg1.value + arg2

        Clazz.m3 = func
        inst = Clazz()
        inst.m4 = func
    '''
    astroid = builder.parse(data)
    cls = astroid['Clazz']
    # test del statement not returned by getattr
    for method in ('m1', 'm2', 'm3'):
        inferred = list(cls.igetattr(method))
        assert len(inferred) == 1
        assert isinstance(inferred[0], UnboundMethod)

        inferred = list(Instance(cls).igetattr(method))
        assert len(inferred) == 1
        assert isinstance(inferred[0], BoundMethod)

    inferred = list(Instance(cls).igetattr('m4'))
    assert len(inferred) == 1
    assert isinstance(inferred[0], nodes.FunctionDef)


def test_getattr_from_grandpa():
    data = '''
        class Future:
            attr = 1

        class Present(Future):
            pass

        class Past(Present):
            pass
    '''
    astroid = builder.parse(data)
    past = astroid['Past']
    attr = past.getattr('attr')
    assert len(attr) == 1
    attr1 = attr[0]
    assert isinstance(attr1, nodes.AssignName)
    assert attr1.name == 'attr'


def test_function_with_decorator_lineno():
    data = '''
        @f(a=2,
           b=3)
        def g1(x):
            print(x)

        @f(a=2,
           b=3)
        def g2():
            pass
    '''
    astroid = builder.parse(data)
    assert astroid['g1'].fromlineno == 4
    assert astroid['g1'].tolineno == 5
    assert astroid['g2'].fromlineno == 9
    assert astroid['g2'].tolineno == 10


@test_utils.require_version(maxver='3.0')
def test_simple_metaclass():
    astroid = builder.parse("""
        class Test(object):
            __metaclass__ = type
    """)
    klass = astroid['Test']
    metaclass = klass.metaclass()
    assert isinstance(metaclass, scoped_nodes.ClassDef)
    assert metaclass.name == 'type'


def test_metaclass_error():
    astroid = builder.parse("""
        class Test(object):
            __metaclass__ = typ
    """)
    klass = astroid['Test']
    assert not klass.metaclass()


@test_utils.require_version(maxver='3.0')
def test_metaclass_imported():
    astroid = builder.parse("""
        from abc import ABCMeta
        class Test(object):
            __metaclass__ = ABCMeta
    """)
    klass = astroid['Test']

    metaclass = klass.metaclass()
    assert isinstance(metaclass, scoped_nodes.ClassDef)
    assert metaclass.name == 'ABCMeta'


def test_metaclass_yes_leak():
    astroid = builder.parse("""
        # notice `ab` instead of `abc`
        from ab import ABCMeta

        class Meta(object):
            __metaclass__ = ABCMeta
    """)
    klass = astroid['Meta']
    assert klass.metaclass() is None


@test_utils.require_version(maxver='3.0')
def test_newstyle_and_metaclass_good():
    astroid = builder.parse("""
        from abc import ABCMeta
        class Test:
            __metaclass__ = ABCMeta
    """)
    klass = astroid['Test']
    assert klass.newstyle
    assert klass.metaclass().name == 'ABCMeta'
    astroid = builder.parse("""
        from abc import ABCMeta
        __metaclass__ = ABCMeta
        class Test:
            pass
    """)
    klass = astroid['Test']
    assert klass.newstyle
    assert klass.metaclass().name == 'ABCMeta'


@test_utils.require_version(maxver='3.0')
def test_nested_metaclass():
    astroid = builder.parse("""
        from abc import ABCMeta
        class A(object):
            __metaclass__ = ABCMeta
            class B: pass

        __metaclass__ = ABCMeta
        class C:
           __metaclass__ = type
           class D: pass
    """)
    a = astroid['A']
    b = a.locals['B'][0]
    c = astroid['C']
    d = c.locals['D'][0]
    assert a.metaclass().name == 'ABCMeta'
    assert not b.newstyle
    assert b.metaclass() is None
    assert c.metaclass().name == 'type'
    assert d.metaclass().name == 'ABCMeta'


@test_utils.require_version(maxver='3.0')
def test_parent_metaclass():
    astroid = builder.parse("""
        from abc import ABCMeta
        class Test:
            __metaclass__ = ABCMeta
        class SubTest(Test): pass
    """)
    klass = astroid['SubTest']
    assert klass.newstyle
    metaclass = klass.metaclass()
    assert isinstance(metaclass, scoped_nodes.ClassDef)
    assert metaclass.name == 'ABCMeta'


@test_utils.require_version(maxver='3.0')
def test_metaclass_ancestors():
    astroid = builder.parse("""
        from abc import ABCMeta

        class FirstMeta(object):
            __metaclass__ = ABCMeta

        class SecondMeta(object):
            __metaclass__ = type

        class Simple(object):
            pass

        class FirstImpl(FirstMeta): pass
        class SecondImpl(FirstImpl): pass
        class ThirdImpl(Simple, SecondMeta):
            pass
    """)
    classes = {
        'ABCMeta': ('FirstImpl', 'SecondImpl'),
        'type': ('ThirdImpl', )
    }
    for metaclass, names in classes.items():
        for name in names:
            impl = astroid[name]
            meta = impl.metaclass()
            assert isinstance(meta, nodes.ClassDef)
            assert meta.name == metaclass


def test_metaclass_type():
    klass = builder.extract_node("""
        def with_metaclass(meta, base=object):
            return meta("NewBase", (base, ), {})

        class ClassWithMeta(with_metaclass(type)): #@
            pass
    """)
    assert ['NewBase', 'object'] == \
        [base.name for base in klass.ancestors()]


def test_no_infinite_metaclass_loop():
    klass = builder.extract_node("""
        class SSS(object):

            class JJJ(object):
                pass

            @classmethod
            def Init(cls):
                cls.JJJ = type('JJJ', (cls.JJJ,), {})

        class AAA(SSS):
            pass

        class BBB(AAA.JJJ):
            pass
    """)
    assert not scoped_nodes._is_metaclass(klass)
    ancestors = [base.name for base in klass.ancestors()]
    assert 'object' in ancestors
    assert 'JJJ' in ancestors


def test_no_infinite_metaclass_loop_with_redefine():
    ast_nodes = builder.extract_node("""
        import datetime

        class A(datetime.date): #@
            @classmethod
            def now(cls):
                return cls()

        class B(datetime.date): #@
            pass

        datetime.date = A
        datetime.date = B
    """)
    for klass in ast_nodes:
        assert None == klass.metaclass()


def test_metaclass_generator_hack():
    klass = builder.extract_node("""
        import six

        class WithMeta(six.with_metaclass(type, object)): #@
            pass
    """)
    assert ['object'] == \
        [base.name for base in klass.ancestors()]
    assert 'type' == klass.metaclass().name


def test_using_six_add_metaclass():
    klass = builder.extract_node('''
    import six
    import abc

    @six.add_metaclass(abc.ABCMeta)
    class WithMeta(object):
        pass
    ''')
    inferred = next(klass.infer())
    metaclass = inferred.metaclass()
    assert isinstance(metaclass, scoped_nodes.ClassDef)
    assert metaclass.qname() == 'abc.ABCMeta'


def test_using_invalid_six_add_metaclass_call():
    klass = builder.extract_node('''
    import six
    @six.add_metaclass()
    class Invalid(object):
        pass
    ''')
    inferred = next(klass.infer())
    assert inferred.metaclass() is None


def test_nonregr_infer_callresult():
    astroid = builder.parse("""
        class Delegate(object):
            def __get__(self, obj, cls):
                return getattr(obj._subject, self.attribute)

        class CompositeBuilder(object):
            __call__ = Delegate()

        builder = CompositeBuilder(result, composite)
        tgts = builder()
    """)
    instance = astroid['tgts']
    # used to raise "'_Yes' object is not iterable", see
    # https://bitbucket.org/logilab/astroid/issue/17
    assert list(instance.infer()) == [util.Uninferable]


def test_slots():
    astroid = builder.parse("""
        from collections import deque
        from textwrap import dedent

        class First(object): #@
            __slots__ = ("a", "b", 1)
        class Second(object): #@
            __slots__ = "a"
        class Third(object): #@
            __slots__ = deque(["a", "b", "c"])
        class Fourth(object): #@
            __slots__ = {"a": "a", "b": "b"}
        class Fifth(object): #@
            __slots__ = list
        class Sixth(object): #@
            __slots__ = ""
        class Seventh(object): #@
            __slots__ = dedent.__name__
        class Eight(object): #@
            __slots__ = ("parens")
        class Ninth(object): #@
            pass
        class Ten(object): #@
            __slots__ = dict({"a": "b", "c": "d"})
    """)
    expected = [
        ('First', ('a', 'b')),
        ('Second', ('a', )),
        ('Third', None),
        ('Fourth', ('a', 'b')),
        ('Fifth', None),
        ('Sixth', None),
        ('Seventh', ('dedent', )),
        ('Eight', ('parens', )),
        ('Ninth', None),
        ('Ten', ('a', 'c')),
    ]
    for cls, expected_value in expected:
        slots = astroid[cls].slots()
        if expected_value is None:
            assert slots is None
        else:
            assert list(expected_value) == [node.value for node in slots]


@test_utils.require_version(maxver='3.0')
def test_slots_py2():
    module = builder.parse("""
    class UnicodeSlots(object):
        __slots__ = (u"a", u"b", "c")
    """)
    slots = module['UnicodeSlots'].slots()
    assert len(slots) == 3
    assert slots[0].value == "a"
    assert slots[1].value == "b"
    assert slots[2].value == "c"


@test_utils.require_version(maxver='3.0')
def test_slots_py2_not_implemented():
    module = builder.parse("""
    class OldStyle:
        __slots__ = ("a", "b")
    """)
    msg = "The concept of slots is undefined for old-style classes."
    with pytest.raises(NotImplementedError) as cm:
        module['OldStyle'].slots()
    assert str(cm.value) == msg


def test_slots_for_dict_keys():
    module = builder.parse('''
    class Issue(object):
      SlotDefaults = {'id': 0, 'id1':1}
      __slots__ = SlotDefaults.keys()
    ''')
    cls = module['Issue']
    slots = cls.slots()
    assert len(slots) == 2
    assert slots[0].value == 'id'
    assert slots[1].value == 'id1'


def test_slots_empty_list_of_slots():
    module = builder.parse("""
    class Klass(object):
        __slots__ = ()
    """)
    cls = module['Klass']
    assert cls.slots() == []


def test_slots_taken_from_parents():
    module = builder.parse('''
    class FirstParent(object):
        __slots__ = ('a', 'b', 'c')
    class SecondParent(FirstParent):
        __slots__ = ('d', 'e')
    class Third(SecondParent):
        __slots__ = ('d', )
    ''')
    cls = module['Third']
    slots = cls.slots()
    assert sorted(set(slot.value for slot in slots)) == \
        ['a', 'b', 'c', 'd', 'e']


def test_all_ancestors_need_slots():
    module = builder.parse('''
    class A(object):
        __slots__ = ('a', )
    class B(A): pass
    class C(B):
        __slots__ = ('a', )
    ''')
    cls = module['C']
    assert cls.slots() is None
    cls = module['B']
    assert cls.slots() is None


def assertEqualMro(klass, expected_mro):
    assert [member.name for member in klass.mro()] == expected_mro


@test_utils.require_version(maxver='3.0')
def test_no_mro_for_old_style():
    node = builder.extract_node("""
    class Old: pass""")
    with pytest.raises(NotImplementedError) as cm:
        node.mro()
    assert str(cm.value) == "Could not obtain mro for old-style classes."


@test_utils.require_version(maxver='3.0')
def test_combined_newstyle_oldstyle_in_mro():
    node = builder.extract_node('''
    class Old:
        pass
    class New(object):
        pass
    class New1(object):
        pass
    class New2(New, New1):
        pass
    class NewOld(New2, Old): #@
        pass
    ''')
    assertEqualMro(node, ['NewOld', 'New2', 'New', 'New1', 'object', 'Old'])
    assert node.newstyle


def test_with_metaclass_mro():
    astroid = builder.parse("""
    import six

    class C(object):
        pass
    class B(C):
        pass
    class A(six.with_metaclass(type, B)):
        pass
    """)
    assertEqualMro(astroid['A'], ['A', 'B', 'C', 'object'])


def test_mro():
    astroid = builder.parse("""
    class C(object): pass
    class D(dict, C): pass

    class A1(object): pass
    class B1(A1): pass
    class C1(A1): pass
    class D1(B1, C1): pass
    class E1(C1, B1): pass
    class F1(D1, E1): pass
    class G1(E1, D1): pass

    class Boat(object): pass
    class DayBoat(Boat): pass
    class WheelBoat(Boat): pass
    class EngineLess(DayBoat): pass
    class SmallMultihull(DayBoat): pass
    class PedalWheelBoat(EngineLess, WheelBoat): pass
    class SmallCatamaran(SmallMultihull): pass
    class Pedalo(PedalWheelBoat, SmallCatamaran): pass

    class OuterA(object):
        class Inner(object):
            pass
    class OuterB(OuterA):
        class Inner(OuterA.Inner):
            pass
    class OuterC(OuterA):
        class Inner(OuterA.Inner):
            pass
    class OuterD(OuterC):
        class Inner(OuterC.Inner, OuterB.Inner):
            pass
    class Duplicates(str, str): pass

    """)
    assertEqualMro(astroid['D'], ['D', 'dict', 'C', 'object'])
    assertEqualMro(astroid['D1'], ['D1', 'B1', 'C1', 'A1', 'object'])
    assertEqualMro(astroid['E1'], ['E1', 'C1', 'B1', 'A1', 'object'])
    with pytest.raises(InconsistentMroError) as cm:
        astroid['F1'].mro()
    A1 = astroid.getattr('A1')[0]
    B1 = astroid.getattr('B1')[0]
    C1 = astroid.getattr('C1')[0]
    object_ = builder.MANAGER.astroid_cache[BUILTINS].getattr('object')[0]
    assert cm.value.mros == [[B1, C1, A1, object_],
                             [C1, B1, A1, object_]]
    with pytest.raises(InconsistentMroError) as cm:
        astroid['G1'].mro()
    assert cm.value.mros == [[C1, B1, A1, object_],
                             [B1, C1, A1, object_]]
    assertEqualMro(
        astroid['PedalWheelBoat'],
        ["PedalWheelBoat", "EngineLess",
         "DayBoat", "WheelBoat", "Boat", "object"])

    assertEqualMro(
        astroid["SmallCatamaran"],
        ["SmallCatamaran", "SmallMultihull", "DayBoat", "Boat", "object"])

    assertEqualMro(
        astroid["Pedalo"],
        ["Pedalo", "PedalWheelBoat", "EngineLess", "SmallCatamaran",
         "SmallMultihull", "DayBoat", "WheelBoat", "Boat", "object"])

    assertEqualMro(
        astroid['OuterD']['Inner'],
        ['Inner', 'Inner', 'Inner', 'Inner', 'object'])

    with pytest.raises(DuplicateBasesError) as cm:
        astroid['Duplicates'].mro()
    Duplicates = astroid.getattr('Duplicates')[0]
    assert cm.value.cls == Duplicates
    assert isinstance(cm.value, MroError)
    assert isinstance(cm.value, ResolveError)


def test_generator_from_infer_call_result_parent():
    func = builder.extract_node("""
    import contextlib

    @contextlib.contextmanager
    def test(): #@
        yield
    """)
    result = next(func.infer_call_result(func))
    assert isinstance(result, Generator)
    assert result.parent == func


def test_type_three_arguments():
    classes = builder.extract_node("""
    type('A', (object, ), {"a": 1, "b": 2, missing: 3}) #@
    """)
    first = next(classes.infer())
    assert isinstance(first, nodes.ClassDef)
    assert first.name == "A"
    assert first.basenames == ["object"]
    assert isinstance(first["a"], nodes.Const)
    assert first["a"].value == 1
    assert isinstance(first["b"], nodes.Const)
    assert first["b"].value == 2
    with pytest.raises(AttributeInferenceError):
        first.getattr("missing")


def test_implicit_metaclass():
    cls = builder.extract_node("""
    class A(object):
        pass
    """)
    type_cls = scoped_nodes.builtin_lookup("type")[1][0]
    assert cls.implicit_metaclass() == type_cls


def test_implicit_metaclass_lookup():
    cls = builder.extract_node('''
    class A(object):
        pass
    ''')
    instance = cls.instantiate_class()
    func = cls.getattr('mro')
    assert len(func) == 1
    with pytest.raises(AttributeInferenceError):
        instance.getattr('mro')


def test_metaclass_lookup_using_same_class():
    # Check that we don't have recursive attribute access for metaclass
    cls = builder.extract_node('''
    class A(object): pass
    ''')
    assert len(cls.getattr('mro')) == 1


def test_metaclass_lookup_inferrence_errors():
    module = builder.parse('''
    import six

    class Metaclass(type):
        foo = lala

    @six.add_metaclass(Metaclass)
    class B(object): pass
    ''')
    cls = module['B']
    assert util.Uninferable == next(cls.igetattr('foo'))


def test_metaclass_lookup():
    module = builder.parse('''
    import six

    class Metaclass(type):
        foo = 42
        @classmethod
        def class_method(cls):
            pass
        def normal_method(cls):
            pass
        @property
        def meta_property(cls):
            return 42
        @staticmethod
        def static():
            pass

    @six.add_metaclass(Metaclass)
    class A(object):
        pass
    ''')
    acls = module['A']
    normal_attr = next(acls.igetattr('foo'))
    assert isinstance(normal_attr, nodes.Const)
    assert normal_attr.value == 42

    class_method = next(acls.igetattr('class_method'))
    assert isinstance(class_method, BoundMethod)
    assert class_method.bound == module['Metaclass']

    normal_method = next(acls.igetattr('normal_method'))
    assert isinstance(normal_method, BoundMethod)
    assert normal_method.bound == module['A']

    # Attribute access for properties:
    #   from the metaclass is a property object
    #   from the class that uses the metaclass, the value
    #   of the property
    property_meta = next(module['Metaclass'].igetattr('meta_property'))
    assert isinstance(property_meta, UnboundMethod)
    wrapping = scoped_nodes.get_wrapping_class(property_meta)
    assert wrapping == module['Metaclass']

    property_class = next(acls.igetattr('meta_property'))
    assert isinstance(property_class, nodes.Const)
    assert property_class.value == 42

    static = next(acls.igetattr('static'))
    assert isinstance(static, scoped_nodes.FunctionDef)


@test_utils.require_version(maxver='3.0')
def test_implicit_metaclass_is_none():
    cls = builder.extract_node("""
    class A: pass
    """)
    assert cls.implicit_metaclass() is None


def test_local_attr_invalid_mro():
    cls = builder.extract_node("""
    # A has an invalid MRO, local_attr should fallback
    # to using .ancestors.
    class A(object, object):
        test = 42
    class B(A): #@
        pass
    """)
    local = cls.local_attr('test')[0]
    inferred = next(local.infer())
    assert isinstance(inferred, nodes.Const)
    assert inferred.value == 42


def test_has_dynamic_getattr():
    module = builder.parse("""
    class Getattr(object):
        def __getattr__(self, attrname):
            pass

    class Getattribute(object):
        def __getattribute__(self, attrname):
            pass

    class ParentGetattr(Getattr):
        pass
    """)
    assert module['Getattr'].has_dynamic_getattr()
    assert module['Getattribute'].has_dynamic_getattr()
    assert module['ParentGetattr'].has_dynamic_getattr()

    # Test that objects analyzed through the live introspection
    # aren't considered to have dynamic getattr implemented.
    import datetime
    astroid_builder = builder.AstroidBuilder()
    module = astroid_builder.module_build(datetime)
    assert not module['timedelta'].has_dynamic_getattr()


def test_duplicate_bases_namedtuple():
    module = builder.parse("""
    import collections
    _A = collections.namedtuple('A', 'a')

    class A(_A): pass

    class B(A): pass
    """)
    with pytest.raises(DuplicateBasesError):
        module['B'].mro()


def test_instance_bound_method_lambdas():
    ast_nodes = builder.extract_node('''
    class Test(object): #@
        lam = lambda self: self
        not_method = lambda xargs: xargs
    Test() #@
    ''')
    cls = next(ast_nodes[0].infer())
    assert isinstance(next(cls.igetattr('lam')), scoped_nodes.Lambda)
    assert isinstance(next(cls.igetattr('not_method')), scoped_nodes.Lambda)

    instance = next(ast_nodes[1].infer())
    lam = next(instance.igetattr('lam'))
    assert isinstance(lam, BoundMethod)
    not_method = next(instance.igetattr('not_method'))
    assert isinstance(not_method, scoped_nodes.Lambda)


def test_class_extra_decorators_frame_is_not_class():
    ast_node = builder.extract_node('''
    def ala():
        def bala(): #@
            func = 42
    ''')
    assert ast_node.extra_decorators == []


def test_class_extra_decorators_only_callfunc_are_considered():
    ast_node = builder.extract_node('''
    class Ala(object):
         def func(self): #@
             pass
         func = 42
    ''')
    assert ast_node.extra_decorators == []


def test_class_extra_decorators_only_assignment_names_are_considered():
    ast_node = builder.extract_node('''
    class Ala(object):
         def func(self): #@
             pass
         def __init__(self):
             self.func = staticmethod(func)

    ''')
    assert ast_node.extra_decorators == []


def test_class_extra_decorators_only_same_name_considered():
    ast_node = builder.extract_node('''
    class Ala(object):
         def func(self): #@
            pass
         bala = staticmethod(func)
    ''')
    assert ast_node.extra_decorators == []
    assert ast_node.type == 'method'


def test_class_extra_decorators():
    static_method, clsmethod = builder.extract_node('''
    class Ala(object):
         def static(self): #@
             pass
         def class_method(self): #@
             pass
         class_method = classmethod(class_method)
         static = staticmethod(static)
    ''')
    assert len(clsmethod.extra_decorators) == 1
    assert clsmethod.type == 'classmethod'
    assert len(static_method.extra_decorators) == 1
    assert static_method.type == 'staticmethod'


def test_extra_decorators_only_class_level_assignments():
    node = builder.extract_node('''
    def _bind(arg):
        return arg.bind

    class A(object):
        @property
        def bind(self):
            return 42
        def irelevant(self):
            # This is important, because it used to trigger
            # a maximum recursion error.
            bind = _bind(self)
            return bind
    A() #@
    ''')
    inferred = next(node.infer())
    bind = next(inferred.igetattr('bind'))
    assert isinstance(bind, nodes.Const)
    assert bind.value == 42
    parent = bind.scope()
    assert len(parent.extra_decorators) == 0
