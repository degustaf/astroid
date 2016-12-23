# Copyright (c) 2006-2014 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2014-2016 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014-2015 Google, Inc.
# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

"""tests for the astroid builder and rebuilder module"""

import os
import sys
import unittest

import six

from astroid import builder
from astroid import exceptions
from astroid import manager
from astroid import nodes
from astroid import test_utils
from astroid import util
from astroid.tests import resources
import pytest


@pytest.fixture
def MANAGER():
    return manager.AstroidManager()


@pytest.fixture
def BUILTINS():
    return six.moves.builtins.__name__


@pytest.fixture
def builtin_ast(BUILTINS, MANAGER):
    return MANAGER.ast_from_module_name(BUILTINS)


@pytest.fixture
def astroid_body():
    return resources.build_file('data/format.py').body


@pytest.fixture
def _builder():
    return builder.AstroidBuilder()


@pytest.fixture
def guess_encoding():
    return builder._guess_encoding


def test_callfunc_lineno(astroid_body):
    # on line 4:
    #    function('aeozrijz\
    #    earzer', hop)
    discard = astroid_body[0]
    assert isinstance(discard, nodes.Expr)
    assert discard.fromlineno == 4
    assert discard.tolineno == 5
    callfunc = discard.value
    assert isinstance(callfunc, nodes.Call)
    assert callfunc.fromlineno == 4
    assert callfunc.tolineno == 5
    name = callfunc.func
    assert isinstance(name, nodes.Name)
    assert name.fromlineno == 4
    assert name.tolineno == 4
    strarg = callfunc.args[0]
    assert isinstance(strarg, nodes.Const)
    if hasattr(sys, 'pypy_version_info'):
        lineno = 4
    else:
        lineno = 5  # no way for this one in CPython (is 4 actually)
    assert strarg.fromlineno == lineno
    assert strarg.tolineno == lineno
    namearg = callfunc.args[1]
    assert isinstance(namearg, nodes.Name)
    assert namearg.fromlineno == 5
    assert namearg.tolineno == 5


def test_callfunc_lineno2(astroid_body):
    # on line 10:
    #    fonction(1,
    #             2,
    #             3,
    #             4)
    discard = astroid_body[2]
    assert isinstance(discard, nodes.Expr)
    assert discard.fromlineno == 10
    assert discard.tolineno == 13
    callfunc = discard.value
    assert isinstance(callfunc, nodes.Call)
    assert callfunc.fromlineno == 10
    assert callfunc.tolineno == 13
    name = callfunc.func
    assert isinstance(name, nodes.Name)
    assert name.fromlineno == 10
    assert name.tolineno == 10
    for i, arg in enumerate(callfunc.args):
        assert isinstance(arg, nodes.Const)
        assert arg.fromlineno == 10+i
        assert arg.tolineno == 10+i


def test_function_lineno(astroid_body):
    # on line 15:
    #    def definition(a,
    #                   b,
    #                   c):
    #        return a + b + c
    function = astroid_body[3]
    assert isinstance(function, nodes.FunctionDef)
    assert function.fromlineno == 15
    assert function.tolineno == 18
    return_ = function.body[0]
    assert isinstance(return_, nodes.Return)
    assert return_.fromlineno == 18
    assert return_.tolineno == 18
    if sys.version_info < (3, 0):
        assert function.blockstart_tolineno == 17
    else:
        pytest.skip('FIXME  http://bugs.python.org/issue10445 '
                    '(no line number on function args)')


def test_decorated_function_lineno():
    astroid = builder.parse('''
        @decorator
        def function(
            arg):
            print (arg)
        ''', __name__)
    function = astroid['function']
    # XXX discussable, but that's what is expected by pylint right now
    assert function.fromlineno == 3
    assert function.tolineno == 5
    assert function.decorators.fromlineno == 2
    assert function.decorators.tolineno == 2
    if sys.version_info < (3, 0):
        assert function.blockstart_tolineno == 4
    else:
        pytest.skip('FIXME  http://bugs.python.org/issue10445 '
                    '(no line number on function args)')


def test_class_lineno(astroid_body):
    # on line 20:
    #    class debile(dict,
    #                 object):
    #       pass
    class_ = astroid_body[4]
    assert isinstance(class_, nodes.ClassDef)
    assert class_.fromlineno == 20
    assert class_.tolineno == 22
    assert class_.blockstart_tolineno == 21
    pass_ = class_.body[0]
    assert isinstance(pass_, nodes.Pass)
    assert pass_.fromlineno == 22
    assert pass_.tolineno == 22


def test_if_lineno(astroid_body):
    # on line 20:
    #    if aaaa: pass
    #    else:
    #        aaaa,bbbb = 1,2
    #        aaaa,bbbb = bbbb,aaaa
    if_ = astroid_body[5]
    assert isinstance(if_, nodes.If)
    assert if_.fromlineno == 24
    assert if_.tolineno == 27
    assert if_.blockstart_tolineno == 24
    assert if_.orelse[0].fromlineno == 26
    assert if_.orelse[1].tolineno == 27


@pytest.mark.parametrize("code", [
    ('''
     for a in range(4):
       print (a)
       break
     else:
       print ("bouh")
     '''),
    ('''
     while a:
       print (a)
       break
     else:
       print ("bouh")
     ''')
])
def test_for_while_lineno(code):
    astroid = builder.parse(code, __name__)
    stmt = astroid.body[0]
    assert stmt.fromlineno == 2
    assert stmt.tolineno == 6
    assert stmt.blockstart_tolineno == 2
    assert stmt.orelse[0].fromlineno == 6  # XXX
    assert stmt.orelse[0].tolineno == 6


def test_try_except_lineno():
    astroid = builder.parse('''
        try:
          print (a)
        except:
          pass
        else:
          print ("bouh")
        ''', __name__)
    try_ = astroid.body[0]
    assert try_.fromlineno == 2
    assert try_.tolineno == 7
    assert try_.blockstart_tolineno == 2
    assert try_.orelse[0].fromlineno == 7  # XXX
    assert try_.orelse[0].tolineno == 7
    hdlr = try_.handlers[0]
    assert hdlr.fromlineno == 4
    assert hdlr.tolineno == 5
    assert hdlr.blockstart_tolineno == 4


@pytest.mark.parametrize('code,start,end', [
    ('''
     try:
       print (a)
     finally:
       print ("bouh")
     ''', 2, 5),
    ('''
     try:
       print (a)
     except:
       pass
     finally:
       print ("bouh")
     ''', 2, 7)
])
def test_try_finally_lineno(code, start, end):
    astroid = builder.parse(code, __name__)
    try_ = astroid.body[0]
    assert try_.fromlineno == start
    assert try_.tolineno == end
    assert try_.blockstart_tolineno == start
    assert try_.finalbody[0].fromlineno == end  # XXX
    assert try_.finalbody[0].tolineno == end


def test_with_lineno():
    astroid = builder.parse('''
        from __future__ import with_statement
        with file("/tmp/pouet") as f:
            print (f)
        ''', __name__)
    with_ = astroid.body[1]
    assert with_.fromlineno == 3
    assert with_.tolineno == 4
    assert with_.blockstart_tolineno == 3


@pytest.mark.parametrize('string', ['\x00', '"\\x1"'])
def test_data_build_null_bytes(_builder, string):
    with pytest.raises(exceptions.AstroidSyntaxError):
        _builder.string_build(string)


def test_missing_newline():
    """check that a file with no trailing new line is parseable"""
    resources.build_file('data/noendingnewline.py')


def test_missing_file():
    with pytest.raises(exceptions.AstroidBuildingError):
        resources.build_file('data/inexistant.py')


def test_inspect_build0(builtin_ast, BUILTINS):
    """test astroid tree build from a living object"""
    if six.PY2:
        fclass = builtin_ast['file']
        assert 'name' in fclass
        assert 'mode' in fclass
        assert 'read' in fclass
        assert fclass.newstyle
        assert fclass.pytype(), '%s.type' % BUILTINS
        assert isinstance(fclass['read'], nodes.FunctionDef)
        # check builtin function has args.args == None
        dclass = builtin_ast['dict']
        assert dclass['has_key'].args.args is None
    # just check type and object are there
    builtin_ast.getattr('type')
    objectastroid = builtin_ast.getattr('object')[0]
    assert isinstance(objectastroid.getattr('__new__')[0], nodes.FunctionDef)
    # check open file alias
    builtin_ast.getattr('open')
    # check 'help' is there (defined dynamically by site.py)
    builtin_ast.getattr('help')
    # check property has __init__
    pclass = builtin_ast['property']
    assert '__init__' in pclass
    assert isinstance(builtin_ast['None'], nodes.Const)
    assert isinstance(builtin_ast['True'], nodes.Const)
    assert isinstance(builtin_ast['False'], nodes.Const)
    if six.PY3:
        assert isinstance(builtin_ast['Exception'], nodes.ClassDef)
        assert isinstance(builtin_ast['NotImplementedError'], nodes.ClassDef)
    else:
        assert isinstance(builtin_ast['Exception'], nodes.ImportFrom)
        assert isinstance(builtin_ast['NotImplementedError'], nodes.ImportFrom)


@pytest.mark.xfail(os.name == 'java', reason="Fails for Jython")
def test_inspect_build1(MANAGER):
    time_ast = MANAGER.ast_from_module_name('time')
    assert time_ast
    assert time_ast['time'].args.defaults == []


def test_inspect_build2(_builder):
    """test astroid tree build from a living object"""
    mx = pytest.importorskip('mx')
    dt_ast = _builder.inspect_build(mx.DateTime)
    dt_ast.getattr('DateTime')
    # this one is failing since DateTimeType.__module__ = 'builtins' !
    # dt_ast.getattr('DateTimeType')


def test_inspect_build3(_builder):
    _builder.inspect_build(unittest)


@test_utils.require_version(maxver='3.0')
def test_inspect_build_instance(_builder):
    """test astroid tree build from a living object"""
    import exceptions as builtin_exceptions
    builtin_ast = _builder.inspect_build(builtin_exceptions)
    fclass = builtin_ast['OSError']
    # things like OSError.strerror are now (2.5) data descriptors on the
    # class instead of entries in the __dict__ of an instance
    container = fclass
    assert 'errno' in container
    assert 'strerror' in container
    assert 'filename' in container


@pytest.mark.parametrize('attr', ['object', 'type'])
def test_inspect_build_type_object(builtin_ast, attr):
    inferred = list(builtin_ast.igetattr(attr))
    assert len(inferred) == 1
    inferred = inferred[0]
    assert inferred.name == attr
    inferred.as_string()  # no crash test


def test_inspect_transform_module(MANAGER):
    # ensure no cached version of the time module
    MANAGER._mod_file_cache.pop(('time', None), None)
    MANAGER.astroid_cache.pop('time', None)

    def transform_time(node):
        if node.name == 'time':
            node.transformed = True

    MANAGER.register_transform(nodes.Module, transform_time)
    try:
        time_ast = MANAGER.ast_from_module_name('time')
        assert getattr(time_ast, 'transformed', False)
    finally:
        MANAGER.unregister_transform(nodes.Module, transform_time)


@pytest.mark.parametrize('pkg', ['data', 'data.__init__'])
def test_package_name(pkg):
    """test base properties and method of a astroid module"""
    datap = resources.build_file('data/__init__.py', pkg)
    assert datap.name == 'data'
    assert datap.package == 1


def test_yield_parent():
    """check if we added discard nodes as yield parent (w/ compiler)"""
    code = """
        def yiell(): #@
            yield 0
            if noe:
                yield more
    """
    func = builder.extract_node(code)
    assert isinstance(func, nodes.FunctionDef)
    stmt = func.body[0]
    assert isinstance(stmt, nodes.Expr)
    assert isinstance(stmt.value, nodes.Yield)
    assert isinstance(func.body[1].body[0], nodes.Expr)
    assert isinstance(func.body[1].body[0].value, nodes.Yield)


def test_object(_builder):
    obj_ast = _builder.inspect_build(object)
    assert '__setattr__' in obj_ast


def test_newstyle_detection():
    data = '''
        class A:
            "old style"

        class B(A):
            "old style"

        class C(object):
            "new style"

        class D(C):
            "new style"

        __metaclass__ = type

        class E(A):
            "old style"

        class F:
            "new style"
    '''
    mod_ast = builder.parse(data, __name__)
    if six.PY3:
        assert mod_ast['A'].newstyle
        assert mod_ast['B'].newstyle
        assert mod_ast['E'].newstyle
    else:
        assert not mod_ast['A'].newstyle
        assert not mod_ast['B'].newstyle
        assert not mod_ast['E'].newstyle
    assert mod_ast['C'].newstyle
    assert mod_ast['D'].newstyle
    assert mod_ast['F'].newstyle


def test_globals():
    data = '''
        CSTE = 1

        def update_global():
            global CSTE
            CSTE += 1

        def global_no_effect():
            global CSTE2
            print (CSTE)
    '''
    astroid = builder.parse(data, __name__)
    assert len(astroid.getattr('CSTE')) == 2
    assert isinstance(astroid.getattr('CSTE')[0], nodes.AssignName)
    assert astroid.getattr('CSTE')[0].fromlineno == 2
    assert astroid.getattr('CSTE')[1].fromlineno == 6
    with pytest.raises(exceptions.AttributeInferenceError):
        astroid.getattr('CSTE2')
    with pytest.raises(exceptions.InferenceError):
        next(astroid['global_no_effect'].ilookup('CSTE2'))


@pytest.mark.skipif(os.name == 'java',
                    reason='This test is skipped on Jython, because the '
                    'socket object is patched later on with the '
                    'methods we are looking for. Since we do not '
                    'understand setattr in for loops yet, we skip this')
def test_socket_build(_builder):
    import socket
    astroid = _builder.module_build(socket)
    # XXX just check the first one. Actually 3 objects are inferred (look at
    # the socket module) but the last one as those attributes dynamically
    # set and astroid is missing this.
    for fclass in astroid.igetattr('socket'):
        assert 'connect' in fclass
        assert 'send' in fclass
        assert 'close' in fclass
        break


def test_gen_expr_var_scope():
    data = 'l = list(n for n in range(10))\n'
    astroid = builder.parse(data, __name__)
    # n unavailable outside gen expr scope
    assert 'n' not in astroid
    # test n is inferable anyway
    n = test_utils.get_name_node(astroid, 'n')
    assert n.scope() is not astroid
    assert [i.__class__ for i in n.infer()] == [util.Uninferable.__class__]


@pytest.mark.parametrize('code,expected', [
    ("import sys", set()),
    ("from __future__ import print_function", set(['print_function'])),
    ("""
        from __future__ import print_function
        from __future__ import absolute_import
        """, set(['print_function', 'absolute_import'])),

])
def test_future_imports(code, expected):
    mod = builder.parse(code)
    assert mod.future_imports == expected


def test_inferred_build():
    code = '''
        class A: pass
        A.type = "class"

        def A_assign_type(self):
            print (self)
        A.assign_type = A_assign_type
        '''
    astroid = builder.parse(code)
    lclass = list(astroid.igetattr('A'))
    assert len(lclass) == 1
    lclass = lclass[0]
    assert 'assign_type' in lclass.locals
    assert 'type' in lclass.locals


def test_augassign_attr():
    builder.parse("""
        class Counter:
            v = 0
            def inc(self):
                self.v += 1
        """, __name__)
    # TODO: Check self.v += 1 generate AugAssign(AssAttr(...)),
    # not AugAssign(GetAttr(AssName...))


@pytest.mark.parametrize('string', [None, {}])
def test_inferred_dont_pollute(string):
    code = '''
        def func(a=None):
            a.custom_attr = 0
        def func2(a={}):
            a.custom_attr = 0
        '''
    builder.parse(code)
    nonetype = nodes.const_factory(string)
    # pylint: disable=no-member; union type in const_factory, this shouldn't happen
    assert 'custom_attr' not in nonetype.locals
    assert 'custom_attr' not in nonetype.instance_attrs


def test_asstuple():
    code = 'a, b = range(2)'
    astroid = builder.parse(code)
    assert 'b' in astroid.locals


def test_asstuple2():
    code = '''
        def visit_if(self, node):
            node.test, body = node.tests[0]
        '''
    astroid = builder.parse(code)
    assert 'body' in astroid['visit_if'].locals


def test_build_constants():
    '''test expected values of constants after rebuilding'''
    code = '''
        def func():
            return None
            return
            return 'None'
        '''
    astroid = builder.parse(code)
    none, nothing, chain = [ret.value for ret in astroid.body[0].body]
    assert isinstance(none, nodes.Const)
    assert none.value is None
    assert nothing is None
    assert isinstance(chain, nodes.Const)
    assert chain.value == 'None'


def test_not_implemented():
    node = builder.extract_node('''
    NotImplemented #@
    ''')
    inferred = next(node.infer())
    assert isinstance(inferred, nodes.Const)
    assert inferred.value == NotImplemented


def test_module_base_props(module):
    """test base properties and method of a astroid module"""
    assert module.name == 'data.module'
    assert module.doc == "test module for astroid\n"
    assert module.fromlineno == 0
    assert module.parent is None
    assert module.frame() == module
    assert module.root() == module
    assert module.file == os.path.abspath(resources.find('data/module.py'))
    assert module.pure_python == 1
    assert module.package == 0
    assert not module.is_statement
    assert module.statement() == module


def test_module_locals(module):
    """test the 'locals' dictionary of a astroid module"""
    _locals = module.locals
    assert _locals is module.globals
    keys = sorted(_locals.keys())
    should = ['MY_DICT', 'NameNode', 'YO', 'YOUPI',
              '__revision__', 'global_access', 'modutils', 'four_args',
              'os', 'redirect']
    should.sort()
    assert keys == should


def test_function_base_props(module):
    """test base properties and method of a astroid function"""
    function = module['global_access']
    assert function.name == 'global_access'
    assert function.doc == 'function test'
    assert function.fromlineno == 11
    assert function.parent
    assert function.frame() == function
    assert function.parent.frame() == module
    assert function.root() == module
    assert [n.name for n in function.args.args] == ['key', 'val']
    assert function.type == 'function'


def test_function_locals(module):
    """test the 'locals' dictionary of a astroid function"""
    _locals = module['global_access'].locals
    assert len(_locals) == 4
    keys = sorted(_locals.keys())
    assert keys == ['i', 'key', 'local', 'val']


def test_class_base_props(module, YO_cls):
    """test base properties and method of a astroid class"""
    klass = YO_cls
    assert klass.name == 'YO'
    assert klass.doc == 'hehe'
    assert klass.fromlineno == 25
    assert klass.parent
    assert klass.frame() == klass
    assert klass.parent.frame() == module
    assert klass.root() == module
    assert klass.basenames == []
    if six.PY3:
        assert klass.newstyle
    else:
        assert not klass.newstyle


@pytest.mark.parametrize("cls,expected", [
    ('YO', ['__init__', 'a']),
    ('YOUPI', ['__init__', 'class_attr', 'class_method', 'method', 'static_method']),
])
def test_class_locals(module, cls, expected):
    """test the 'locals' dictionary of a astroid class"""
    klass = module[cls]
    locals_ = klass.locals
    keys = sorted(locals_.keys())
    assert keys == expected


@pytest.mark.parametrize('cls,expected', [('YO', ['yo']), ('YOUPI', ['member'])])
def test_class_instance_attrs(module, cls, expected):
    klass = module[cls]
    assert list(klass.instance_attrs.keys()) == expected


@pytest.mark.parametrize('cls,expected', [('YO', []), ('YOUPI', ['YO'])])
def test_class_basenames(module, cls, expected):
    klass = module[cls]
    assert klass.basenames == expected


@pytest.mark.parametrize('name,expected,expected_type', [
    ('method', ['self'], 'method'),  # "normal" method
    ('class_method', ['cls'], 'classmethod'),  # class method
    ('static_method', [], 'staticmethod'),  # static method
])
def test_method_base_props(YOUPI_cls, name, expected, expected_type):
    """test base properties and method of a astroid method"""
    method = YOUPI_cls[name]
    assert [n.name for n in method.args.args] == expected
    assert method.type == expected_type


def test_method_base_props2(YOUPI_cls):
    method = YOUPI_cls['method']
    assert method.name == 'method'
    assert method.doc == 'method test'
    assert method.fromlineno == 47


@pytest.mark.parametrize('expected', [
    pytest.mark.skipif('sys.version_info <  (3, 0)')(['autre', 'local', 'self']),
    pytest.mark.skipif('sys.version_info >= (3, 0)')(['a', 'autre', 'b', 'local', 'self']),
])
def test_method_locals(YOUPI_cls, expected):
    """test the 'locals' dictionary of a astroid method"""
    method = YOUPI_cls['method']
    _locals = sorted(method.locals)
    assert len(_locals) == len(expected)
    assert _locals == expected


def test_unknown_encoding():
    with pytest.raises(exceptions.AstroidSyntaxError):
        resources.build_file('data/invalid_encoding.py')


@pytest.mark.skipif(six.PY3, reason="guess_encoding not used on Python 3")
@pytest.mark.parametrize("code,encoding", [
    ('# -*- coding: UTF-8  -*-', 'UTF-8'),
    ('# -*- coding:UTF-8 -*-', 'UTF-8'),
    ('''
    ### -*- coding: ISO-8859-1  -*-
    ''', 'ISO-8859-1'),
    ('''

    ### -*- coding: ISO-8859-1  -*-
    ''', None),

])
def testEmacs(guess_encoding, code, encoding):
    e = guess_encoding(code)
    assert e == encoding


@pytest.mark.skipif(six.PY3, reason="guess_encoding not used on Python 3")
@pytest.mark.parametrize("code,encoding", [
    ('# vim:fileencoding=UTF-8', 'UTF-8'),
    ('''
    ### vim:fileencoding=ISO-8859-1
    ''', 'ISO-8859-1'),
    ('''

    ### vim:fileencoding= ISO-8859-1
    ''', None),

])
def testVim(guess_encoding, code, encoding):
    e = guess_encoding(code)
    assert e == encoding


@pytest.mark.skipif(six.PY3, reason="guess_encoding not used on Python 3")
@pytest.mark.parametrize("code", [
    "coding = UTF-8",  # setting "coding" variable
    "coding:UTF-8",  # setting a dictionary entry
    "def do_something(a_word_with_coding=None):",  # setting an argument
])
def test_wrong_coding(guess_encoding, code):
    e = guess_encoding(code)
    assert e is None


@pytest.mark.skipif(six.PY3, reason="guess_encoding not used on Python 3")
@pytest.mark.parametrize("code,encoding", [
    ('\xef\xbb\xbf any UTF-8 data', 'UTF-8'),
    (' any UTF-8 data \xef\xbb\xbf', None),
])
def testUTF8(guess_encoding, code, encoding):
    e = guess_encoding(code)
    assert e == encoding
