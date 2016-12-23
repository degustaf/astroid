# Copyright (c) 2006-2007, 2009-2014 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2013-2016 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2015 Florian Bruhin <me@the-compiler.org>
# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

"""tests for specific behaviour of astroid nodes
"""
import os
import sys
import textwrap
import unittest
import warnings

import six

import astroid
from astroid import bases
from astroid import builder
from astroid import context as contextmod
from astroid import exceptions
from astroid import node_classes
from astroid import nodes
from astroid import parse
from astroid import util
from astroid import test_utils
from astroid import transforms
from astroid.tests import resources
import pytest


abuilder = builder.AstroidBuilder()
BUILTINS = six.moves.builtins.__name__


class TestAsString(resources.SysPathSetup):

    def test_tuple_as_string(self):
        def build(string):
            return abuilder.string_build(string).body[0].value

        assert build('1,').as_string() == '(1, )'
        assert build('1, 2, 3').as_string() == '(1, 2, 3)'
        assert build('(1, )').as_string() == '(1, )'
        assert build('1, 2, 3').as_string() == '(1, 2, 3)'

    @test_utils.require_version(minver='3.0')
    def test_func_signature_issue_185(self):
        code = textwrap.dedent('''
        def test(a, b, c=42, *, x=42, **kwargs):
            print(a, b, c, args)
        ''')
        node = parse(code)
        assert node.as_string().strip() == code.strip()

    def test_as_string_for_list_containing_uninferable(self):
        node = builder.extract_node('''
        def foo():
            bar = [arg] * 1
        ''')
        binop = node.body[0].value
        inferred = next(binop.infer())
        assert inferred.as_string() == '[Uninferable]'
        assert binop.as_string() == '([arg]) * (1)'

    def test_frozenset_as_string(self):
        ast_nodes = builder.extract_node('''
        frozenset((1, 2, 3)) #@
        frozenset({1, 2, 3}) #@
        frozenset([1, 2, 3,]) #@

        frozenset(None) #@
        frozenset(1) #@
        ''')
        ast_nodes = [next(node.infer()) for node in ast_nodes]

        assert ast_nodes[0].as_string() == 'frozenset((1, 2, 3))'
        assert ast_nodes[1].as_string() == 'frozenset({1, 2, 3})'
        assert ast_nodes[2].as_string() == 'frozenset([1, 2, 3])'

        assert ast_nodes[3].as_string() != 'frozenset(None)'
        assert ast_nodes[4].as_string() != 'frozenset(1)'

    def test_varargs_kwargs_as_string(self):
        ast = abuilder.string_build('raise_string(*args, **kwargs)').body[0]
        assert ast.as_string() == 'raise_string(*args, **kwargs)'

    def test_module_as_string(self):
        """check as_string on a whole module prepared to be returned identically
        """
        module = resources.build_file('data/module.py', 'data.module')
        with open(resources.find('data/module.py'), 'r') as fobj:
            assert module.as_string() == fobj.read()

    def test_module2_as_string(self):
        """check as_string on a whole module prepared to be returned identically
        """
        module2 = resources.build_file('data/module2.py', 'data.module2')
        with open(resources.find('data/module2.py'), 'r') as fobj:
            assert module2.as_string() == fobj.read()

    def test_as_string(self):
        """check as_string for python syntax >= 2.7"""
        code = '''one_two = {1, 2}
b = {v: k for (k, v) in enumerate('string')}
cdd = {k for k in b}\n\n'''
        ast = abuilder.string_build(code)
        assert ast.as_string() == code

    @test_utils.require_version('3.0')
    def test_3k_as_string(self):
        """check as_string for python 3k syntax"""
        code = '''print()

def function(var):
    nonlocal counter
    try:
        hello
    except NameError as nexc:
        (*hell, o) = b'hello'
        raise AttributeError from nexc
\n'''
        ast = abuilder.string_build(code)
        assert ast.as_string() == code

    @test_utils.require_version('3.0')
    @pytest.mark.xfail
    def test_3k_annotations_and_metaclass(self):
        code_annotations = textwrap.dedent('''
        def function(var:int):
            nonlocal counter

        class Language(metaclass=Natural):
            """natural language"""
        ''')

        ast = abuilder.string_build(code_annotations)
        assert ast.as_string() == code_annotations

    def test_ellipsis(self):
        ast = abuilder.string_build('a[...]').body[0]
        assert ast.as_string() == 'a[...]'

    def test_slices(self):
        for code in ('a[0]', 'a[1:3]', 'a[:-1:step]', 'a[:,newaxis]',
                     'a[newaxis,:]', 'del L[::2]', 'del A[1]', 'del Br[:]'):
            ast = abuilder.string_build(code).body[0]
            assert ast.as_string() == code

    def test_slice_and_subscripts(self):
        code = """a[:1] = bord[2:]
a[:1] = bord[2:]
del bree[3:d]
bord[2:]
del av[d::f], a[df:]
a[:1] = bord[2:]
del SRC[::1,newaxis,1:]
tous[vals] = 1010
del thousand[key]
del a[::2], a[:-1:step]
del Fee.form[left:]
aout.vals = miles.of_stuff
del (ccok, (name.thing, foo.attrib.value)), Fee.form[left:]
if all[1] == bord[0:]:
    pass\n\n"""
        ast = abuilder.string_build(code)
        assert ast.as_string() == code


class _NodeTest(unittest.TestCase):
    """test transformation of If Node"""
    CODE = None

    @property
    def astroid(self):
        try:
            return self.__class__.__dict__['CODE_Astroid']
        except KeyError:
            module = builder.parse(self.CODE)
            self.__class__.CODE_Astroid = module
            return module


class TestIfNode(_NodeTest):
    """test transformation of If Node"""
    CODE = """
        if 0:
            print()

        if True:
            print()
        else:
            pass

        if "":
            print()
        elif []:
            raise

        if 1:
            print()
        elif True:
            print()
        elif func():
            pass
        else:
            raise
    """

    def test_if_elif_else_node(self):
        """test transformation for If node"""
        assert len(self.astroid.body) == 4
        for stmt in self.astroid.body:
            assert isinstance(stmt, nodes.If)
        assert not self.astroid.body[0].orelse  # simple If
        assert isinstance(self.astroid.body[1].orelse[0], nodes.Pass)  # If / else
        assert isinstance(self.astroid.body[2].orelse[0], nodes.If)  # If / elif
        assert isinstance(self.astroid.body[3].orelse[0].orelse[0], nodes.If)

    def test_block_range(self):
        # XXX ensure expected values
        assert self.astroid.block_range(1) == (0, 22)
        assert self.astroid.block_range(10) == (0, 22)  # XXX (10, 22) ?
        assert self.astroid.body[1].block_range(5) == (5, 6)
        assert self.astroid.body[1].block_range(6) == (6, 6)
        assert self.astroid.body[1].orelse[0].block_range(7) == (7, 8)
        assert self.astroid.body[1].orelse[0].block_range(8) == (8, 8)


class TestTryExceptNode(_NodeTest):
    CODE = """
        try:
            print ('pouet')
        except IOError:
            pass
        except UnicodeError:
            print()
        else:
            print()
    """

    def test_block_range(self):
        # XXX ensure expected values
        assert self.astroid.body[0].block_range(1) == (1, 8)
        assert self.astroid.body[0].block_range(2) == (2, 2)
        assert self.astroid.body[0].block_range(3) == (3, 8)
        assert self.astroid.body[0].block_range(4) == (4, 4)
        assert self.astroid.body[0].block_range(5) == (5, 5)
        assert self.astroid.body[0].block_range(6) == (6, 6)
        assert self.astroid.body[0].block_range(7) == (7, 7)
        assert self.astroid.body[0].block_range(8) == (8, 8)


class TestTryFinallyNode(_NodeTest):
    CODE = """
        try:
            print ('pouet')
        finally:
            print ('pouet')
    """

    def test_block_range(self):
        # XXX ensure expected values
        assert self.astroid.body[0].block_range(1) == (1, 4)
        assert self.astroid.body[0].block_range(2) == (2, 2)
        assert self.astroid.body[0].block_range(3) == (3, 4)
        assert self.astroid.body[0].block_range(4) == (4, 4)


class TestTryExceptFinallyNode(_NodeTest):
    CODE = """
        try:
            print('pouet')
        except Exception:
            print ('oops')
        finally:
            print ('pouet')
    """

    def test_block_range(self):
        # XXX ensure expected values
        assert self.astroid.body[0].block_range(1) == (1, 6)
        assert self.astroid.body[0].block_range(2) == (2, 2)
        assert self.astroid.body[0].block_range(3) == (3, 4)
        assert self.astroid.body[0].block_range(4) == (4, 4)
        assert self.astroid.body[0].block_range(5) == (5, 5)
        assert self.astroid.body[0].block_range(6) == (6, 6)


@pytest.mark.skipif(six.PY3, reason="Python 2 specific test.")
class TestTryExcept2xNode(_NodeTest):
    CODE = """
        try:
            hello
        except AttributeError, (retval, desc):
            pass
    """

    def test_tuple_attribute(self):
        handler = self.astroid.body[0].handlers[0]
        assert isinstance(handler.name, nodes.Tuple)


def test_import_self_resolve(module2):
    myos = next(module2.igetattr('myos'))
    assert isinstance(myos, nodes.Module), myos
    assert myos.name == 'os'
    assert myos.qname() == 'os'
    assert myos.pytype() == '%s.module' % BUILTINS


def test_from_self_resolve(module):
    namenode = next(module.igetattr('NameNode'))
    assert isinstance(namenode, nodes.ClassDef), namenode
    assert namenode.root().name == 'astroid.node_classes'
    assert namenode.qname() == 'astroid.node_classes.Name'
    assert namenode.pytype() == '%s.type' % BUILTINS


def test_from_self_resolve2(module2):
    abspath = next(module2.igetattr('abspath'))
    assert isinstance(abspath, nodes.FunctionDef), abspath
    assert abspath.root().name == 'os.path'
    assert abspath.qname() == 'os.path.abspath'
    assert abspath.pytype() == '%s.function' % BUILTINS


def test_real_name(module, module2):
    from_ = module['NameNode']
    assert from_.real_name('NameNode') == 'Name'
    imp_ = module['os']
    assert imp_.real_name('os') == 'os'
    with pytest.raises(exceptions.AttributeInferenceError):
        imp_.real_name('os.path')
    imp_ = module['NameNode']
    assert imp_.real_name('NameNode') == 'Name'
    with pytest.raises(exceptions.AttributeInferenceError):
        imp_.real_name('Name')
    imp_ = module2['YO']
    assert imp_.real_name('YO') == 'YO'
    with pytest.raises(exceptions.AttributeInferenceError):
        imp_.real_name('data')


def test_as_string(module):
    ast = module['modutils']
    assert ast.as_string() == "from astroid import modutils"
    ast = module['NameNode']
    assert ast.as_string() == "from astroid.node_classes import Name as NameNode"
    ast = module['os']
    assert ast.as_string() == "import os.path"
    code = """from . import here
from .. import door
from .store import bread
from ..cave import wine\n\n"""
    ast = abuilder.string_build(code)
    assert ast.as_string() == code


def test_bad_import_inference():
    # Explication of bug
    '''When we import PickleError from nonexistent, a call to the infer
    method of this From node will be made by unpack_infer.
    inference.infer_from will try to import this module, which will fail and
    raise a InferenceException (by mixins.do_import_module). The infer_name
    will catch this exception and yield and Uninferable instead.
    '''

    code = '''
        try:
            from pickle import PickleError
        except ImportError:
            from nonexistent import PickleError

        try:
            pass
        except PickleError:
            pass
    '''
    module = builder.parse(code)
    handler_type = module.body[1].handlers[0].type

    excs = list(node_classes.unpack_infer(handler_type))
    # The number of returned object can differ on Python 2
    # and Python 3. In one version, an additional item will
    # be returned, from the _pickle module, which is not
    # present in the other version.
    assert isinstance(excs[0], nodes.ClassDef)
    assert excs[0].name == 'PickleError'
    assert excs[-1] is util.Uninferable


def test_absolute_import():
    module = resources.build_file('data/absimport.py')
    ctx = contextmod.InferenceContext()
    # will fail if absolute import failed
    ctx.lookupname = 'message'
    next(module['message'].infer(ctx))
    ctx.lookupname = 'email'
    m = next(module['email'].infer(ctx))
    assert not m.file.startswith(os.path.join('data', 'email.py'))


@pytest.mark.skip(reason="Why is this failing in pytest")
def test_more_absolute_import():
    module = resources.build_file('data/module1abs/__init__.py', 'data.module1abs')
    print(module.locals)
    assert 'sys' in module.locals


def test_as_string2():
    ast = abuilder.string_build("a == 2").body[0]
    assert ast.as_string() == "a == 2"


@pytest.mark.parametrize('value', [None, True, 1, 1.0, 1.0j, 'a', u'a'])
def test_const_node(value):
    # pylint: disable=no-member; union type in const_factory, this shouldn't happen
    node = nodes.const_factory(value)
    assert isinstance(node._proxied, nodes.ClassDef)
    assert node._proxied.name == value.__class__.__name__
    assert node.value is value
    assert node._proxied.parent
    assert node._proxied.root().name == value.__class__.__module__


def test_assign_to_True():
    """test that True and False assignments don't crash"""
    code = """
        True = False
        def hello(False):
            pass
        del True
    """
    if sys.version_info >= (3, 0):
        with pytest.raises(exceptions.AstroidBuildingError):
            builder.parse(code)
    else:
        ast = builder.parse(code)
        assign_true = ast['True']
        assert isinstance(assign_true, nodes.AssignName)
        assert assign_true.name == "True"
        del_true = ast.body[2].targets[0]
        assert isinstance(del_true, nodes.DelName)
        assert del_true.name == "True"


def test_linenumbering():
    ast = builder.parse('''
        def func(a,
            b): pass
        x = lambda x: None
    ''')
    assert ast['func'].args.fromlineno == 2
    assert not ast['func'].args.is_statement
    xlambda = next(ast['x'].infer())
    assert xlambda.args.fromlineno == 4
    assert xlambda.args.tolineno == 4
    assert not xlambda.args.is_statement
    if sys.version_info < (3, 0):
        assert ast['func'].args.tolineno == 3
    else:
        pytest.skip('FIXME  http://bugs.python.org/issue10445 '
                    '(no line number on function args)')


def test_no_super_getattr():
    # This is a test for issue
    # https://bitbucket.org/logilab/astroid/issue/91, which tests
    # that UnboundMethod doesn't call super when doing .getattr.

    ast = builder.parse('''
    class A(object):
        def test(self):
            pass
    meth = A.test
    ''')
    node = next(ast['meth'].infer())
    with pytest.raises(exceptions.AttributeInferenceError):
        node.getattr('__missssing__')
    name = node.getattr('__name__')[0]
    assert isinstance(name, nodes.Const)
    assert name.value == 'test'


def test_is_property():
    ast = builder.parse('''
    import abc

    def cached_property():
        # Not a real decorator, but we don't care
        pass
    def reify():
        # Same as cached_property
        pass
    def lazy_property():
        pass
    def lazyproperty():
        pass
    def lazy(): pass
    class A(object):
        @property
        def builtin_property(self):
            return 42
        @abc.abstractproperty
        def abc_property(self):
            return 42
        @cached_property
        def cached_property(self): return 42
        @reify
        def reified(self): return 42
        @lazy_property
        def lazy_prop(self): return 42
        @lazyproperty
        def lazyprop(self): return 42
        def not_prop(self): pass
        @lazy
        def decorated_with_lazy(self): return 42

    cls = A()
    builtin_property = cls.builtin_property
    abc_property = cls.abc_property
    cached_p = cls.cached_property
    reified = cls.reified
    not_prop = cls.not_prop
    lazy_prop = cls.lazy_prop
    lazyprop = cls.lazyprop
    decorated_with_lazy = cls.decorated_with_lazy
    ''')
    for prop in ('builtin_property', 'abc_property', 'cached_p', 'reified',
                 'lazy_prop', 'lazyprop', 'decorated_with_lazy'):
        inferred = next(ast[prop].infer())
        assert isinstance(inferred, nodes.Const), prop
        assert inferred.value == 42, prop

    inferred = next(ast['not_prop'].infer())
    assert isinstance(inferred, bases.BoundMethod)


@pytest.fixture
def transformer():
    return transforms.TransformVisitor()


def parse_transform(transformer, code):
    module = parse(code, apply_transforms=False)
    return transformer.visit(module)


def test_aliases(transformer):
    def test_from(node):
        node.names = node.names + [('absolute_import', None)]
        return node

    def test_class(node):
        node.name = 'Bar'
        return node

    def test_function(node):
        node.name = 'another_test'
        return node

    def test_callfunc(node):
        if node.func.name == 'Foo':
            node.func.name = 'Bar'
            return node

    def test_assname(node):
        if node.name == 'foo':
            return nodes.AssignName('bar', node.lineno, node.col_offset,
                                    node.parent)

    def test_assattr(node):
        if node.attrname == 'a':
            node.attrname = 'b'
            return node

    def test_getattr(node):
        if node.attrname == 'a':
            node.attrname = 'b'
            return node

    def test_genexpr(node):
        if node.elt.value == 1:
            node.elt = nodes.Const(2, node.lineno, node.col_offset,
                                   node.parent)
            return node

    transformer.register_transform(nodes.From, test_from)
    transformer.register_transform(nodes.Class, test_class)
    transformer.register_transform(nodes.Function, test_function)
    transformer.register_transform(nodes.CallFunc, test_callfunc)
    transformer.register_transform(nodes.AssName, test_assname)
    transformer.register_transform(nodes.AssAttr, test_assattr)
    transformer.register_transform(nodes.Getattr, test_getattr)
    transformer.register_transform(nodes.GenExpr, test_genexpr)

    string = '''
    from __future__ import print_function

    class Foo: pass

    def test(a): return a

    foo = Foo()
    foo.a = test(42)
    foo.a
    (1 for _ in range(0, 42))
    '''

    module = parse_transform(transformer, string)

    assert len(module.body[0].names) == 2
    assert isinstance(module.body[0], nodes.ImportFrom)
    assert module.body[1].name == 'Bar'
    assert isinstance(module.body[1], nodes.ClassDef)
    assert module.body[2].name == 'another_test'
    assert isinstance(module.body[2], nodes.FunctionDef)
    assert module.body[3].targets[0].name == 'bar'
    assert isinstance(module.body[3].targets[0], nodes.AssignName)
    assert module.body[3].value.func.name == 'Bar'
    assert isinstance(module.body[3].value, nodes.Call)
    assert module.body[4].targets[0].attrname == 'b'
    assert isinstance(module.body[4].targets[0], nodes.AssignAttr)
    assert isinstance(module.body[5], nodes.Expr)
    assert module.body[5].value.attrname == 'b'
    assert isinstance(module.body[5].value, nodes.Attribute)
    assert module.body[6].value.elt.value == 2
    assert isinstance(module.body[6].value, nodes.GeneratorExp)


@pytest.mark.skipif(six.PY3, reason="Python 3 doesn't have Repr nodes.")
def test_repr(transformer):
    def test_backquote(node):
        node.value.name = 'bar'
        return node

    transformer.register_transform(nodes.Backquote, test_backquote)

    module = parse_transform(transformer, '`foo`')

    assert module.body[0].value.value.name == 'bar'
    assert isinstance(module.body[0].value, nodes.Repr)


def test_asstype_warnings():
    string = '''
    class C: pass
    c = C()
    with warnings.catch_warnings(record=True) as w:
        pass
    '''
    module = parse(string)
    filter_stmts_mixin = module.body[0]
    assign_type_mixin = module.body[1].targets[0]
    parent_assign_type_mixin = module.body[2]

    with warnings.catch_warnings(record=True) as w:
        with test_utils.enable_warning(PendingDeprecationWarning):
            filter_stmts_mixin.ass_type()
            assert isinstance(w[0].message, PendingDeprecationWarning)
    with warnings.catch_warnings(record=True) as w:
        with test_utils.enable_warning(PendingDeprecationWarning):
            assign_type_mixin.ass_type()
            assert isinstance(w[0].message, PendingDeprecationWarning)
    with warnings.catch_warnings(record=True) as w:
        with test_utils.enable_warning(PendingDeprecationWarning):
            parent_assign_type_mixin.ass_type()
            assert isinstance(w[0].message, PendingDeprecationWarning)


def test_isinstance_warnings():
    msg_format = ("%r is deprecated and slated for removal in astroid "
                  "2.0, use %r instead")
    for cls in (nodes.Discard, nodes.Backquote, nodes.AssName,
                nodes.AssAttr, nodes.Getattr, nodes.CallFunc, nodes.From):
        with warnings.catch_warnings(record=True) as w:
            with test_utils.enable_warning(PendingDeprecationWarning):
                isinstance(42, cls)
        assert isinstance(w[0].message, PendingDeprecationWarning)
        actual_msg = msg_format % (cls.__class__.__name__, cls.__wrapped__.__name__)
        assert str(w[0].message) == actual_msg


@test_utils.require_version('3.5')
def test_async_await_keywords():
    async_def, async_for, async_with, await_node = builder.extract_node('''
    async def func(): #@
        async for i in range(10): #@
            f = __(await i)
        async with test(): #@
            pass
    ''')
    assert isinstance(async_def, nodes.AsyncFunctionDef)
    assert isinstance(async_for, nodes.AsyncFor)
    assert isinstance(async_with, nodes.AsyncWith)
    assert isinstance(await_node, nodes.Await)
    assert isinstance(await_node.value, nodes.Name)


@test_utils.require_version('3.5')
def test_await_async_as_string():
    codes = [
        textwrap.dedent('''
        async def function():
            await 42
        '''),
        textwrap.dedent('''
        async def function():
            async with (42):
                pass
        '''),
        textwrap.dedent('''
        async def function():
            async for i in range(10):
                await 42
        '''),
    ]
    for code in codes:
        ast_node = parse(code)
        assert ast_node.as_string().strip() == code.strip()


@pytest.mark.parametrize('code', ['f[1]', '[]', '(1, )'])
def test_subscript_load(code):
    node = builder.extract_node(code)
    assert node.ctx is astroid.Load


@pytest.mark.parametrize('code', ['del f[1]', 'del []'])
def test_del(code):
    node = builder.extract_node(code)
    assert node.targets[0].ctx is astroid.Del


def test_subscript_store():
    node = builder.extract_node('f[1] = 2')
    subscript = node.targets[0]
    assert subscript.ctx is astroid.Store


@pytest.mark.parametrize('code', ['[0] = 2', '(1, ) = 3'])
def test_store(code):
    with pytest.raises(exceptions.AstroidSyntaxError):
        builder.extract_node(code)


@test_utils.require_version(minver='3.5')
def test_starred_load():
    node = builder.extract_node('a = *b')
    starred = node.value
    assert starred.ctx is astroid.Load


@test_utils.require_version(minver='3.0')
def test_starred_store():
    node = builder.extract_node('a, *b = 1, 2')
    starred = node.targets[0].elts[1]
    assert starred.ctx is astroid.Store
