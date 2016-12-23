# Copyright (c) 2013-2014 Google, Inc.
# Copyright (c) 2014-2016 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2015 Philip Lorenz <philip@bithub.de>
# Copyright (c) 2015 LOGILAB S.A. (Paris, FRANCE) <contact@logilab.fr>
# Copyright (c) 2015 raylu <lurayl@gmail.com>
# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

"""Tests for basic functionality in astroid.brain."""
import sys
import six

import pytest

from astroid import MANAGER
from astroid import bases
from astroid import builder
from astroid import nodes
from astroid import util
import astroid


@pytest.fixture(scope="session")
def has_multiprocessing():
    return pytest.importorskip("multiprocessing")


@pytest.fixture(scope="session")
def has_enum():
    if six.PY2:
        return pytest.importorskip("enum34")
    return pytest.importorskip("enum")


@pytest.fixture(scope="session")
def has_nose():
    return pytest.importorskip("nose")


@pytest.fixture(scope="session")
def has_dateutil():
    return pytest.importorskip("dateutil")


@pytest.fixture(scope="session")
def has_numpy():
    return pytest.importorskip("numpy")


def test_hashlib():
    """Tests that brain extensions for hashlib work."""
    hashlib_module = MANAGER.ast_from_module_name('hashlib')
    for class_name in ['md5', 'sha1']:
        class_obj = hashlib_module[class_name]
        assert 'update' in class_obj
        assert 'digest' in class_obj
        assert 'hexdigest' in class_obj
        assert 'block_size' in class_obj
        assert 'digest_size' in class_obj
        assert len(class_obj['__init__'].args.args) == 2
        assert len(class_obj['__init__'].args.defaults) == 1
        assert len(class_obj['update'].args.args) == 2
        assert len(class_obj['digest'].args.args) == 1
        assert len(class_obj['hexdigest'].args.args) == 1


def test_namedtuple_base():
    klass = builder.extract_node("""
    from collections import namedtuple

    class X(namedtuple("X", ["a", "b", "c"])):
       pass
    """)
    assert [anc.name for anc in klass.ancestors()] == ['X', 'tuple', 'object']
    for anc in klass.ancestors():
        assert not (anc.parent is None)


def test_namedtuple_inference():
    klass = builder.extract_node("""
    from collections import namedtuple

    name = "X"
    fields = ["a", "b", "c"]
    class X(namedtuple(name, fields)):
       pass
    """)
    base = next(base for base in klass.ancestors() if base.name == 'X')
    assert {"a", "b", "c"} == set(base.instance_attrs)


def test_namedtuple_inference_failure():
    klass = builder.extract_node("""
    from collections import namedtuple

    def foo(fields):
       return __(namedtuple("foo", fields))
    """)
    assert util.Uninferable is next(klass.infer())


def test_namedtuple_advanced_inference():
    # urlparse return an object of class ParseResult, which has a
    # namedtuple call and a mixin as base classes
    result = builder.extract_node("""
    import six

    result = __(six.moves.urllib.parse.urlparse('gopher://'))
    """)
    instance = next(result.infer())
    assert len(instance.getattr('scheme')) == 1
    assert len(instance.getattr('port')) == 1
    with pytest.raises(astroid.AttributeInferenceError):
        instance.getattr('foo')
    assert len(instance.getattr('geturl')) == 1
    assert instance.name == 'ParseResult'


def test_namedtuple_instance_attrs():
    result = builder.extract_node('''
    from collections import namedtuple
    namedtuple('a', 'a b c')(1, 2, 3) #@
    ''')
    inferred = next(result.infer())
    for name, attr in inferred.instance_attrs.items():
        assert attr[0].attrname == name


def test_namedtuple_uninferable_fields():
    node = builder.extract_node('''
    x = [A] * 2
    from collections import namedtuple
    l = namedtuple('a', x)
    l(1)
    ''')
    inferred = next(node.infer())
    assert util.Uninferable is inferred


def testExtensionModules():
    transformer = MANAGER._transform
    for extender, _ in transformer.transforms[nodes.Module]:
        n = nodes.Module('__main__', None)
        extender(n)


@pytest.mark.usefixtures("has_nose")
def test_nose_tools():
    methods = builder.extract_node("""
    from nose.tools import assert_equal
    from nose.tools import assert_equals
    from nose.tools import assert_true
    assert_equal = assert_equal #@
    assert_true = assert_true #@
    assert_equals = assert_equals #@
    """)
    assert_equal = next(methods[0].value.infer())
    assert_true = next(methods[1].value.infer())
    assert_equals = next(methods[2].value.infer())

    assert isinstance(assert_equal, astroid.BoundMethod)
    assert isinstance(assert_true, astroid.BoundMethod)
    assert isinstance(assert_equals, astroid.BoundMethod)
    assert assert_equal.qname() == 'unittest.case.TestCase.assertEqual'
    assert assert_true.qname() == 'unittest.case.TestCase.assertTrue'
    assert assert_equals.qname() == 'unittest.case.TestCase.assertEqual'


def test_attribute_access():
    ast_nodes = builder.extract_node('''
    import six
    six.moves.http_client #@
    six.moves.urllib_parse #@
    six.moves.urllib_error #@
    six.moves.urllib.request #@
    ''')
    http_client = next(ast_nodes[0].infer())
    assert isinstance(http_client, nodes.Module)
    assert http_client.name == 'http.client' if six.PY3 else 'httplib'

    urllib_parse = next(ast_nodes[1].infer())
    if six.PY3:
        assert isinstance(urllib_parse, nodes.Module)
        assert urllib_parse.name == 'urllib.parse'
    else:
        # On Python 2, this is a fake module, the same behaviour
        # being mimicked in brain's tip for six.moves.
        assert isinstance(urllib_parse, astroid.Instance)
    urljoin = next(urllib_parse.igetattr('urljoin'))
    urlencode = next(urllib_parse.igetattr('urlencode'))
    if six.PY2:
        # In reality it's a function, but our implementations
        # transforms it into a method.
        assert isinstance(urljoin, astroid.BoundMethod)
        assert urljoin.qname() == 'urlparse.urljoin'
        assert isinstance(urlencode, astroid.BoundMethod)
        assert urlencode.qname() == 'urllib.urlencode'
    else:
        assert isinstance(urljoin, nodes.FunctionDef)
        assert urljoin.qname() == 'urllib.parse.urljoin'
        assert isinstance(urlencode, nodes.FunctionDef)
        assert urlencode.qname() == 'urllib.parse.urlencode'

    urllib_error = next(ast_nodes[2].infer())
    if six.PY3:
        assert isinstance(urllib_error, nodes.Module)
        assert urllib_error.name == 'urllib.error'
    else:
        # On Python 2, this is a fake module, the same behaviour
        # being mimicked in brain's tip for six.moves.
        assert isinstance(urllib_error, astroid.Instance)
    urlerror = next(urllib_error.igetattr('URLError'))
    assert isinstance(urlerror, nodes.ClassDef)
    content_too_short = next(urllib_error.igetattr('ContentTooShortError'))
    assert isinstance(content_too_short, nodes.ClassDef)

    urllib_request = next(ast_nodes[3].infer())
    if six.PY3:
        assert isinstance(urllib_request, nodes.Module)
        assert urllib_request.name == 'urllib.request'
    else:
        assert isinstance(urllib_request, astroid.Instance)
    urlopen = next(urllib_request.igetattr('urlopen'))
    urlretrieve = next(urllib_request.igetattr('urlretrieve'))
    if six.PY2:
        # In reality it's a function, but our implementations
        # transforms it into a method.
        assert isinstance(urlopen, astroid.BoundMethod)
        assert urlopen.qname() == 'urllib2.urlopen'
        assert isinstance(urlretrieve, astroid.BoundMethod)
        assert urlretrieve.qname() == 'urllib.urlretrieve'
    else:
        assert isinstance(urlopen, nodes.FunctionDef)
        assert urlopen.qname() == 'urllib.request.urlopen'
        assert isinstance(urlretrieve, nodes.FunctionDef)
        assert urlretrieve.qname() == 'urllib.request.urlretrieve'


def test_from_imports():
    ast_node = builder.extract_node('''
    from six.moves import http_client
    http_client.HTTPSConnection #@
    ''')
    inferred = next(ast_node.infer())
    assert isinstance(inferred, nodes.ClassDef)
    if six.PY3:
        qname = 'http.client.HTTPSConnection'
    else:
        qname = 'httplib.HTTPSConnection'
    assert inferred.qname() == qname


@pytest.mark.usefixture("has_multiprocessing")
def test_multiprocessing_module_attributes():
    # Test that module attributes are working,
    # especially on Python 3.4+, where they are obtained
    # from a context.
    module = builder.extract_node("""
    import multiprocessing
    """)
    module = module.do_import_module('multiprocessing')
    cpu_count = next(module.igetattr('cpu_count'))
    if sys.version_info < (3, 4):
        assert isinstance(cpu_count, nodes.FunctionDef)
    else:
        assert isinstance(cpu_count, astroid.BoundMethod)


@pytest.mark.usefixture("has_multiprocessing")
def test_module_name():
    module = builder.extract_node("""
    import multiprocessing
    multiprocessing.SyncManager()
    """)
    inferred_sync_mgr = next(module.infer())
    module = inferred_sync_mgr.root()
    assert module.name == 'multiprocessing.managers'


@pytest.mark.usefixture("has_multiprocessing")
@pytest.fixture(scope="module")
def multiprocessing_module():
    return builder.parse("""
    import multiprocessing
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    joinable_queue = manager.JoinableQueue()
    event = manager.Event()
    rlock = manager.RLock()
    bounded_semaphore = manager.BoundedSemaphore()
    condition = manager.Condition()
    barrier = manager.Barrier()
    pool = manager.Pool()
    list = manager.list()
    dict = manager.dict()
    value = manager.Value()
    array = manager.Array()
    namespace = manager.Namespace()
    """)


@pytest.mark.parametrize("attr, name", [
    ('queue', "{}.Queue".format(six.moves.queue.__name__)),
    ('joinable_queue', "{}.Queue".format(six.moves.queue.__name__)),
    ('event', "threading.{}".format("Event" if six.PY3 else "_Event")),
    ('rlock', "threading._RLock"),
    ('bounded_semaphore', "threading.{}".format("BoundedSemaphore" if six.PY3
                                                else "_BoundedSemaphore")),
    ('pool', "multiprocessing.pool.Pool"),
    ('list', "{}.{}".format(bases.BUILTINS, 'list')),
    ('dict', "{}.{}".format(bases.BUILTINS, 'dict')),
    ('array', "array.array"),

])
def test_multiprocessing_manager(multiprocessing_module, attr, name):
    # Test that we have the proper attributes
    # for a multiprocessing.managers.SyncManager
    item = next(multiprocessing_module[attr].infer())
    assert item.qname() == name


@pytest.mark.usefixture("has_multiprocessing")
def test_multiprocessing_manager_old(multiprocessing_module):
    # Test that we have the proper attributes
    # for a multiprocessing.managers.SyncManager
    manager = next(multiprocessing_module['manager'].infer())
    # Verify that we have these attributes
    assert manager.getattr('start')
    assert manager.getattr('shutdown')


def test_threading():
    module = builder.extract_node("""
    import threading
    threading.Lock()
    """)
    inferred = next(module.infer())
    assert isinstance(inferred, astroid.Instance)
    assert inferred.root().name == 'threading'
    assert isinstance(inferred.getattr('acquire')[0], astroid.FunctionDef)
    assert isinstance(inferred.getattr('release')[0], astroid.FunctionDef)


@pytest.mark.usefixture("has_enum")
def test_simple_enum():
    module = builder.parse("""
    import enum

    class MyEnum(enum.Enum):
        one = "one"
        two = "two"

        def mymethod(self, x):
            return 5

    """)

    enumeration = next(module['MyEnum'].infer())
    one = enumeration['one']
    assert one.pytype() == '.MyEnum.one'

    property_type = '{}.property'.format(bases.BUILTINS)
    for propname in ('name', 'value'):
        prop = next(iter(one.getattr(propname)))
        assert property_type in prop.decoratornames()

    meth = one.getattr('mymethod')[0]
    assert isinstance(meth, astroid.FunctionDef)


@pytest.mark.usefixture("has_enum")
def test_looks_like_enum_false_positive():
    # Test that a class named Enumeration is not considered a builtin enum.
    module = builder.parse('''
    class Enumeration(object):
        def __init__(self, name, enum_list):
            pass
        test = 42
    ''')
    enumeration = module['Enumeration']
    test = next(enumeration.igetattr('test'))
    assert test.value == 42


@pytest.mark.usefixture("has_enum")
def test_enum_multiple_base_classes():
    module = builder.parse("""
    import enum

    class Mixin:
        pass

    class MyEnum(Mixin, enum.Enum):
        one = 1
    """)
    enumeration = next(module['MyEnum'].infer())
    one = enumeration['one']

    clazz = one.getattr('__class__')[0]
    assert clazz.is_subtype_of('.Mixin'), \
        'Enum instance should share base classes with generating class'


@pytest.mark.usefixture("has_enum")
def test_int_enum():
    module = builder.parse("""
    import enum

    class MyEnum(enum.IntEnum):
        one = 1
    """)

    enumeration = next(module['MyEnum'].infer())
    one = enumeration['one']

    clazz = one.getattr('__class__')[0]
    int_type = '{}.{}'.format(bases.BUILTINS, 'int')
    assert clazz.is_subtype_of(int_type), \
        'IntEnum based enums should be a subtype of int'


@pytest.mark.usefixture("has_enum")
def test_enum_func_form_is_class_not_instance():
    cls, instance = builder.extract_node('''
    from enum import Enum
    f = Enum('Audience', ['a', 'b', 'c'])
    f #@
    f(1) #@
    ''')
    inferred_cls = next(cls.infer())
    assert isinstance(inferred_cls, bases.Instance)
    inferred_instance = next(instance.infer())
    assert isinstance(inferred_instance, bases.Instance)
    assert isinstance(next(inferred_instance.igetattr('name')), nodes.Const)
    assert isinstance(next(inferred_instance.igetattr('value')), nodes.Const)


@pytest.mark.usefixture("has_dateutil")
def test_parser():
    module = builder.parse("""
    from dateutil.parser import parse
    d = parse('2000-01-01')
    """)
    d_type = next(module['d'].infer())
    assert d_type.qname() == "datetime.datetime"


@pytest.mark.usefixture("has_numpy")
def test_numpy():
    node = builder.extract_node('''
    import numpy
    numpy.ones #@
    ''')
    inferred = next(node.infer())
    assert isinstance(inferred, nodes.FunctionDef)


def test_pytest():
    ast_node = builder.extract_node('''
    import pytest
    pytest #@
    ''')
    module = next(ast_node.infer())
    attrs = ['deprecated_call', 'warns', 'exit', 'fail', 'skip',
             'importorskip', 'xfail', 'mark', 'raises', 'freeze_includes',
             'set_trace', 'fixture', 'yield_fixture']
    if pytest.__version__.split('.')[0] == '3':
        attrs += ['approx', 'register_assert_rewrite']

    for attr in attrs:
        assert attr in module


@pytest.mark.skipif(six.PY2, reason='Needs Python 3 io model')
@pytest.mark.parametrize("name", ['__stdout__', '__stderr__', '__stdin__'])
def test_sys_streams(name):
    node = astroid.extract_node('''
    import sys
    sys.{}
    '''.format(name))
    inferred = next(node.infer())
    buffer_attr = next(inferred.igetattr('buffer'))
    assert isinstance(buffer_attr, astroid.Instance)
    assert buffer_attr.name == 'BufferedWriter'
    raw = next(buffer_attr.igetattr('raw'))
    assert isinstance(raw, astroid.Instance)
    assert raw.name == 'FileIO'
