# Copyright (c) 2013-2016 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2014 Google, Inc.
# Copyright (c) 2015-2016 Cara Vinson <ceridwenv@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

from textwrap import dedent
import pytest

from astroid import nodes
from astroid.node_classes import Assign, Expr, YieldFrom, Name, Const
from astroid.builder import AstroidBuilder, extract_node
from astroid.scoped_nodes import ClassDef, FunctionDef
from astroid.test_utils import require_version


@pytest.fixture(scope='module')
def builder():
    return AstroidBuilder()


@require_version('3.0')
def test_starred_notation(builder):
    astroid = builder.string_build("*a, b = [1, 2, 3]", 'test', 'test')

    # Get the star node
    node = next(next(next(astroid.get_children()).get_children()).get_children())

    assert isinstance(node.assign_type(), Assign)


@require_version('3.3')
def test_yield_from(builder):
    body = dedent("""
    def func():
        yield from iter([1, 2])
    """)
    astroid = builder.string_build(body)
    func = astroid.body[0]
    assert isinstance(func, FunctionDef)
    yieldfrom_stmt = func.body[0]

    assert isinstance(yieldfrom_stmt, Expr)
    assert isinstance(yieldfrom_stmt.value, YieldFrom)
    assert yieldfrom_stmt.as_string() == 'yield from iter([1, 2])'


@require_version('3.3')
def test_yield_from_is_generator(builder):
    body = dedent("""
    def func():
        yield from iter([1, 2])
    """)
    astroid = builder.string_build(body)
    func = astroid.body[0]
    assert isinstance(func, FunctionDef)
    assert func.is_generator()


@require_version('3.3')
def test_yield_from_as_string(builder):
    body = dedent("""
    def func():
        yield from iter([1, 2])
        value = yield from other()
    """)
    astroid = builder.string_build(body)
    func = astroid.body[0]
    assert func.as_string().strip() == body.strip()

# metaclass tests


@require_version('3.0')
def test_simple_metaclass(builder):
    astroid = builder.string_build("class Test(metaclass=type): pass")
    klass = astroid.body[0]

    metaclass = klass.metaclass()
    assert isinstance(metaclass, ClassDef)
    assert metaclass.name == 'type'


@require_version('3.0')
def test_metaclass_error(builder):
    astroid = builder.string_build("class Test(metaclass=typ): pass")
    klass = astroid.body[0]
    assert not klass.metaclass()


@require_version('3.0')
def test_metaclass_imported(builder):
    astroid = builder.string_build(dedent("""
    from abc import ABCMeta
    class Test(metaclass=ABCMeta): pass"""))
    klass = astroid.body[1]

    metaclass = klass.metaclass()
    assert isinstance(metaclass, ClassDef)
    assert metaclass.name == 'ABCMeta'


@require_version('3.0')
def test_as_string(builder):
    body = dedent("""
    from abc import ABCMeta
    class Test(metaclass=ABCMeta): pass""")
    astroid = builder.string_build(body)
    klass = astroid.body[1]

    assert klass.as_string() == '\n\nclass Test(metaclass=ABCMeta):\n    pass\n'


@require_version('3.0')
def test_old_syntax_works(builder):
    astroid = builder.string_build(dedent("""
    class Test:
        __metaclass__ = type
    class SubTest(Test): pass
    """))
    klass = astroid['SubTest']
    metaclass = klass.metaclass()
    assert metaclass is None


@require_version('3.0')
def test_metaclass_yes_leak(builder):
    astroid = builder.string_build(dedent("""
    # notice `ab` instead of `abc`
    from ab import ABCMeta

    class Meta(metaclass=ABCMeta): pass
    """))
    klass = astroid['Meta']
    assert klass.metaclass() is None


@require_version('3.0')
def test_parent_metaclass(builder):
    astroid = builder.string_build(dedent("""
    from abc import ABCMeta
    class Test(metaclass=ABCMeta): pass
    class SubTest(Test): pass
    """))
    klass = astroid['SubTest']
    assert klass.newstyle
    metaclass = klass.metaclass()
    assert isinstance(metaclass, ClassDef)
    assert metaclass.name == 'ABCMeta'


@require_version('3.0')
def test_metaclass_ancestors(builder):
    astroid = builder.string_build(dedent("""
    from abc import ABCMeta

    class FirstMeta(metaclass=ABCMeta): pass
    class SecondMeta(metaclass=type):
        pass

    class Simple:
        pass

    class FirstImpl(FirstMeta): pass
    class SecondImpl(FirstImpl): pass
    class ThirdImpl(Simple, SecondMeta):
        pass
    """))
    classes = {
        'ABCMeta': ('FirstImpl', 'SecondImpl'),
        'type': ('ThirdImpl', )
    }
    for metaclass, names in classes.items():
        for name in names:
            impl = astroid[name]
            meta = impl.metaclass()
            assert isinstance(meta, ClassDef)
            assert meta.name == metaclass


@require_version('3.0')
def test_annotation_support(builder):
    astroid = builder.string_build(dedent("""
    def test(a: int, b: str, c: None, d, e,
             *args: float, **kwargs: int)->int:
        pass
    """))
    func = astroid['test']
    assert isinstance(func.args.varargannotation, Name)
    assert func.args.varargannotation.name == 'float'
    assert isinstance(func.args.kwargannotation, Name)
    assert func.args.kwargannotation.name == 'int'
    assert isinstance(func.returns, Name)
    assert func.returns.name == 'int'
    arguments = func.args
    assert isinstance(arguments.annotations[0], Name)
    assert arguments.annotations[0].name == 'int'
    assert isinstance(arguments.annotations[1], Name)
    assert arguments.annotations[1].name == 'str'
    assert isinstance(arguments.annotations[2], Const)
    assert arguments.annotations[2].value is None
    assert arguments.annotations[3] is None
    assert arguments.annotations[4] is None

    astroid = builder.string_build(dedent("""
    def test(a: int=1, b: str=2):
        pass
    """))
    func = astroid['test']
    assert isinstance(func.args.annotations[0], Name)
    assert func.args.annotations[0].name == 'int'
    assert isinstance(func.args.annotations[1], Name)
    assert func.args.annotations[1].name == 'str'
    assert func.returns is None


@require_version('3.0')
def test_annotation_as_string():
    code1 = dedent('''
    def test(a, b:int=4, c=2, f:'lala'=4)->2:
        pass''')
    code2 = dedent('''
    def test(a:typing.Generic[T], c:typing.Any=24)->typing.Iterable:
        pass''')
    for code in (code1, code2):
        func = extract_node(code)
        assert func.as_string() == code


@require_version('3.5')
def test_unpacking_in_dicts():
    code = "{'x': 1, **{'y': 2}}"
    node = extract_node(code)
    assert node.as_string() == code
    keys = [key for (key, _) in node.items]
    assert isinstance(keys[0], nodes.Const)
    assert isinstance(keys[1], nodes.DictUnpack)


@require_version('3.5')
def test_nested_unpacking_in_dicts():
    code = "{'x': 1, **{'y': 2, **{'z': 3}}}"
    node = extract_node(code)
    assert node.as_string() == code


@require_version('3.5')
def test_unpacking_in_dict_getitem():
    node = extract_node('{1:2, **{2:3, 3:4}, **{5: 6}}')
    for key, expected in ((1, 2), (2, 3), (3, 4), (5, 6)):
        value = node.getitem(nodes.Const(key))
        assert isinstance(value, nodes.Const)
        assert value.value == expected


@require_version('3.6')
def test_format_string():
    code = "f'{greetings} {person}'"
    node = extract_node(code)
    assert node.as_string() == code


@require_version('3.6')
def test_underscores_in_numeral_literal():
    pairs = [
        ('10_1000', 101000),
        ('10_000_000', 10000000),
        ('0x_FF_FF', 65535),
    ]
    for value, expected in pairs:
        node = extract_node(value)
        inferred = next(node.infer())
        assert isinstance(inferred, nodes.Const)
        assert inferred.value == expected
