# Copyright (c) 2016 Claudiu Popa <pcmanticore@gmail.com>
# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER

import xml

import six

import astroid
from astroid import builder
from astroid import exceptions
from astroid import MANAGER
from astroid import test_utils
from astroid import objects
import pytest


BUILTINS = MANAGER.astroid_cache[six.moves.builtins.__name__]


def test_instance_special_model():
    ast_nodes = builder.extract_node('''
    class A:
        "test"
        def __init__(self):
            self.a = 42
    a = A()
    a.__class__ #@
    a.__module__ #@
    a.__doc__ #@
    a.__dict__ #@
    ''', module_name='collections')

    cls = next(ast_nodes[0].infer())
    assert isinstance(cls, astroid.ClassDef)
    assert cls.name == 'A'

    module = next(ast_nodes[1].infer())
    assert isinstance(module, astroid.Const)
    assert module.value == 'collections'

    doc = next(ast_nodes[2].infer())
    assert isinstance(doc, astroid.Const)
    assert doc.value == 'test'

    dunder_dict = next(ast_nodes[3].infer())
    assert isinstance(dunder_dict, astroid.Dict)
    attr = next(dunder_dict.getitem(astroid.Const('a')).infer())
    assert isinstance(attr, astroid.Const)
    assert attr.value == 42


@pytest.mark.xfail
def test_instance_local_attributes_overrides_object_model():
    # The instance lookup needs to be changed in order for this to work.
    ast_node = builder.extract_node('''
    class A:
        @property
        def __dict__(self):
              return []
    A().__dict__
    ''')
    inferred = next(ast_node.infer())
    assert isinstance(inferred, astroid.List)
    assert inferred.elts == []


def test_bound_method_model():
    ast_nodes = builder.extract_node('''
    class A:
        def test(self): pass
    a = A()
    a.test.__func__ #@
    a.test.__self__ #@
    ''')

    func = next(ast_nodes[0].infer())
    assert isinstance(func, astroid.FunctionDef)
    assert func.name == 'test'

    self_ = next(ast_nodes[1].infer())
    assert isinstance(self_, astroid.Instance)
    assert self_.name == 'A'


def test_unbound_method_model():
    ast_nodes = builder.extract_node('''
    class A:
        def test(self): pass
    t = A.test
    t.__class__ #@
    t.__func__ #@
    t.__self__ #@
    t.im_class #@
    t.im_func #@
    t.im_self #@
    ''')

    cls = next(ast_nodes[0].infer())
    assert isinstance(cls, astroid.ClassDef)
    unbound_name = 'instancemethod' if six.PY2 else 'function'
    assert cls.name == unbound_name

    func = next(ast_nodes[1].infer())
    assert isinstance(func, astroid.FunctionDef)
    assert func.name == 'test'

    self_ = next(ast_nodes[2].infer())
    assert isinstance(self_, astroid.Const)
    assert self_.value is None

    assert cls.name == next(ast_nodes[3].infer()).name
    assert func == next(ast_nodes[4].infer())
    assert next(ast_nodes[5].infer()).value is None


def test_priority_to_local_defined_values():
    ast_node = builder.extract_node('''
    class A:
        __doc__ = "first"
    A.__doc__ #@
    ''')
    inferred = next(ast_node.infer())
    assert isinstance(inferred, astroid.Const)
    assert inferred.value == "first"


@test_utils.require_version(maxver='3.0')
def test__mro__old_style():
    ast_node = builder.extract_node('''
    class A:
        pass
    A.__mro__
    ''')
    with pytest.raises(exceptions.InferenceError):
        next(ast_node.infer())


@test_utils.require_version(maxver='3.0')
def test__subclasses__old_style():
    ast_node = builder.extract_node('''
    class A:
        pass
    A.__subclasses__
    ''')
    with pytest.raises(exceptions.InferenceError):
        next(ast_node.infer())


def test_class_model_correct_mro_subclasses_proxied():
    ast_nodes = builder.extract_node('''
    class A(object):
        pass
    A.mro #@
    A.__subclasses__ #@
    ''')
    for node in ast_nodes:
        inferred = next(node.infer())
        assert isinstance(inferred, astroid.BoundMethod)
        assert isinstance(inferred._proxied, astroid.FunctionDef)
        assert isinstance(inferred.bound, astroid.ClassDef)
        assert inferred.bound.name == 'type'


@pytest.mark.skipif(six.PY3, reason="Needs old style classes")
def test_old_style_classes_no_mro():
    ast_node = builder.extract_node('''
    class A:
        pass
    A.mro #@
    ''')
    with pytest.raises(exceptions.InferenceError):
        next(ast_node.infer())


def test_class_model():
    ast_nodes = builder.extract_node('''
    class A(object):
        "test"

    class B(A): pass
    class C(A): pass

    A.__module__ #@
    A.__name__ #@
    A.__qualname__ #@
    A.__doc__ #@
    A.__mro__ #@
    A.mro() #@
    A.__bases__ #@
    A.__class__ #@
    A.__dict__ #@
    A.__subclasses__() #@
    ''', module_name='collections')

    module = next(ast_nodes[0].infer())
    assert isinstance(module, astroid.Const)
    assert module.value == 'collections'

    name = next(ast_nodes[1].infer())
    assert isinstance(name, astroid.Const)
    assert name.value == 'A'

    qualname = next(ast_nodes[2].infer())
    assert isinstance(qualname, astroid.Const)
    assert qualname.value == 'collections.A'

    doc = next(ast_nodes[3].infer())
    assert isinstance(doc, astroid.Const)
    assert doc.value == 'test'

    mro = next(ast_nodes[4].infer())
    assert isinstance(mro, astroid.Tuple)
    assert [cls.name for cls in mro.elts] == ['A', 'object']

    called_mro = next(ast_nodes[5].infer())
    assert called_mro.elts == mro.elts

    bases = next(ast_nodes[6].infer())
    assert isinstance(bases, astroid.Tuple)
    assert [cls.name for cls in bases.elts] == ['object']

    cls = next(ast_nodes[7].infer())
    assert isinstance(cls, astroid.ClassDef)
    assert cls.name == 'type'

    cls_dict = next(ast_nodes[8].infer())
    assert isinstance(cls_dict, astroid.Dict)

    subclasses = next(ast_nodes[9].infer())
    assert isinstance(subclasses, astroid.List)
    assert [cls.name for cls in subclasses.elts] == ['B', 'C']


def test_priority_to_local_defined_values2():
    ast_node = astroid.parse('''
    __file__ = "mine"
    ''')
    file_value = next(ast_node.igetattr('__file__'))
    assert isinstance(file_value, astroid.Const)
    assert file_value.value == "mine"


def test__path__not_a_package():
    ast_node = builder.extract_node('''
    import sys
    sys.__path__ #@
    ''')
    with pytest.raises(exceptions.InferenceError):
        next(ast_node.infer())


def test_module_model():
    ast_nodes = builder.extract_node('''
    import xml
    xml.__path__ #@
    xml.__name__ #@
    xml.__doc__ #@
    xml.__file__ #@
    xml.__spec__ #@
    xml.__loader__ #@
    xml.__cached__ #@
    xml.__package__ #@
    xml.__dict__ #@
    ''')

    path = next(ast_nodes[0].infer())
    assert isinstance(path, astroid.List)
    assert isinstance(path.elts[0], astroid.Const)
    assert path.elts[0].value == xml.__path__[0]

    name = next(ast_nodes[1].infer())
    assert isinstance(name, astroid.Const)
    assert name.value == 'xml'

    doc = next(ast_nodes[2].infer())
    assert isinstance(doc, astroid.Const)
    assert doc.value == xml.__doc__

    file_ = next(ast_nodes[3].infer())
    assert isinstance(file_, astroid.Const)
    assert file_.value == xml.__file__.replace(".pyc", ".py")

    for ast_node in ast_nodes[4:7]:
        inferred = next(ast_node.infer())
        assert inferred is astroid.Uninferable

    package = next(ast_nodes[7].infer())
    assert isinstance(package, astroid.Const)
    assert package.value == 'xml'

    dict_ = next(ast_nodes[8].infer())
    assert isinstance(dict_, astroid.Dict)


def test_partial_descriptor_support():
    bound, result = builder.extract_node('''
    class A(object): pass
    def test(self): return 42
    f = test.__get__(A(), A)
    f #@
    f() #@
    ''')
    bound = next(bound.infer())
    assert isinstance(bound, astroid.BoundMethod)
    assert bound._proxied._proxied.name == 'test'
    result = next(result.infer())
    assert isinstance(result, astroid.Const)
    assert result.value == 42


@pytest.mark.xfail
def test_descriptor_not_inferrring_self():
    # We can't infer __get__(X, Y)() when the bounded function
    # uses self, because of the tree's parent not being propagating good enough.
    result = builder.extract_node('''
    class A(object):
        x = 42
    def test(self): return self.x
    f = test.__get__(A(), A)
    f() #@
    ''')
    result = next(result.infer())
    assert isinstance(result, astroid.Const)
    assert result.value == 42


def test_descriptors_binding_invalid():
    ast_nodes = builder.extract_node('''
    class A: pass
    def test(self): return 42
    test.__get__()() #@
    test.__get__(1)() #@
    test.__get__(2, 3, 4) #@
    ''')
    for node in ast_nodes:
        with pytest.raises(exceptions.InferenceError):
            next(node.infer())


def test_function_model():
    ast_nodes = builder.extract_node('''
    def func(a=1, b=2):
        """test"""
    func.__name__ #@
    func.__doc__ #@
    func.__qualname__ #@
    func.__module__  #@
    func.__defaults__ #@
    func.__dict__ #@
    func.__globals__ #@
    func.__code__ #@
    func.__closure__ #@
    ''', module_name='collections')

    name = next(ast_nodes[0].infer())
    assert isinstance(name, astroid.Const)
    assert name.value == 'func'

    doc = next(ast_nodes[1].infer())
    assert isinstance(doc, astroid.Const)
    assert doc.value == 'test'

    qualname = next(ast_nodes[2].infer())
    assert isinstance(qualname, astroid.Const)
    assert qualname.value == 'collections.func'

    module = next(ast_nodes[3].infer())
    assert isinstance(module, astroid.Const)
    assert module.value == 'collections'

    defaults = next(ast_nodes[4].infer())
    assert isinstance(defaults, astroid.Tuple)
    assert [default.value for default in defaults.elts] == [1, 2]

    dict_ = next(ast_nodes[5].infer())
    assert isinstance(dict_, astroid.Dict)

    globals_ = next(ast_nodes[6].infer())
    assert isinstance(globals_, astroid.Dict)

    for ast_node in ast_nodes[7:9]:
        assert next(ast_node.infer()) is astroid.Uninferable


@test_utils.require_version(minver='3.0')
def test_empty_return_annotation():
    ast_node = builder.extract_node('''
    def test(): pass
    test.__annotations__
    ''')
    annotations = next(ast_node.infer())
    assert isinstance(annotations, astroid.Dict)
    assert len(annotations.items) == 0


@test_utils.require_version(minver='3.0')
def test_annotations_kwdefaults():
    ast_node = builder.extract_node('''
    def test(a: 1, *args: 2, f:4='lala', **kwarg:3)->2: pass
    test.__annotations__ #@
    test.__kwdefaults__ #@
    ''')
    annotations = next(ast_node[0].infer())
    assert isinstance(annotations, astroid.Dict)
    assert isinstance(annotations.getitem(astroid.Const('return')), astroid.Const)
    assert annotations.getitem(astroid.Const('return')).value == 2
    assert isinstance(annotations.getitem(astroid.Const('a')), astroid.Const)
    assert annotations.getitem(astroid.Const('a')).value == 1
    assert annotations.getitem(astroid.Const('args')).value == 2
    assert annotations.getitem(astroid.Const('kwarg')).value == 3

    # Currently not enabled.
    # assert annotations.getitem('f').value == 4

    kwdefaults = next(ast_node[1].infer())
    assert isinstance(kwdefaults, astroid.Dict)
    # assert kwdefaults.getitem('f').value == 'lala'


@test_utils.require_version(maxver='3.0')
def test_function_model_for_python2():
    ast_nodes = builder.extract_node('''
    def test(a=1):
      "a"

    test.func_name #@
    test.func_doc #@
    test.func_dict #@
    test.func_globals #@
    test.func_defaults #@
    test.func_code #@
    test.func_closure #@
    ''')
    name = next(ast_nodes[0].infer())
    assert isinstance(name, astroid.Const)
    assert name.value == 'test'
    doc = next(ast_nodes[1].infer())
    assert isinstance(doc, astroid.Const)
    assert doc.value == 'a'
    pydict = next(ast_nodes[2].infer())
    assert isinstance(pydict, astroid.Dict)
    pyglobals = next(ast_nodes[3].infer())
    assert isinstance(pyglobals, astroid.Dict)
    defaults = next(ast_nodes[4].infer())
    assert isinstance(defaults, astroid.Tuple)
    for node in ast_nodes[5:]:
        assert next(node.infer()) is astroid.Uninferable


def test_model():
    ast_nodes = builder.extract_node('''
    def test():
       "a"
       yield

    gen = test()
    gen.__name__ #@
    gen.__doc__ #@
    gen.gi_code #@
    gen.gi_frame #@
    gen.send #@
    ''')

    name = next(ast_nodes[0].infer())
    assert name.value == 'test'

    doc = next(ast_nodes[1].infer())
    assert doc.value == 'a'

    gi_code = next(ast_nodes[2].infer())
    assert isinstance(gi_code, astroid.ClassDef)
    assert gi_code.name == 'gi_code'

    gi_frame = next(ast_nodes[3].infer())
    assert isinstance(gi_frame, astroid.ClassDef)
    assert gi_frame.name == 'gi_frame'

    send = next(ast_nodes[4].infer())
    assert isinstance(send, astroid.BoundMethod)


@pytest.mark.skipif(six.PY2, reason="needs Python 3")
def test_model_py3():
    ast_nodes = builder.extract_node('''
    try:
        x[42]
    except ValueError as err:
       err.args #@
       err.__traceback__ #@

       err.message #@
    ''')
    args = next(ast_nodes[0].infer())
    assert isinstance(args, astroid.Tuple)
    tb = next(ast_nodes[1].infer())
    assert isinstance(tb, astroid.Instance)
    assert tb.name == 'traceback'

    with pytest.raises(exceptions.InferenceError):
        next(ast_nodes[2].infer())


@pytest.mark.skipif(six.PY3, reason="needs Python 2")
def test_model_py2():
    ast_nodes = builder.extract_node('''
    try:
        x[42]
    except ValueError as err:
       err.args #@
       err.message #@

       err.__traceback__ #@
    ''')
    args = next(ast_nodes[0].infer())
    assert isinstance(args, astroid.Tuple)
    message = next(ast_nodes[1].infer())
    assert isinstance(message, astroid.Const)

    with pytest.raises(exceptions.InferenceError):
        next(ast_nodes[2].infer())


def test__class__():
    ast_node = builder.extract_node('{}.__class__')
    inferred = next(ast_node.infer())
    assert isinstance(inferred, astroid.ClassDef)
    assert inferred.name == 'dict'


def test_attributes_inferred_as_methods():
    ast_nodes = builder.extract_node('''
    {}.values #@
    {}.items #@
    {}.keys #@
    ''')
    for node in ast_nodes:
        inferred = next(node.infer())
        assert isinstance(inferred, astroid.BoundMethod)


@pytest.mark.skipif(six.PY3, reason="needs Python 2")
def test_concrete_objects_for_dict_methods():
    ast_nodes = builder.extract_node('''
    {1:1, 2:3}.values() #@
    {1:1, 2:3}.keys() #@
    {1:1, 2:3}.items() #@
    ''')
    values = next(ast_nodes[0].infer())
    assert isinstance(values, astroid.List)
    assert [value.value for value in values.elts] == [1, 3]

    keys = next(ast_nodes[1].infer())
    assert isinstance(keys, astroid.List)
    assert [key.value for key in keys.elts] == [1, 2]

    items = next(ast_nodes[2].infer())
    assert isinstance(items, astroid.List)
    for expected, elem in zip([(1, 1), (2, 3)], items.elts):
        assert isinstance(elem, astroid.Tuple)
        assert list(expected) == [elt.value for elt in elem.elts]


@pytest.mark.skipif(six.PY2, reason="needs Python 3")
def test_wrapper_objects_for_dict_methods_python3():
    ast_nodes = builder.extract_node('''
    {1:1, 2:3}.values() #@
    {1:1, 2:3}.keys() #@
    {1:1, 2:3}.items() #@
    ''')
    values = next(ast_nodes[0].infer())
    assert isinstance(values, objects.DictValues)
    assert [elt.value for elt in values.elts] == [1, 3]
    keys = next(ast_nodes[1].infer())
    assert isinstance(keys, objects.DictKeys)
    assert [elt.value for elt in keys.elts] == [1, 2]
    items = next(ast_nodes[2].infer())
    assert isinstance(items, objects.DictItems)


@pytest.mark.skipif(six.PY2, reason="needs Python 3")
def test_lru_cache():
    ast_nodes = builder.extract_node('''
    import functools
    class Foo(object):
        @functools.lru_cache()
        def foo():
            pass
    f = Foo()
    f.foo.cache_clear #@
    f.foo.__wrapped__ #@
    f.foo.cache_info() #@
    ''')
    cache_clear = next(ast_nodes[0].infer())
    assert isinstance(cache_clear, astroid.BoundMethod)
    wrapped = next(ast_nodes[1].infer())
    assert isinstance(wrapped, astroid.FunctionDef)
    assert wrapped.name == 'foo'
    cache_info = next(ast_nodes[2].infer())
    assert isinstance(cache_info, astroid.Instance)
