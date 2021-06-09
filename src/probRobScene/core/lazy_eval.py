"""Support for lazy evaluation of expressions and specifiers."""
import abc
import itertools
from inspect import signature
from typing import Mapping, Any, Sequence, Callable

from multimethod import multimethod


class LazilyEvaluable(abc.ABC):
    """Values which may require evaluation in the context of an object being constructed.

    If a LazilyEvaluable specifies any properties it depends on, then it cannot be evaluated to a
    normal value except during the construction of an object which already has values for those
    properties.
    """

    @abc.abstractmethod
    def required_properties(self) -> set:
        pass


class DelayedArgument(LazilyEvaluable):
    """Specifier arguments requiring other properties to be evaluated first.

    The value of a DelayedArgument is given by a function mapping the context (object under
    construction) to a value.
    """

    def required_properties(self) -> set:
        return self.rps

    def __init__(self, required_props, valuation_func: Callable):
        self.value_func : Callable = valuation_func
        self.rps = required_props

    def __getattr__(self, name):
        return DelayedArgument(self.rps,
                               lambda context: getattr(evaluate_in(self, context), name))

    def __call__(self, *args, **kwargs):
        dargs = [toDelayedArgument(arg) for arg in args]
        kwdargs = {name: toDelayedArgument(arg) for name, arg in kwargs.items()}
        subprops = (darg.required_properties() for darg in itertools.chain(dargs, kwdargs.values()))
        props = self.required_properties().union(*subprops)

        def value(context):
            subvalues = (evaluate_in(darg, context) for darg in dargs)
            kwsvs = {name: darg.evaluate_in(context) for name, darg in kwdargs.items()}
            return evaluate_in(self, context)(*subvalues, **kwsvs)

        return DelayedArgument(props, value)


@multimethod
def evaluate_inner(x: LazilyEvaluable, context: Mapping[Any, Any]) -> Any:
    cls = x.__class__
    init_params = list(signature(cls.__init__).parameters.keys())[1:]
    current_vals = [x.__getattribute__(p) for p in init_params]
    context_vals = [value_in_context(cv, context) for cv in current_vals]

    return cls(*context_vals)

@multimethod
def evaluate_inner(x: DelayedArgument, context: Any) -> Any:
    return x.value_func(context)


def evaluate_in(x: LazilyEvaluable, context: Any) -> Any:
    """Evaluate this value in the context of an object being constructed.

    The object must define all of the properties on which this value depends.
    """
    assert all(hasattr(context, prop) for prop in x.required_properties())
    value = evaluate_inner(x, context)
    if needs_lazy_evaluation(value):
        print(needs_lazy_evaluation(value))
        print(evaluate_inner(x, context))
    assert not needs_lazy_evaluation(value), f"value {value} should not require further evaluation"
    return value




# Operators which can be applied to DelayedArguments
allowedOperators = [
    '__neg__',
    '__pos__',
    '__abs__',
    '__lt__', '__le__',
    '__eq__', '__ne__',
    '__gt__', '__ge__',
    '__add__', '__radd__',
    '__sub__', '__rsub__',
    '__mul__', '__rmul__',
    '__truediv__', '__rtruediv__',
    '__floordiv__', '__rfloordiv__',
    '__mod__', '__rmod__',
    '__divmod__', '__rdivmod__',
    '__pow__', '__rpow__',
    '__round__',
    '__len__',
    '__getitem__'
]


def make_delayed_operator_handler(op):
    def handler(self, *args):
        dargs = [toDelayedArgument(arg) for arg in args]
        props = self.required_properties().union(*(darg._requiredProperties for darg in dargs))

        def value(context):
            subvalues = (evaluate_in(darg, context) for darg in dargs)
            return getattr(self.evaluate_in(context), op)(*subvalues)

        return DelayedArgument(props, value)

    return handler


for op in allowedOperators:
    setattr(DelayedArgument, op, make_delayed_operator_handler(op))


def makeDelayedFunctionCall(func, args, kwargs):
    """Utility function for creating a lazily-evaluated function call."""
    dargs = [toDelayedArgument(arg) for arg in args]
    kwdargs = {name: toDelayedArgument(arg) for name, arg in kwargs.items()}
    props = set().union(*(darg.required_properties()
                          for darg in itertools.chain(dargs, kwdargs.values())))

    def value(context):
        subvalues = (evaluate_in(darg, context) for darg in dargs)
        kwsubvals = {name: darg.evaluate_in(context) for name, darg in kwdargs.items()}
        return func(*subvalues, **kwsubvals)

    return DelayedArgument(props, value)


@multimethod
def value_in_context(l: Sequence, context: Mapping) -> Sequence:
    return [value_in_context(x, context) for x in l]


@multimethod
def value_in_context(d: Mapping, context: Mapping) -> Mapping:
    return {k: value_in_context(v, context) for k, v in d.items()}


@multimethod
def value_in_context(value: Any, context: Mapping) -> Any:
    """Evaluate something in the context of an object being constructed."""
    try:
        return value.evaluate_in(context)
    except AttributeError:
        return value


def toDelayedArgument(thing):
    if isinstance(thing, DelayedArgument):
        return thing
    return DelayedArgument(set(), lambda context: thing)


def requiredProperties(thing):
    if isinstance(thing, LazilyEvaluable):
        return thing.required_properties()
    return set()


def needs_lazy_evaluation(thing) -> bool:
    del_arg = isinstance(thing, DelayedArgument)
    req_p = requiredProperties(thing)
    return del_arg or req_p