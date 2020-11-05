"""Support for checking Scenic types."""

import inspect
import sys

import numpy as np

from probRobScene.core.distributions import Distribution
from probRobScene.core.lazy_eval import (DelayedArgument, value_in_context, requiredProperties,
                                         needs_lazy_evaluation)
from probRobScene.core.utils import RuntimeParseError
from probRobScene.core.vectors import Vector, Vector3D


# Typing and coercion rules:
#
# coercible to a scalar:
#   float
#   int (by conversion to float)
#	numpy real scalar types (likewise)
# coercible to a heading:
#	anything coercible to a scalar
#	anything with a toHeading() method
# coercible to a Vector:
#   anything with a toVector() method
# coercible to an object of type T:
#   instances of T
#
# Finally, Distributions are coercible to T iff their valueType is.

## Basic types

class Heading:
    """Dummy class used as a target for type coercions to headings."""
    pass


def underlyingType(thing):
    """What type this value ultimately evaluates to, if we can tell."""
    if isinstance(thing, Distribution):
        return thing.valueType
    elif isinstance(thing, TypeChecker) and len(thing.types) == 1:
        return thing.types[0]
    else:
        return type(thing)


def isA(thing, ty):
    """Does this evaluate to a member of the given Scenic type?"""
    return issubclass(underlyingType(thing), ty)


def unifyingType(opts):  # TODO improve?
    """Most specific type unifying the given types."""
    types = [underlyingType(opt) for opt in opts]
    if all(issubclass(ty, (float, int)) for ty in types):
        return float
    mro = inspect.getmro(types[0])
    for parent in mro:
        if all(issubclass(ty, parent) for ty in types):
            return parent
    raise RuntimeError(f'broken MRO for types {types}')


def coerce(thing, types, error):
    """Coerce something into any of the given types, printing an error if impossible."""
    for ty in types:
        if isinstance(thing, Distribution):
            return thing
        if ty is float:
            return float(thing)
        elif ty is Heading:
            return thing.toHeading() if hasattr(thing, 'toHeading') else float(thing)
        elif ty is Vector:
            return thing.toVector()
        elif ty is Vector3D:
            return thing.to_vector_3d()
        elif isinstance(thing, ty):
            return thing

    print(f'Failed to coerce {thing} to {types}', file=sys.stderr)
    raise RuntimeParseError(error)


## Top-level type checking/conversion API

def toTypes(thing, types, typeError='wrong type'):
    """Convert something to any of the given types, printing an error if impossible."""
    if needs_lazy_evaluation(thing):
        # cannot check the type now; create proxy object to check type after evaluation
        return TypeChecker(thing, types, typeError)
    else:
        return coerce(thing, types, typeError)


def toType(thing, ty, typeError='wrong type'):
    """Convert something to a given type, printing an error if impossible."""
    return toTypes(thing, (ty,), typeError)


def toScalar(thing, typeError='non-scalar in scalar context'):
    """Convert something to a scalar, printing an error if impossible."""
    return toType(thing, float, typeError)


def toHeading(thing, typeError='non-heading in heading context'):
    """Convert something to a heading, printing an error if impossible."""
    return toType(thing, Heading, typeError)


def toVector(thing, typeError='non-vector in vector context'):
    """Convert something to a vector, printing an error if impossible."""
    return toType(thing, Vector, typeError)


def evaluateRequiringEqualTypes(func, thingA, thingB, typeError='type mismatch'):
    """Evaluate the func, assuming thingA and thingB have the same type.

    If func produces a lazy value, it should not have any required properties beyond
    those of thingA and thingB."""
    if not needs_lazy_evaluation(thingA) and not needs_lazy_evaluation(thingB):
        if underlyingType(thingA) is not underlyingType(thingB):
            raise RuntimeParseError(typeError)
        return func()
    else:
        # cannot check the types now; create proxy object to check types after evaluation
        return TypeEqualityChecker(func, thingA, thingB, typeError)


## Proxy objects for lazy type checking

class TypeChecker(DelayedArgument):
    """Checks that a given lazy value has one of a given list of types."""

    def __init__(self, arg, types, error):
        def check(context):
            val = arg.evaluateIn(context)
            return coerce(val, types, error)

        super().__init__(requiredProperties(arg), check)
        self.inner = arg
        self.types = types

    def __str__(self):
        return f'TypeChecker({self.inner},{self.types})'


class TypeEqualityChecker(DelayedArgument):
    """Lazily evaluates a function, after checking that two lazy values have the same type."""

    def __init__(self, func, checkA, checkB, error):
        props = requiredProperties(checkA) | requiredProperties(checkB)

        def check(context):
            ca = value_in_context(checkA, context)
            cb = value_in_context(checkB, context)
            if underlyingType(ca) is not underlyingType(cb):
                raise RuntimeParseError(error)
            return value_in_context(func(), context)

        super().__init__(props, check)
        self.inner = func
        self.checkA = checkA
        self.checkB = checkB

    def __str__(self):
        return f'TypeEqualityChecker({self.inner},{self.checkA},{self.checkB})'
