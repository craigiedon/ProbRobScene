from __future__ import annotations
import abc
import collections
import functools
import itertools
import math
import random
from dataclasses import dataclass
from inspect import signature
from typing import Any, Iterable, List, Tuple, Optional, Sequence, Mapping, Callable

import scipy
from multimethod import multimethod

from probRobScene.core.lazy_eval import (LazilyEvaluable, needs_lazy_evaluation, makeDelayedFunctionCall, requiredProperties)

from probRobScene.core.utils import argsToString, areEquivalent, sqrt2


## Misc

def support_interval(thing):
    """Lower and upper bounds on this value, if known."""
    if hasattr(thing, "support_interval"):
        return thing.support_interval()
    if isinstance(thing, (int, float)):
        return thing, thing

    return None, None


def underlying_function(thing):
    """Original function underlying a distribution wrapper."""
    return getattr(thing, '__wrapped__', thing)


class RejectionException(Exception):
    """Exception used to signal that the sample currently being generated must be rejected."""
    pass


## Abstract distributions

class DefaultIdentityDict(dict):
    """Dictionary which is the identity map by default."""

    def __getitem__(self, key):
        if not isinstance(key, Samplable):  # to allow non-hashable objects
            return key
        return super().__getitem__(key)

    def __missing__(self, key):
        return key


class Samplable(LazilyEvaluable, abc.ABC):
    def required_properties(self) -> set:
        rps = set()
        for dep in self.dependencies():
            rps.update(requiredProperties(dep))
        return rps

    def sample_given_dependencies(self, dep_values):
        cls = self.__class__
        init_params = list(signature(cls.__init__).parameters.keys())[1:]
        current_vals = [self.__getattribute__(p) for p in init_params]
        context_vals = [dep_values[cv] for cv in current_vals]

        return cls(*context_vals)

    def dependencies(self) -> List:
        cls = self.__class__
        init_params = list(signature(cls.__init__).parameters.keys())[1:]
        current_vals = [self.__getattribute__(p) for p in init_params]
        return [cv for cv in current_vals if needs_sampling(cv) or needs_lazy_evaluation(cv)]


class Distribution(Samplable, abc.ABC):
    """Abstract class for distributions."""

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):  # ignore special attributes
            return super().__getattr__(name)
        if name == '_conditioned':
            return super().__getattr__(name)
        return AttributeDistribution(name, self)


@dataclass(init=False)
class TupleDistribution(Distribution, collections.abc.Sequence):
    """Distributions over tuples (or namedtuples, or lists)."""
    coordinates: Sequence
    builder: Callable

    def __init__(self, *coordinates, builder=tuple):
        self.coordinates = coordinates
        self.builder = builder

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]

    def sample_given_dependencies(self, dep_values):
        return self.builder(dep_values[coordinate] for coordinate in self.coordinates)

    def isEquivalentTo(self, other):
        if not type(other) is TupleDistribution:
            return False
        return (areEquivalent(self.coordinates, other.coordinates)
                and self.builder == other.builder)

    def __str__(self):
        coords = ', '.join(str(c) for c in self.coordinates)
        return f'({coords}, builder={self.builder})'


@dataclass(eq=False)
class FunctionDistribution(Distribution):
    """Distribution resulting from passing distributions to a function"""
    function: Callable
    arguments: Sequence
    kwargs: Mapping
    support: Optional = None

    def __post_init__(self):
        self.arguments = tuple(to_distribution(arg) for arg in self.arguments)
        self.kwargs = {name: to_distribution(arg) for name, arg in self.kwargs.items()}

    def sample_given_dependencies(self, dep_values):
        args = tuple(dep_values[arg] for arg in self.arguments)
        kwargs = {name: dep_values[arg] for name, arg in self.kwargs.items()}
        return self.function(*args, **kwargs)

    def dependencies(self) -> List:
        return [x for x in (*self.arguments, *self.kwargs.values()) if needs_sampling(x) or needs_lazy_evaluation(x)]

    def support_interval(self):
        if self.support is None:
            return None, None
        subsupports = (support_interval(arg) for arg in self.arguments)
        kwss = {name: support_interval(arg) for name, arg in self.kwargs.items()}
        return self.support(*subsupports, **kwss)

    def isEquivalentTo(self, other):
        if not type(other) is FunctionDistribution:
            return False
        return (self.function == other.function
                and areEquivalent(self.arguments, other.arguments)
                and areEquivalent(self.kwargs, other.kwargs)
                and self.support == other.support)

    def __str__(self):
        args = argsToString(itertools.chain(self.arguments, self.kwargs.items()))
        return f'{self.function.__name__}{args}'


@dataclass(unsafe_hash=True)
class MethodDistribution(Distribution):
    """Distribution resulting from passing distributions to a method of a fixed object"""
    method: Any
    object: Distribution
    arguments: Sequence
    kwargs: Mapping

    def __post_init__(self):
        self.arguments = tuple(to_distribution(arg) for arg in self.arguments)
        self.kwargs = {name: to_distribution(arg) for name, arg in self.kwargs.items()}

    def sample_given_dependencies(self, dep_values):
        args = (dep_values[arg] for arg in self.arguments)
        kwargs = {name: dep_values[arg] for name, arg in self.kwargs.items()}
        return self.method(self.object, *args, **kwargs)

    def isEquivalentTo(self, other):
        if not type(other) is MethodDistribution:
            return False
        return (self.method == other.method
                and areEquivalent(self.object, other.object)
                and areEquivalent(self.arguments, other.arguments)
                and areEquivalent(self.kwargs, other.kwargs))

    def __str__(self):
        args = argsToString(itertools.chain(self.arguments, self.kwargs.items()))
        return f'{self.object}.{self.method.__name__}{args}'


@dataclass(frozen=True, eq=False)
class AttributeDistribution(Distribution):
    """Distribution resulting from accessing an attribute of a distribution"""
    attribute: str
    object: Distribution

    def __post_init__(self):
        assert self.attribute != '_conditioned'

    def sample_given_dependencies(self, dep_values):
        obj = dep_values[self.object]
        return getattr(obj, self.attribute)

    def support_interval(self):
        # obj = self.object
        # if isinstance(obj, Options):
        #     attrs = (getattr(opt, self.attribute) for opt in obj.options)
        #     mins, maxes = zip(*(support_interval(attr) for attr in attrs))
        #     l = None if any(sl is None for sl in mins) else min(mins)
        #     r = None if any(sr is None for sr in maxes) else max(maxes)
        #     return l, r
        return None, None

    def isEquivalentTo(self, other):
        if not type(other) is AttributeDistribution:
            return False
        return (self.attribute == other.attribute
                and areEquivalent(self.object, other.object))

    # def __str__(self):
    #     return f'{self.object}.{self.attribute}'


@dataclass(frozen=True, eq=False)
class OperatorDistribution(Distribution):
    """Distribution resulting from applying an operator to one or more distributions"""
    operator: str
    obj: Distribution
    operands: Tuple

    def sample_given_dependencies(self, dep_values):
        first = dep_values[self.obj]
        rest = [dep_values[child] for child in self.operands]
        op = getattr(first, self.operator)
        result = op(*rest)
        # handle horrible int/float mismatch
        # TODO what is the right way to fix this???
        if result is NotImplemented and isinstance(first, int):
            first = float(first)
            op = getattr(first, self.operator)
            result = op(*rest)
        return result

    def support_interval(self):
        if self.operator in ('__add__', '__radd__', '__sub__', '__rsub__', '__truediv__'):
            assert len(self.operands) == 1
            l1, r1 = support_interval(self.obj)
            l2, r2 = support_interval(self.operands[0])
            if l1 is None or l2 is None or r1 is None or r2 is None:
                return None, None
            if self.operator == '__add__' or self.operator == '__radd__':
                l = l1 + l2
                r = r1 + r2
            elif self.operator == '__sub__':
                l = l1 - r2
                r = r1 - l2
            elif self.operator == '__rsub__':
                l = l2 - r1
                r = r2 - l1
            elif self.operator == '__truediv__':
                if l2 > 0:
                    l = l1 / r2 if l1 >= 0 else l1 / l2
                    r = r1 / l2 if r1 >= 0 else r1 / r2
                else:
                    l, r = None, None  # TODO improve
            return l, r
        return None, None

    def isEquivalentTo(self, other):
        if not type(other) is OperatorDistribution:
            return False
        return (self.operator == other.operator
                and areEquivalent(self.obj, other.object)
                and areEquivalent(self.operands, other.operands))

    def __str__(self):
        return f'{self.obj}.{self.operator}{argsToString(self.operands)}'


# Operators which can be applied to distributions.
# Note that we deliberately do not include comparisons and __bool__,
# since Scenic does not allow control flow to depend on random variables.
allowedOperators = [
    '__neg__',
    '__pos__',
    '__abs__',
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
    '__getitem__',
    '__call__'
]


def makeOperatorHandler(op):
    def handler(self, *args):
        return OperatorDistribution(op, self, args)

    return handler


for op in allowedOperators:
    setattr(Distribution, op, makeOperatorHandler(op))


## Simple distributions

@dataclass(frozen=True, eq=False)
class Range(Distribution):
    """Uniform distribution over a range"""
    low: float
    high: float

    def support_interval(self):
        return self.low, self.high

    def __contains__(self, obj):
        return self.low <= obj <= self.high

    def sample_given_dependencies(self, dep_values):
        return random.uniform(dep_values[self.low], dep_values[self.high])

    def isEquivalentTo(self, other):
        if not type(other) is Range:
            return False
        return (areEquivalent(self.low, other.low)
                and areEquivalent(self.high, other.high))

    def __str__(self):
        return f'Range({self.low}, {self.high})'


@dataclass(frozen=True, eq=False)
class Normal(Distribution):
    """Normal distribution"""
    mean: float
    stddev: float

    @staticmethod
    def cdf(mean, stddev, x):
        return (1 + math.erf((x - mean) / (sqrt2 * stddev))) / 2

    @staticmethod
    def cdfinv(mean, stddev, x):
        return mean + (sqrt2 * stddev * scipy.special.erfinv(2 * x - 1))

    def clone(self):
        return Normal(self.mean, self.stddev)

    def sample_given_dependencies(self, dep_values):
        return random.gauss(dep_values[self.mean], dep_values[self.stddev])

    def isEquivalentTo(self, other):
        if not type(other) is Normal:
            return False
        return (areEquivalent(self.mean, other.mean)
                and areEquivalent(self.stddev, other.stddev))

    def __str__(self):
        return f'Normal({self.mean}, {self.stddev})'


@dataclass(frozen=True, eq=False)
class DiscreteRange(Distribution):
    """Distribution over a range of integers."""
    low: int
    high: int
    weights: Optional[Tuple] = None

    def __contains__(self, obj):
        return self.low <= obj <= self.high

    def sample_given_dependencies(self, dep_values):
        options = tuple(range(self.low, self.high + 1))
        if self.weights is None:
            return random.choices(options)[0]

        return random.choices(options, weights=self.cumulativeWeights)[0]

    def isEquivalentTo(self, other):
        if not type(other) is DiscreteRange:
            return False
        return (self.low == other.low and self.high == other.high
                and self.weights == other.weights)

    def __str__(self):
        return f'DiscreteRange({self.low}, {self.high}, {self.weights})'


def sample(obj: Samplable, subsamples=None):
    """Sample this value, optionally given some values already sampled."""
    if subsamples is None:
        subsamples = DefaultIdentityDict()

    c_obj = obj
    if hasattr(obj, "_conditioned"):
        c_obj = obj._conditioned
    deps = c_obj.dependencies()
    for child in deps:
        if child not in subsamples:
            ch_sample = sample(child, subsamples)
            subsamples[child] = ch_sample
    c_obj_samp = c_obj.sample_given_dependencies(subsamples)
    assert not needs_sampling(c_obj_samp)
    return c_obj_samp


def sample_all(quantities: Iterable[Samplable]):
    """Sample all the given Samplables, which may have dependencies in common.

    Reproducibility note: the order in which the quantities are given can affect the
    order in which calls to random are made, affecting the final result.
    """
    subsamples = DefaultIdentityDict()
    for q in quantities:
        if q not in subsamples:
            subsamples[q] = sample(q, subsamples) if isinstance(q, Samplable) else q
    return {q: subsamples[q] for q in quantities}


def needs_sampling(thing) -> bool:
    """Whether this value requires sampling."""
    if not isinstance(thing, Samplable):
        return False

    return isinstance(thing, Distribution) or len(thing.dependencies()) > 0


def dependency_tree(s: Samplable):
    """Debugging method to print the dependency tree of a Samplable."""
    dep_tree = [str(s)]
    for dep in s.dependencies():
        for line in dep.dependency_tree():
            dep_tree.append('  ' + line)
    return dep_tree


def to_distribution(val):
    """Wrap Python data types with Distributions, if necessary.

    For example, tuples containing Samplables need to be converted into TupleDistributions
    in order to keep track of dependencies properly."""
    if isinstance(val, (tuple, list)):
        coords = [to_distribution(c) for c in val]
        if any(needs_sampling(c) or needs_lazy_evaluation(c) for c in coords):
            if isinstance(val, tuple) and hasattr(val, '_fields'):  # namedtuple
                builder = type(val)._make
            else:
                builder = type(val)
            return TupleDistribution(*coords, builder=builder)
    return val


def distributionFunction(method, support=None):
    """Decorator for wrapping a function so that it can take distributions as arguments."""

    @functools.wraps(method)
    def helper(*args, **kwargs):
        args = tuple(to_distribution(arg) for arg in args)
        kwargs = {name: to_distribution(arg) for name, arg in kwargs.items()}
        if any(needs_sampling(arg) for arg in itertools.chain(args, kwargs.values())):
            return FunctionDistribution(method, args, kwargs, support)
        elif any(needs_lazy_evaluation(arg) for arg in itertools.chain(args, kwargs.values())):
            # recursively call this helper (not the original function), since the delayed
            # arguments may evaluate to distributions, in which case we'll have to make a
            # FunctionDistribution
            return makeDelayedFunctionCall(helper, args, kwargs)
        else:
            return method(*args, **kwargs)

    return helper


def distributionMethod(method):
    """Decorator for wrapping a method so that it can take distributions as arguments."""

    @functools.wraps(method)
    def helper(self, *args, **kwargs):
        args = tuple(to_distribution(arg) for arg in args)
        kwargs = {name: to_distribution(arg) for name, arg in kwargs.items()}
        if any(needs_sampling(arg) for arg in itertools.chain(args, kwargs.values())):
            return MethodDistribution(method, self, args, kwargs)
        elif any(needs_lazy_evaluation(arg) for arg in itertools.chain(args, kwargs.values())):
            # see analogous comment in distributionFunction
            return makeDelayedFunctionCall(helper, (self,) + args, kwargs)
        else:
            return method(self, *args, **kwargs)

    return helper


def monotonicDistributionFunction(method):
    """Like distributionFunction, but additionally specifies that the function is monotonic."""

    def support(*subsupports, **kwss):
        mins, maxes = zip(*subsupports)
        kwmins = {name: interval[0] for name, interval in kwss.items()}
        kwmaxes = {name: interval[1] for name, interval in kwss.items()}
        l = None if None in mins or None in kwmins else method(*mins, **kwmins)
        r = None if None in maxes or None in kwmaxes else method(*maxes, **kwmaxes)
        return l, r

    return distributionFunction(method, support=support)
