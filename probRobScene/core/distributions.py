"""Objects representing distributions that can be sampled from."""
import abc
import collections
import functools
import itertools
import math
import random
import warnings
from typing import Iterable, Tuple

import numpy
import scipy

from probRobScene.core.lazy_eval import (LazilyEvaluable, requiredProperties, needs_lazy_evaluation, value_in_context,
                                         makeDelayedFunctionCall)
from probRobScene.core.utils import argsToString, areEquivalent, RuntimeParseError, cached, sqrt2


## Misc

def dependencies(thing):
    """Dependencies which must be sampled before this value."""
    return getattr(thing, '_dependencies', ())


def needs_sampling(thing):
    """Whether this value requires sampling."""
    return isinstance(thing, Distribution) or dependencies(thing)


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


class Bucketable(abc.ABC):
    @abc.abstractmethod
    def bucket(self, buckets=None):
        raise NotImplementedError

    @abc.abstractmethod
    def clone(self):
        raise NotImplementedError


class Samplable(LazilyEvaluable, abc.ABC):
    """Abstract class for values which can be sampled, possibly depending on other values.

    Samplables may specify a proxy object 'self._conditioned' which must have the same
    distribution as the original after conditioning on the scenario's requirements. This
    allows transparent conditioning without modifying Samplable fields of immutable objects.
    """

    def __init__(self, dependencies):
        deps = []
        props = set()
        for dep in dependencies:
            if needs_sampling(dep) or needs_lazy_evaluation(dep):
                deps.append(dep)
                props.update(requiredProperties(dep))
        super().__init__(props)
        self._dependencies = tuple(deps)  # fixed order for reproducibility
        self._conditioned = self  # version (partially) conditioned on requirements

    @abc.abstractmethod
    def sample_given_dependencies(self, dep_values):
        raise NotImplementedError("Must specify how to sample class when given dependencies")

    def evaluateIn(self, context):
        """See LazilyEvaluable.evaluateIn."""
        value = super().evaluateIn(context)
        # Check that all dependencies have been evaluated
        assert all(not needs_lazy_evaluation(dep) for dep in value._dependencies)
        return value


def condition_to(s: Samplable, value: Samplable):
    """Condition this value to another value with the same conditional distribution."""
    assert isinstance(value, Samplable)
    s._conditioned = value


def dependency_tree(s: Samplable):
    """Debugging method to print the dependency tree of a Samplable."""
    dep_tree = [str(s)]
    for dep in dependencies(s):
        for line in dep.dependency_tree():
            dep_tree.append('  ' + line)
    return dep_tree


def sample(obj: Samplable, subsamples=None):
    """Sample this value, optionally given some values already sampled."""
    if subsamples is None:
        subsamples = DefaultIdentityDict()
    for child in obj._conditioned._dependencies:
        if child not in subsamples:
            ch_sample = sample(child, subsamples)
            subsamples[child] = ch_sample
    return obj._conditioned.sample_given_dependencies(subsamples)


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


class Distribution(Samplable, abc.ABC):
    """Abstract class for distributions."""

    defaultValueType = float

    def __init__(self, *dependencies, value_type=None):
        super().__init__(dependencies)
        self.valueType = self.defaultValueType if value_type is None else value_type

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):  # ignore special attributes
            return super().__getattr__(name)
        return AttributeDistribution(name, self)


## Derived distributions

class CustomDistribution(Distribution):
    """Distribution with a custom sampler given by an arbitrary function"""

    def __init__(self, sampler, *dependencies, name='CustomDistribution', evaluator=None):
        super().__init__(*dependencies)
        self.sampler = sampler
        self.name = name
        self.evaluator = evaluator

    def sample_given_dependencies(self, dep_values):
        return self.sampler(dep_values)

    def evaluateInner(self, context):
        if self.evaluator is None:
            raise NotImplementedError('evaluateIn() not supported by this distribution')
        return self.evaluator(self, context)

    def isEquivalentTo(self, other):
        if not type(other) is CustomDistribution:
            return False
        return (self.sampler == other.sampler
                and self.name == other.name
                and self.evaluator == other.evaluator)

    def __str__(self):
        return f'{self.name}{argsToString(self.dependencies)}'


class TupleDistribution(Distribution, collections.abc.Sequence):
    """Distributions over tuples (or namedtuples, or lists)."""

    def __init__(self, *coordinates, builder=tuple):
        super().__init__(*coordinates)
        self.coordinates = coordinates
        self.builder = builder

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]

    def sample_given_dependencies(self, dep_values):
        return self.builder(dep_values[coordinate] for coordinate in self.coordinates)

    def evaluateInner(self, context):
        coordinates = (value_in_context(coord, context) for coord in self.coordinates)
        return TupleDistribution(*coordinates, builder=self.builder)

    def isEquivalentTo(self, other):
        if not type(other) is TupleDistribution:
            return False
        return (areEquivalent(self.coordinates, other.coordinates)
                and self.builder == other.builder)

    def __str__(self):
        coords = ', '.join(str(c) for c in self.coordinates)
        return f'({coords}, builder={self.builder})'


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


class FunctionDistribution(Distribution):
    """Distribution resulting from passing distributions to a function"""

    def __init__(self, func, args, kwargs, support=None):
        args = tuple(to_distribution(arg) for arg in args)
        kwargs = {name: to_distribution(arg) for name, arg in kwargs.items()}
        super().__init__(*args, *kwargs.values())
        self.function = func
        self.arguments = args
        self.kwargs = kwargs
        self.support = support

    def sample_given_dependencies(self, dep_values):
        args = tuple(dep_values[arg] for arg in self.arguments)
        kwargs = {name: dep_values[arg] for name, arg in self.kwargs.items()}
        return self.function(*args, **kwargs)

    def evaluateInner(self, context):
        function = value_in_context(self.function, context)
        arguments = tuple(value_in_context(arg, context) for arg in self.arguments)
        kwargs = {name: value_in_context(arg, context) for name, arg in self.kwargs.items()}
        return FunctionDistribution(function, arguments, kwargs)

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


class MethodDistribution(Distribution):
    """Distribution resulting from passing distributions to a method of a fixed object"""

    def __init__(self, method, obj, args, kwargs):
        args = tuple(to_distribution(arg) for arg in args)
        kwargs = {name: to_distribution(arg) for name, arg in kwargs.items()}
        super().__init__(*args, *kwargs.values())
        self.method = method
        self.object = obj
        self.arguments = args
        self.kwargs = kwargs

    def sample_given_dependencies(self, dep_values):
        args = (dep_values[arg] for arg in self.arguments)
        kwargs = {name: dep_values[arg] for name, arg in self.kwargs.items()}
        return self.method(self.object, *args, **kwargs)

    def evaluateInner(self, context):
        obj = value_in_context(self.object, context)
        arguments = tuple(value_in_context(arg, context) for arg in self.arguments)
        kwargs = {name: value_in_context(arg, context) for name, arg in self.kwargs.items()}
        return MethodDistribution(self.method, obj, arguments, kwargs)

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


class AttributeDistribution(Distribution):
    """Distribution resulting from accessing an attribute of a distribution"""

    def __init__(self, attribute, obj):
        super().__init__(obj)
        self.attribute = attribute
        self.object = obj

    def sample_given_dependencies(self, dep_values):
        obj = dep_values[self.object]
        return getattr(obj, self.attribute)

    def evaluateInner(self, context):
        obj = value_in_context(self.object, context)
        return AttributeDistribution(self.attribute, obj)

    def support_interval(self):
        obj = self.object
        if isinstance(obj, Options):
            attrs = (getattr(opt, self.attribute) for opt in obj.options)
            mins, maxes = zip(*(support_interval(attr) for attr in attrs))
            l = None if any(sl is None for sl in mins) else min(mins)
            r = None if any(sr is None for sr in maxes) else max(maxes)
            return l, r
        return None, None

    def isEquivalentTo(self, other):
        if not type(other) is AttributeDistribution:
            return False
        return (self.attribute == other.attribute
                and areEquivalent(self.object, other.object))

    def __str__(self):
        return f'{self.object}.{self.attribute}'


class OperatorDistribution(Distribution):
    """Distribution resulting from applying an operator to one or more distributions"""

    def __init__(self, operator, obj, operands):
        operands = tuple(to_distribution(arg) for arg in operands)
        super().__init__(obj, *operands)
        self.operator = operator
        self.object = obj
        self.operands = operands

    def sample_given_dependencies(self, dep_values):
        first = dep_values[self.object]
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

    def evaluateInner(self, context):
        obj = value_in_context(self.object, context)
        operands = tuple(value_in_context(arg, context) for arg in self.operands)
        return OperatorDistribution(self.operator, obj, operands)

    def support_interval(self):
        if self.operator in ('__add__', '__radd__', '__sub__', '__rsub__', '__truediv__'):
            assert len(self.operands) == 1
            l1, r1 = support_interval(self.object)
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
                and areEquivalent(self.object, other.object)
                and areEquivalent(self.operands, other.operands))

    def __str__(self):
        return f'{self.object}.{self.operator}{argsToString(self.operands)}'


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

import probRobScene.core.type_support as type_support


class MultiplexerDistribution(Distribution):
    """Distribution selecting among values based on another distribution."""

    def __init__(self, index, options):
        self.index = index
        self.options = tuple(to_distribution(opt) for opt in options)
        assert len(self.options) > 0
        valueType = type_support.unifyingType(self.options)
        super().__init__(index, *self.options, value_type=valueType)

    def sample_given_dependencies(self, dep_values):
        idx = dep_values[self.index]
        assert 0 <= idx < len(self.options), (idx, len(self.options))
        return dep_values[self.options[idx]]

    def evaluateInner(self, context):
        return type(self)(value_in_context(self.index, context),
                          (value_in_context(opt, context) for opt in self.options))

    def isEquivalentTo(self, other):
        if not type(other) == type(self):
            return False
        return (areEquivalent(self.index, other.index)
                and areEquivalent(self.options, other.options))


## Simple distributions

class Range(Distribution, Bucketable):
    """Uniform distribution over a range"""

    def support_interval(self):
        return self.low, self.high

    def __init__(self, low, high):
        low = type_support.toScalar(low, f'Range endpoint {low} is not a scalar')
        high = type_support.toScalar(high, f'Range endpoint {high} is not a scalar')
        super().__init__(low, high)
        self.low = low
        self.high = high

    def __contains__(self, obj):
        return self.low <= obj <= self.high

    def clone(self):
        return Range(self.low, self.high)

    def bucket(self, buckets=None):
        if buckets is None:
            buckets = 5
        if not isinstance(buckets, int) or buckets < 1:
            raise RuntimeError(f'Invalid buckets for Range.bucket: {buckets}')
        if not isinstance(self.low, float) or not isinstance(self.high, float):
            raise RuntimeError(f'Cannot bucket Range with non-constant endpoints')
        endpoints = numpy.linspace(self.low, self.high, buckets + 1)
        ranges = []
        for i, left in enumerate(endpoints[:-1]):
            right = endpoints[i + 1]
            ranges.append(Range(left, right))
        return Options(ranges)

    def sample_given_dependencies(self, dep_values):
        return random.uniform(dep_values[self.low], dep_values[self.high])

    def evaluateInner(self, context):
        low = value_in_context(self.low, context)
        high = value_in_context(self.high, context)
        return Range(low, high)

    def isEquivalentTo(self, other):
        if not type(other) is Range:
            return False
        return (areEquivalent(self.low, other.low)
                and areEquivalent(self.high, other.high))

    def __str__(self):
        return f'Range({self.low}, {self.high})'


class Normal(Distribution, Bucketable):
    """Normal distribution"""

    def __init__(self, mean, stddev):
        mean = type_support.toScalar(mean, f'Normal mean {mean} is not a scalar')
        stddev = type_support.toScalar(stddev, f'Normal stddev {stddev} is not a scalar')
        super().__init__(mean, stddev)
        self.mean = mean
        self.stddev = stddev

    @staticmethod
    def cdf(mean, stddev, x):
        return (1 + math.erf((x - mean) / (sqrt2 * stddev))) / 2

    @staticmethod
    def cdfinv(mean, stddev, x):
        return mean + (sqrt2 * stddev * scipy.special.erfinv(2 * x - 1))

    def clone(self):
        return Normal(self.mean, self.stddev)

    def bucket(self, buckets=None):
        if not isinstance(self.stddev, float):  # TODO relax restriction?
            raise RuntimeError(f'Cannot bucket Normal with non-constant standard deviation')
        if buckets is None:
            buckets = 5
        if isinstance(buckets, int):
            if buckets < 1:
                raise RuntimeError(f'Invalid buckets for Normal.bucket: {buckets}')
            elif buckets == 1:
                endpoints = []
            elif buckets == 2:
                endpoints = [0]
            else:
                left = self.stddev * (-(buckets - 3) / 2 - 0.5)
                right = self.stddev * ((buckets - 3) / 2 + 0.5)
                endpoints = numpy.linspace(left, right, buckets - 1)
        else:
            endpoints = tuple(buckets)
            for i, v in enumerate(endpoints[:-1]):
                if v >= endpoints[i + 1]:
                    raise RuntimeError('Non-increasing bucket endpoints for '
                                       f'Normal.bucket: {endpoints}')
        if len(endpoints) == 0:
            return Options([self.clone()])
        buckets = [(-math.inf, endpoints[0])]
        buckets.extend((v, endpoints[i + 1]) for i, v in enumerate(endpoints[:-1]))
        buckets.append((endpoints[-1], math.inf))
        pieces = []
        probs = []
        for left, right in buckets:
            pieces.append(self.mean + TruncatedNormal(0, self.stddev, left, right))
            prob = (Normal.cdf(0, self.stddev, right)
                    - Normal.cdf(0, self.stddev, left))
            probs.append(prob)
        assert math.isclose(math.fsum(probs), 1), probs
        return Options(dict(zip(pieces, probs)))

    def sample_given_dependencies(self, dep_values):
        return random.gauss(dep_values[self.mean], dep_values[self.stddev])

    def evaluateInner(self, context):
        mean = value_in_context(self.mean, context)
        stddev = value_in_context(self.stddev, context)
        return Normal(mean, stddev)

    def isEquivalentTo(self, other):
        if not type(other) is Normal:
            return False
        return (areEquivalent(self.mean, other.mean)
                and areEquivalent(self.stddev, other.stddev))

    def __str__(self):
        return f'Normal({self.mean}, {self.stddev})'


class TruncatedNormal(Normal, Bucketable):
    """Truncated normal distribution."""

    def __init__(self, mean, stddev, low, high):
        if (not isinstance(low, (float, int))
                or not isinstance(high, (float, int))):  # TODO relax restriction?
            raise RuntimeError('Endpoints of TruncatedNormal must be constant')
        super().__init__(mean, stddev)
        self.low = low
        self.high = high

    def support_interval(self):
        return self.low, self.high

    def clone(self):
        return TruncatedNormal(self.mean, self.stddev, self.low, self.high)

    def bucket(self, buckets=None):
        if not isinstance(self.stddev, float):  # TODO relax restriction?
            raise RuntimeError('Cannot bucket TruncatedNormal with '
                               'non-constant standard deviation')
        if buckets is None:
            buckets = 5
        if isinstance(buckets, int):
            if buckets < 1:
                raise RuntimeError(f'Invalid buckets for TruncatedNormal.bucket: {buckets}')
            endpoints = numpy.linspace(self.low, self.high, buckets + 1)
        else:
            endpoints = tuple(buckets)
            if len(endpoints) < 2:
                raise RuntimeError('Too few bucket endpoints for '
                                   f'TruncatedNormal.bucket: {endpoints}')
            if endpoints[0] != self.low or endpoints[-1] != self.high:
                raise RuntimeError(f'TruncatedNormal.bucket endpoints {endpoints} '
                                   'do not match domain')
            for i, v in enumerate(endpoints[:-1]):
                if v >= endpoints[i + 1]:
                    raise RuntimeError('Non-increasing bucket endpoints for '
                                       f'TruncatedNormal.bucket: {endpoints}')
        pieces, probs = [], []
        for i, left in enumerate(endpoints[:-1]):
            right = endpoints[i + 1]
            pieces.append(TruncatedNormal(self.mean, self.stddev, left, right))
            prob = (Normal.cdf(self.mean, self.stddev, right)
                    - Normal.cdf(self.mean, self.stddev, left))
            probs.append(prob)
        return Options(dict(zip(pieces, probs)))

    def sample_given_dependencies(self, dep_values):
        # TODO switch to method less prone to underflow?
        mean, stddev = dep_values[self.mean], dep_values[self.stddev]
        alpha = (self.low - mean) / stddev
        beta = (self.high - mean) / stddev
        alpha_cdf = Normal.cdf(0, 1, alpha)
        beta_cdf = Normal.cdf(0, 1, beta)
        if beta_cdf - alpha_cdf < 1e-15:
            warnings.warn('low precision when sampling TruncatedNormal')
        unif = random.random()
        p = alpha_cdf + unif * (beta_cdf - alpha_cdf)
        return mean + (stddev * Normal.cdfinv(0, 1, p))

    def evaluateInner(self, context):
        mean = value_in_context(self.mean, context)
        stddev = value_in_context(self.stddev, context)
        return TruncatedNormal(mean, stddev, self.low, self.high)

    def isEquivalentTo(self, other):
        if not type(other) is TruncatedNormal:
            return False
        return (areEquivalent(self.mean, other.mean)
                and areEquivalent(self.stddev, other.stddev)
                and self.low == other.low and self.high == other.high)

    def __str__(self):
        return f'TruncatedNormal({self.mean}, {self.stddev}, {self.low}, {self.high})'


class DiscreteRange(Distribution, Bucketable):
    """Distribution over a range of integers."""

    def __init__(self, low, high, weights=None):
        if not isinstance(low, int):
            raise RuntimeError(f'DiscreteRange endpoint {low} is not a constant integer')
        if not isinstance(high, int):
            raise RuntimeError(f'DiscreteRange endpoint {high} is not a constant integer')
        if not low <= high:
            raise RuntimeError(f'DiscreteRange lower bound {low} is above upper bound {high}')
        if weights is None:
            weights = (1,) * (high - low + 1)
        else:
            weights = tuple(weights)
            assert len(weights) == high - low + 1
        super().__init__()
        self.low = low
        self.high = high
        self.weights = weights
        self.cumulativeWeights = tuple(itertools.accumulate(weights))
        self.options = tuple(range(low, high + 1))

    def __contains__(self, obj):
        return self.low <= obj <= self.high

    def clone(self):
        return DiscreteRange(self.low, self.high, self.weights)

    def bucket(self, buckets=None):
        return self.clone()  # already bucketed

    def sample_given_dependencies(self, dep_values):
        return random.choices(self.options, cum_weights=self.cumulativeWeights)[0]

    def isEquivalentTo(self, other):
        if not type(other) is DiscreteRange:
            return False
        return (self.low == other.low and self.high == other.high
                and self.weights == other.weights)

    def __str__(self):
        return f'DiscreteRange({self.low}, {self.high}, {self.weights})'


class Options(MultiplexerDistribution, Bucketable):
    """Distribution over a finite list of options.

    Specified by a dict giving probabilities; otherwise uniform over a given iterable.
    """

    def __init__(self, opts):
        if isinstance(opts, dict):
            options, weights = [], []
            for opt, prob in opts.items():
                if not isinstance(prob, (float, int)):
                    raise RuntimeParseError(f'discrete distribution weight {prob}'
                                            ' is not a constant number')
                if prob < 0:
                    raise RuntimeParseError(f'discrete distribution weight {prob} is negative')
                if prob == 0:
                    continue
                options.append(opt)
                weights.append(prob)
            self.optWeights = dict(zip(options, weights))
        else:
            weights = None
            options = tuple(opts)
            self.optWeights = None
        if len(options) == 0:
            raise RuntimeParseError('tried to make discrete distribution over empty domain!')

        index = self.makeSelector(len(options) - 1, weights)
        super().__init__(index, options)

    @staticmethod
    def makeSelector(n, weights):
        return DiscreteRange(0, n, weights)

    def clone(self):
        return Options(self.optWeights if self.optWeights else self.options)

    def bucket(self, buckets=None):
        return self.clone()  # already bucketed

    def evaluateInner(self, context):
        if self.optWeights is None:
            return type(self)(value_in_context(opt, context) for opt in self.options)
        else:
            return type(self)({value_in_context(opt, context): wt
                               for opt, wt in self.optWeights.items()})

    def isEquivalentTo(self, other):
        if not type(other) == type(self):
            return False
        return (areEquivalent(self.index, other.index)
                and areEquivalent(self.options, other.options))

    def __str__(self):
        if self.optWeights is not None:
            return f'{type(self).__name__}({self.optWeights})'
        else:
            return f'{type(self).__name__}{argsToString(self.options)}'
