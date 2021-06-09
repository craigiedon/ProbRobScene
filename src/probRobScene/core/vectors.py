from __future__ import annotations
import collections
import functools
import itertools
import math
from collections import Sequence, Callable
from dataclasses import dataclass
from math import sin, cos
from typing import Tuple, Mapping, List

import numpy as np
from scipy.spatial.transform import Rotation as R

import probRobScene.core.utils as utils
from probRobScene.core.distributions import (Samplable, Distribution, MethodDistribution,
                                             makeOperatorHandler, distributionFunction, needs_sampling)
from probRobScene.core.lazy_eval import needs_lazy_evaluation, makeDelayedFunctionCall


class VectorDistribution(Distribution):
    """A distribution over Vectors."""
    def toVector(self):
        return self


class CustomVectorDistribution(VectorDistribution):
    """Distribution with a custom sampler given by an arbitrary function."""

    def __init__(self, sampler, *dependencies, name='CustomVectorDistribution', evaluator=None):
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

    def __str__(self):
        deps = utils.argsToString(self.dependencies)
        return f'{self.name}{deps}'


class VectorOperatorDistribution(VectorDistribution):
    """Vector version of OperatorDistribution."""

    def __init__(self, operator, obj, operands):
        self.operator = operator
        self.obj = obj
        self.operands = operands

    def sample_given_dependencies(self, dep_values):
        first = dep_values[self.obj]
        rest = (dep_values[child] for child in self.operands)
        op = getattr(first, self.operator)
        return op(*rest)

    def __str__(self):
        ops = utils.argsToString(self.operands)
        return f'{self.obj}.{self.operator}{ops}'


@dataclass(frozen=True, eq=False)
class VectorMethodDistribution(VectorDistribution):
    """Vector version of MethodDistribution."""
    method: Callable
    obj: Vector3D
    args: Tuple
    kwargs: Mapping

    def sample_given_dependencies(self, dep_values):
        args = [dep_values[arg] for arg in self.args]
        kwargs = {name: dep_values[arg] for name, arg in self.kwargs.items()}
        samp = self.method(self.obj, *args, **kwargs)
        return samp

    def dependencies(self) -> List:
        return [x for x in (self.obj, *self.args, *self.kwargs.values()) if needs_sampling(x) or needs_lazy_evaluation(x)]

    def __str__(self):
        args = utils.argsToString(itertools.chain(self.args, self.kwargs.values()))
        return f'{self.obj}.{self.method.__name__}{args}'


def scalarOperator(method):
    """Decorator for vector operators that yield scalars."""
    op = method.__name__
    setattr(VectorDistribution, op, makeOperatorHandler(op))

    @functools.wraps(method)
    def handler2(self, *args, **kwargs):
        if any(needs_sampling(arg) for arg in itertools.chain(args, kwargs.values())):
            return MethodDistribution(method, self, args, kwargs)
        else:
            return method(self, *args, **kwargs)

    return handler2


def makeVectorOperatorHandler(op):
    def handler(self, *args):
        return VectorOperatorDistribution(op, self, args)

    return handler


def vectorOperator(method):
    """Decorator for vector operators that yield vectors."""
    op = method.__name__
    setattr(VectorDistribution, op, makeVectorOperatorHandler(op))

    @functools.wraps(method)
    def handler2(self, *args):
        if needs_sampling(self):
            return VectorOperatorDistribution(op, self, args)
        elif any(needs_sampling(arg) for arg in args):
            return VectorMethodDistribution(method, self, args, {})
        elif any(needs_lazy_evaluation(arg) for arg in args):
            # see analogous comment in distributionFunction
            return makeDelayedFunctionCall(handler2, args, {})
        else:
            return method(self, *args)

    return handler2


def vectorDistributionMethod(method):
    """Decorator for methods that produce vectors. See distributionMethod."""

    @functools.wraps(method)
    def helper(self, *args, **kwargs):
        from probRobScene.core.sampling import needs_sampling
        if any(needs_sampling(arg) for arg in itertools.chain(args, kwargs.values())):
            return VectorMethodDistribution(method, self, args, kwargs)
        elif any(needs_lazy_evaluation(arg) for arg in itertools.chain(args, kwargs.values())):
            # see analogous comment in distributionFunction
            return makeDelayedFunctionCall(helper, (self,) + args, kwargs)
        else:
            return method(self, *args, **kwargs)

    return helper


@dataclass(frozen=True, eq=False)
class Vector(Samplable, collections.abc.Sequence):
    """A 2D vector, whose coordinates can be distributions."""
    x: float
    y: float

    @property
    def coordinates(self):
        return [self.x, self.y]

    @vectorOperator
    def rotatedBy(self, angle):
        """Return a vector equal to this one rotated counterclockwise by the given angle."""
        x, y = self.x, self.y
        c, s = cos(angle), sin(angle)
        return Vector((c * x) - (s * y), (s * x) + (c * y))

    @vectorOperator
    def offsetRotated(self, heading, offset):
        ro = offset.rotatedBy(heading)
        return self + ro

    @vectorOperator
    def offsetRadially(self, radius, heading):
        return self.offsetRotated(heading, Vector(0, radius))

    @scalarOperator
    def distanceTo(self, other):
        dx, dy = other.toVector() - self
        return math.hypot(dx, dy)

    @scalarOperator
    def angleTo(self, other):
        dx, dy = other.toVector() - self
        return normalize_angle(math.atan2(dy, dx) - (math.pi / 2))

    @vectorOperator
    def __add__(self, other):
        return Vector(self[0] + other[0], self[1] + other[1])

    @vectorOperator
    def __radd__(self, other):
        return Vector(self[0] + other[0], self[1] + other[1])

    @vectorOperator
    def __sub__(self, other):
        return Vector(self[0] - other[0], self[1] - other[1])

    @vectorOperator
    def __rsub__(self, other):
        return Vector(other[0] - self[0], other[1] - self[1])

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]


@dataclass(frozen=True, eq=False)
class Vector3D(Samplable, Sequence):
    x: float
    y: float
    z: float

    def to_vector_3d(self):
        return self

    @property
    def coordinates(self):
        return [self.x, self.y, self.z]

    @scalarOperator
    def distanceTo(self, other):
        dx, dy, dz = other.to_vector_3d() - self
        math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    @vectorOperator
    def __add__(self, other):
        return Vector3D(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    @vectorOperator
    def __sub__(self, other):
        return Vector3D(self[0] - other[0], self[1] - other[1], self[2] - other[2])

    @vectorOperator
    def __rsub__(self, other):
        return Vector3D(other[0] - self[0], other[1] - self[1], other[2] - self[2])

    @vectorOperator
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3D(self[0] * other, self[1] * other, self[2] * other)
        raise ValueError(
            f"Multiplication of Vector3D by {type(other)} --- Only multiplication by real scalars is currently supported")

    @vectorOperator
    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3D(self[0] * other, self[1] * other, self[2] * other)
        raise ValueError(
            f"Multiplication of Vector3D by {type(other)} --- Only multiplication by real scalars is currently supported")

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, index):
        return self.coordinates[index]


@distributionFunction
def rotation_to_vec(from_vec: Vector3D, to_vec: Vector3D) -> R:
    from_vec = from_vec / np.linalg.norm(from_vec)
    to_vec = to_vec / np.linalg.norm(to_vec)

    angle_rad = np.arccos(np.dot(from_vec, to_vec))
    cross_prod = np.cross(from_vec, to_vec)

    if np.linalg.norm(cross_prod) == 0.0:
        arbitrary_vec = np.array([from_vec[1], -from_vec[2], from_vec[0]])
        arb_cross = np.cross(from_vec, arbitrary_vec)
        rot_vec = R.from_rotvec(arb_cross / np.linalg.norm(arb_cross) * angle_rad)
    else:
        rot_vec = R.from_rotvec(cross_prod / np.linalg.norm(cross_prod) * angle_rad)

    return rot_vec


@distributionFunction
def rotation_to_euler(from_vec: Vector3D, to_vec: Vector3D) -> Vector3D:
    rot_vec = rotation_to_vec(from_vec, to_vec)
    return Vector3D(*rot_vec.as_euler('zyx'))


@distributionFunction
def offset_beyond(origin: Vector3D, offset: Vector3D, from_perspective_pos: Vector3D) -> Vector3D:
    diff = origin - from_perspective_pos
    assert np.linalg.norm(
        diff) > 0.0, "Origin and perspective cannot be the same. Perhaps you just want offset specifier?"
    rot_vec = rotation_to_vec(Vector3D(1.0, 0.0, 0.0), diff)

    rotated_offset = rot_vec.apply(offset)
    return origin + rotated_offset


@distributionFunction
def rotate_euler_v3d(vec: Vector3D, euler_rot: Vector3D) -> Vector3D:
    return Vector3D(*rotate_euler(vec, euler_rot))


def rotate_euler(v: np.array, euler_rot: np.array) -> np.array:
    rot = R.from_euler('zyx', euler_rot)
    return rot.apply(v)


@distributionFunction
def reverse_euler(euler_rot: Vector3D) -> Vector3D:
    rot = R.from_euler('zyx', euler_rot)
    inv = rot.inv()
    return inv.as_euler('zyx')


def normalize_angle(angle):
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    assert -math.pi <= angle <= math.pi
    return angle