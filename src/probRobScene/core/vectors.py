"""Scenic vectors and vector fields."""

import collections
import functools
import itertools
import math
from math import sin, cos

import numpy as np
from scipy.spatial.transform import Rotation as R

import probRobScene.core.utils as utils
from probRobScene.core.distributions import (Samplable, Distribution, MethodDistribution,
                                             needs_sampling, makeOperatorHandler, distributionMethod, distributionFunction)
from probRobScene.core.geometry import normalize_angle
from probRobScene.core.lazy_eval import value_in_context, needs_lazy_evaluation, makeDelayedFunctionCall


class VectorDistribution(Distribution):
    """A distribution over Vectors."""
    defaultValueType = None  # will be set after Vector is defined

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
        super().__init__(obj, *operands)
        self.operator = operator
        self.object = obj
        self.operands = operands

    def sample_given_dependencies(self, dep_values):
        first = dep_values[self.object]
        rest = (dep_values[child] for child in self.operands)
        op = getattr(first, self.operator)
        return op(*rest)

    def evaluateInner(self, context):
        obj = value_in_context(self.object, context)
        operands = tuple(value_in_context(arg, context) for arg in self.operands)
        return VectorOperatorDistribution(self.operator, obj, operands)

    def __str__(self):
        ops = utils.argsToString(self.operands)
        return f'{self.object}.{self.operator}{ops}'


class VectorMethodDistribution(VectorDistribution):
    """Vector version of MethodDistribution."""

    def __init__(self, method, obj, args, kwargs):
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
        return VectorMethodDistribution(self.method, obj, arguments, kwargs)

    def __str__(self):
        args = utils.argsToString(itertools.chain(self.arguments, self.kwargs.values()))
        return f'{self.object}.{self.method.__name__}{args}'


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
        if any(needs_sampling(arg) for arg in itertools.chain(args, kwargs.values())):
            return VectorMethodDistribution(method, self, args, kwargs)
        elif any(needs_lazy_evaluation(arg) for arg in itertools.chain(args, kwargs.values())):
            # see analogous comment in distributionFunction
            return makeDelayedFunctionCall(helper, (self,) + args, kwargs)
        else:
            return method(self, *args, **kwargs)

    return helper


class Vector(Samplable, collections.abc.Sequence):
    """A 2D vector, whose coordinates can be distributions."""

    def __init__(self, x, y):
        self.coordinates = (x, y)
        super().__init__(self.coordinates)

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    def toVector(self):
        return self

    def sample_given_dependencies(self, dep_values):
        return Vector(*(dep_values[coord] for coord in self.coordinates))

    def evaluateInner(self, context):
        return Vector(*(value_in_context(coord, context) for coord in self.coordinates))

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

    def __repr__(self):
        return f'({self.x} @ {self.y})'

    def __eq__(self, other):
        if type(other) is not Vector:
            return NotImplemented
        return other.coordinates == self.coordinates

    def __hash__(self):
        return hash(self.coordinates)


VectorDistribution.defaultValueType = Vector


class Vector3D(Samplable, collections.abc.Sequence):
    """A 3D Vector, whose coordinates can be distributions"""

    def __init__(self, x, y, z):
        self.coordinates = (x, y, z)
        super().__init__(self.coordinates)

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]

    @property
    def z(self):
        return self.coordinates[2]

    def to_vector_3d(self):
        return self

    def sample_given_dependencies(self, dep_values):
        return Vector3D(*(dep_values[coord] for coord in self.coordinates))

    def evaluateInner(self, context):
        return Vector3D(*(value_in_context(coord, context) for coord in self.coordinates))

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

    def __repr__(self):
        return f'({self.x} @ {self.y} @ {self.z})'

    def __eq__(self, other):
        if type(other) is not Vector3D:
            return False
        return other.coordinates == self.coordinates

    def __hash__(self):
        return hash(self.coordinates)


class OrientedVector(Vector):
    def __init__(self, x, y, heading):
        super().__init__(x, y)
        self.heading = heading

    def toHeading(self):
        return self.heading

    def __eq__(self, other):
        if type(other) is not OrientedVector:
            return NotImplemented
        return (other.coordinates == self.coordinates
                and other.heading == self.heading)

    def __hash__(self):
        return hash((self.coordinates, self.heading))


class VectorField:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.valueType = float

    @distributionMethod
    def __getitem__(self, pos):
        return self.value(pos)

    @vectorDistributionMethod
    def followFrom(self, pos, dist, steps=4):
        step = dist / steps
        for i in range(steps):
            pos = pos.offsetRadially(step, self[pos])
        return pos

    def __str__(self):
        return f'<{type(self).__name__} {self.name}>'


class VectorField3D:
    def __init__(self, name, value):
        self.name = name
        self.value = value
        self.valueType = float

    @distributionMethod
    def __getitem__(self, pos):
        return self.value(pos)

    @vectorDistributionMethod
    def follow_from(self, pos, dist, steps=4):
        step_size = dist / steps
        followed_pos = pos
        for i in range(steps):
            followed_pos += rotate_euler_v3d(Vector3D(step_size, 0.0, 0.0), self[pos])
        return followed_pos

    def __str__(self):
        return f'<{type(self).__name__} {self.name}>'


class PolygonalVectorField(VectorField):
    def __init__(self, name, cells, headingFunction=None, defaultHeading=None):
        self.cells = tuple(cells)
        if headingFunction is None and defaultHeading is not None:
            headingFunction = lambda pos: defaultHeading
        self.headingFunction = headingFunction
        for cell, heading in self.cells:
            if heading is None and headingFunction is None and defaultHeading is None:
                raise RuntimeError(f'missing heading for cell of PolygonalVectorField')
        self.defaultHeading = defaultHeading
        super().__init__(name, self.valueAt)

    def valueAt(self, pos):
        point = shapely.geometry.Point(pos)
        for cell, heading in self.cells:
            if cell.intersects(point):
                return self.headingFunction(pos) if heading is None else heading
        if self.defaultHeading is not None:
            return self.defaultHeading
        raise RuntimeError(f'evaluated PolygonalVectorField at undefined point {pos}')


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
