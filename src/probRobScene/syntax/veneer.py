"""Python implementations of Scenic language constructs.

This module is automatically imported by all Scenic programs. In addition to
defining the built-in functions, operators, specifiers, etc., it also stores
global state such as the list of all created Scenic objects.
"""

from typing import Union, Callable, Any

from probRobScene.core.distributions import Range, Normal, to_distribution
# various Python types and functions used in the language but defined elsewhere
from probRobScene.core.geometry import sin, cos, hypot, max, min
from probRobScene.core.regions import (Region, PointSet,
                                       Cuboid, Spherical,
                                       HalfSpace, PointInRegionDistribution, Rectangle3D)
from probRobScene.core.type_support import toType, underlyingType, toVector, toTypes, evaluateRequiringEqualTypes
from probRobScene.core.vectors import Vector, Vector3D, offset_beyond, \
    rotation_to_euler, rotate_euler_v3d, normalize_angle

from probRobScene.core.object_types import Object, Point3D, OrientedPoint3D
from probRobScene.core.specifiers import PropertyDefault  # TODO remove

# everything that should not be directly accessible from the language is imported here:
import inspect
from probRobScene.core.distributions import Distribution
from probRobScene.core.geometry import apparentHeadingAtPoint
from probRobScene.core.object_types import Constructible
from probRobScene.core.specifiers import Specifier
from probRobScene.core.lazy_eval import DelayedArgument
from probRobScene.core.utils import RuntimeParseError
import numpy as np


### Internals

class VeneerState:
    globalParameters = {}
    pendingRequirements = {}


v_state = VeneerState()


def require(reqID, req, line, prob=1):
    """Function implementing the require statement."""
    # the translator wrapped the requirement in a lambda to prevent evaluation,
    # so we need to save the current values of all referenced names; throw in

    v_state.pendingRequirements[reqID] = (req, getAllGlobals(req), line, prob)


def param(*quotedParams, **params):
    """Function implementing the param statement."""
    for name, value in params.items():
        v_state.globalParameters[name] = to_distribution(value)
    it = iter(quotedParams)
    for name, value in zip(it, it):
        v_state.globalParameters[name] = to_distribution(value)


### Primitive statements and functions


def getAllGlobals(req, restrict_to=None):
    """Find all names the given lambda depends on, along with their current bindings."""
    namespace = req.__globals__
    if restrict_to is not None and restrict_to is not namespace:
        return {}
    externals = inspect.getclosurevars(req)
    assert not externals.nonlocals  # TODO handle these
    globs = dict(externals.builtins)
    for name, value in externals.globals.items():
        globs[name] = value
        if inspect.isfunction(value):
            subglobs = getAllGlobals(value, restrict_to=namespace)
            for n, v in subglobs.items():
                if n in globs:
                    assert v is globs[n]
                else:
                    globs[n] = v
    return globs


def resample(dist):
    """The built-in resample function."""
    return dist.clone() if isinstance(dist, Distribution) else dist


### Prefix operators


# front of <object>, etc.
ops = (
    'front', 'back', 'left', 'right',
    'front left', 'front right',
    'back left', 'back right', 'top', 'bottom', 'top front', 'top back'
)
template = '''\
def {function}(X):
    """The '{syntax} of <object>' operator."""
    if not isinstance(X, Object):
        raise RuntimeParseError('"{syntax} of X" with X not an Object')
    return X.{property}
'''
for op in ops:
    func = ''.join(word.capitalize() for word in op.split(' '))
    prop = '_'.join(word for word in op.split(' '))
    definition = template.format(function=func, syntax=op, property=prop)
    exec(definition)


### Infix operators

def RelativeTo3D(X, Y):
    X = toTypes(X, (Vector3D, float))
    Y = toTypes(Y, (Vector3D, float))
    return evaluateRequiringEqualTypes(lambda: X + Y, X, Y, "X relative to Y with different types")


def DistanceFrom(X, Y):
    """The 'distance from <vector> [to <vector>]' operator."""
    X = toVector(X, '"distance from X to Y" with X not a vector')
    Y = toVector(Y, '"distance from X to Y" with Y not a vector')
    return X.distanceTo(Y)


### Specifiers

def With(prop, val):
    """The 'with <property> <value>' specifier.

    Specifies the given property, with no dependencies.
    """
    return Specifier(prop, val)


def At3D(pos):
    """
    Specifies the 3d position with no dependencies
    """

    return Specifier('position', pos)


def In3D(region):
    # region = toType(region, Region, 'specifier "in R" with R not a Region')
    return Specifier('position', PointInRegionDistribution(region))


def Beyond3D(pos, offset, from_pt):
    pos = toType(pos, Vector3D)
    d_type = underlyingType(offset)
    if d_type is float or d_type is int:
        offset = Vector3D(offset, 0, 0)
    elif d_type is not Vector3D:
        raise RuntimeParseError('specifier "beyond X by Y from Z" with Z not a vector')
    from_pt = toType(from_pt, Vector3D)
    new_pos = offset_beyond(pos, offset, from_pt)
    return Specifier('position', new_pos)


# def OffsetBy3D(offset):
#     offset = toType(offset, Vector3D)
#     pos = RelativeTo3D(offset, ego()).to_vector_3d()
#     return Specifier('position', pos)


def Facing3D(direction: Vector3D) -> Specifier:
    orientation = rotation_to_euler(Vector3D(0, 1, 0), direction)
    return Specifier('orientation', orientation)


def FacingToward3D(pos):
    pos = toType(pos, Vector3D)
    return Specifier('orientation', DelayedArgument({'position'}, lambda s: rotation_to_euler(Vector3D(0, 1, 0), Vector3D(pos[0], pos[1], s.position[2]) - s.position)))


eps = 1e-9


def directional_spec_helper(syntax: str, pos: Union[Object, Vector3D, Point3D], dist, axis: str, to_components: Callable, make_offset: Callable) -> Specifier:
    d_type = underlyingType(dist)
    assert d_type in (float, int, Vector3D), f'"{syntax} X by D" with D {dist} : {d_type} not a number of Vector3D'

    if d_type in (float, int):
        offset_vec = to_components(dist)
    elif d_type is Vector3D:
        offset_vec = dist

    if isinstance(pos, Object):
        val = lambda self: pos.position + rotate_euler_v3d(make_offset(self, *offset_vec), pos.orientation)
        new = DelayedArgument({axis}, val)
        return Specifier('position', new)
    if isinstance(pos, (Vector3D, Point3D)):
        pos = toType(pos, Vector3D)
        val = lambda self: pos + make_offset(self, *offset_vec)
        new = DelayedArgument({axis}, val)
        return Specifier('position', new)


def OnTopOf(thing: Union[Point3D, Vector3D, Object, Rectangle3D], dist: float = eps, strict: bool = False) -> Specifier:
    if isinstance(thing, Rectangle3D):
        if strict:
            new = DelayedArgument({'width', 'length', 'height'}, lambda s: PointInRegionDistribution(on_top_of_rect(s, thing, dist, strict)))
        else:
            new = DelayedArgument({'height'}, lambda s: PointInRegionDistribution(on_top_of_rect(s, thing, dist)))
    elif isinstance(thing, Object):
        if strict:
            new = DelayedArgument({'width', 'length', 'height'}, lambda s: PointInRegionDistribution(top_surface_region(s, thing, dist, strict)))
        else:
            new = DelayedArgument({'height'}, lambda s: PointInRegionDistribution(top_surface_region(s, thing, dist)))
    elif isinstance(thing, Point3D):
        new = DelayedArgument({'height'}, lambda s: thing.position + Vector3D(0, 0, dist + s.height / 2.0))
    elif isinstance(thing, Vector3D):
        new = DelayedArgument({'height'}, lambda s: thing + Vector3D(0, 0, dist + s.height / 2.0))
    else:
        raise TypeError(f'Asking to be on top of {thing} which has unsupported type {type(thing)}')
    return Specifier('position', new)


def OnTopOfStrict(thing: Union[Point3D, Vector3D, Object, Rectangle3D], dist=eps) -> Specifier:
    return OnTopOf(thing, dist, True)


def AlignedWith(thing: Union[Point3D, Object], axis: str) -> Specifier:
    align_point = thing.position
    if axis == 'x':
        reg = Rectangle3D(100.0, 100.0, align_point, Vector3D(0.0, np.pi / 2.0, 0.0))
    elif axis == 'y':
        reg = Rectangle3D(100.0, 100.0, align_point, Vector3D(0.0, 0.0, np.pi / 2.0))
    elif axis == 'z':
        reg = Rectangle3D(100.0, 100.0, align_point, Vector3D(0.0, 0.0, 0.0))
    else:
        raise ValueError("Specified axis must be one of 'x', 'y', or 'z'")

    new = PointInRegionDistribution(reg)
    return Specifier('position', new)


def LeftRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    return direction_of_rough(obj, Vector3D(-1, 0, 0), 0, min_amount, max_amount)


def RightRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    return direction_of_rough(obj, Vector3D(1, 0, 0), 0, min_amount, max_amount)


def AheadRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    return direction_of_rough(obj, Vector3D(0, 1, 0), 1, min_amount, max_amount)


def BehindRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    return direction_of_rough(obj, Vector3D(0, -1, 0), 1, min_amount, max_amount)


def AboveRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    return direction_of_rough(obj, Vector3D(0, 0, 1), 2, min_amount, max_amount)


def BelowRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    return direction_of_rough(obj, Vector3D(0, 0, -1), 2, min_amount, max_amount)


def direction_of_rough(obj: Union[Object, Point3D],
                       offset_normal: Vector3D,
                       axis_index: int,
                       min_amount: float = 0.0,
                       max_amount: float = 1000.0) -> Specifier:
    assert isinstance(obj, (Object, Point3D, Vector3D)), f'Object {obj} of type {type(obj)} not supported by Ahead. Only supports Vector3D, Point3D and Object types'
    assert np.isclose(np.linalg.norm(offset_normal), 1.0), f'Offset normal {offset_normal} (magnitude: {np.linalg.norm(offset_normal)} does not have normalized magnitude'

    if isinstance(obj, Object):
        new = DelayedArgument({'length', 'width', 'height'}, lambda s: hs_from_obj(s, obj, offset_normal, axis_index, min_amount, max_amount))
    elif isinstance(obj, Point3D):
        new = DelayedArgument({'length', 'width', 'height'}, lambda s: hs_from_pos(s, obj.position, offset_normal, axis_index, min_amount, max_amount))
    elif isinstance(obj, Vector3D):
        new = DelayedArgument({'length', 'width', 'height'}, lambda s: hs_from_pos(s, obj, offset_normal, axis_index, min_amount, max_amount))

    return Specifier('position', new)


def hs_from_pos(s: Object, pos: Vector3D, offset_normal: Vector3D, axis_index: int, min_amount: float, max_amount: float) -> PointInRegionDistribution:
    dist: float = max_amount - min_amount
    origin: Vector3D = pos + (0.5 * s.dimensions[axis_index] + min_amount) * offset_normal
    return PointInRegionDistribution(HalfSpace(origin, offset_normal, dist))


def hs_from_obj(s: Object, ref_obj: Object, offset_normal: Vector3D, axis_index: int, min_amount: float, max_amount: float) -> PointInRegionDistribution:
    dist: float = max_amount - min_amount
    unrotated_offset: Vector3D = (0.5 * s.dimensions[axis_index] + min_amount) * offset_normal
    origin: Vector3D = ref_obj.position + rotate_euler_v3d(unrotated_offset, ref_obj.orientation)
    normal = rotate_euler_v3d(offset_normal, ref_obj.orientation)
    return PointInRegionDistribution(HalfSpace(origin, normal, max_amount - min_amount))


def on_top_of_rect(obj_to_place: Object, r: Rectangle3D, dist: float, strict: bool = False) -> Rectangle3D:
    offset = rotate_euler_v3d(Vector3D(0, 0, dist + obj_to_place.height / 2.0), r.rot)
    if strict:
        # assert r.width > obj_to_place.width and r.length > obj_to_place.length
        return Rectangle3D(r.width - obj_to_place.width, r.length - obj_to_place.length, r.origin + offset, r.rot)

    return Rectangle3D(r.width, r.length, r.origin + offset, r.rot)


def top_surface_region(obj_to_place: Object, ref_obj: Object, dist: float, strict: bool = False) -> Rectangle3D:
    ref_top_surface = ref_obj.position + rotate_euler_v3d(Vector3D(0, 0, ref_obj.height / 2.0), ref_obj.orientation)
    rotated_offset = rotate_euler_v3d(Vector3D(0, 0, dist + obj_to_place.height / 2.0), ref_obj.orientation)
    region_pos = rotated_offset + ref_top_surface

    if strict:
        # assert ref_obj.width > obj_to_place.width and ref_obj.length > obj_to_place.length
        return Rectangle3D(ref_obj.width - obj_to_place.width, ref_obj.length - obj_to_place.length, region_pos, ref_obj.orientation)
    return Rectangle3D(ref_obj.width, ref_obj.length, region_pos, ref_obj.orientation)
