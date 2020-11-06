"""Python implementations of Scenic language constructs.

This module is automatically imported by all Scenic programs. In addition to
defining the built-in functions, operators, specifiers, etc., it also stores
global state such as the list of all created Scenic objects.
"""

from typing import Union

from probRobScene.core.distributions import Range, Options, Normal, distributionFunction
# various Python types and functions used in the language but defined elsewhere
from probRobScene.core.geometry import sin, cos, hypot, max, min
from probRobScene.core.regions import (Region, PointSetRegion,
                                       everywhere, nowhere, CuboidRegion, SphericalRegion,
                                       HalfSpaceRegion, PointInRegionDistribution, Rectangle3DRegion)
from probRobScene.core.vectors import Vector, VectorField, PolygonalVectorField, Vector3D, offset_beyond, \
    rotation_to_euler, rotate_euler_v3d, VectorField3D

Uniform = lambda *opts: Options(opts)  # TODO separate these?
Discrete = Options
from probRobScene.core.external_params import (VerifaiParameter, VerifaiRange, VerifaiDiscreteRange, VerifaiOptions)
from probRobScene.core.object_types import Mutator, Object, Point3D, OrientedPoint3D, Oriented
from probRobScene.core.specifiers import PropertyDefault  # TODO remove

# everything that should not be directly accessible from the language is imported here:
import inspect
from probRobScene.core.distributions import Distribution, to_distribution
from probRobScene.core.type_support import isA, toType, toTypes, toScalar, toHeading, toVector
from probRobScene.core.type_support import evaluateRequiringEqualTypes, underlyingType
from probRobScene.core.geometry import normalize_angle, apparentHeadingAtPoint
from probRobScene.core.object_types import Constructible
from probRobScene.core.specifiers import Specifier
from probRobScene.core.lazy_eval import DelayedArgument
from probRobScene.core.utils import RuntimeParseError
from probRobScene.core.external_params import ExternalParameter


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

    pos = toType(pos, Vector3D)
    return Specifier('position', pos)


def In3D(region):
    region = toType(region, Region, 'specifier "in R" with R not a Region')
    return Specifier('position', PointInRegionDistribution(region))


def alwaysProvidesOrientation(region):
    """Whether a Region or distribution over Regions always provides an orientation."""
    if isinstance(region, Region):
        return region.orientation is not None
    elif isinstance(region, Options):
        return all(alwaysProvidesOrientation(opt) for opt in region.options)
    else:
        return False


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


def OffsetBy3D(offset):
    offset = toType(offset, Vector3D)
    pos = RelativeTo3D(offset, ego()).to_vector_3d()
    return Specifier('position', pos)


def Facing3D(direction: Vector3D) -> Specifier:
    orientation = rotation_to_euler(Vector3D(0, 1, 0), direction)
    return Specifier('orientation', orientation)


def FacingToward3D(pos):
    pos = toType(pos, Vector3D)
    return Specifier('orientation', DelayedArgument({'position'}, lambda s: rotation_to_euler(Vector3D(0, 1, 0), Vector3D(pos[0], pos[1], s.position[2]) - s.position)))


eps = 1e-9


def Ahead3D(pos, dist=eps) -> Specifier:
    return directional_spec_helper('ahead of', pos, dist, 'length', lambda dist: (0, dist, 0),
                                   lambda self, dx, dy, dz: Vector3D(dx, self.length / 2.0 + dy, dz))


def Behind3D(pos, dist=eps) -> Specifier:
    return directional_spec_helper('behind', pos, dist, 'length', lambda dist: (0, dist, 0),
                                   lambda self, dx, dy, dz: Vector3D(dx, -self.length / 2.0 - dy, dz))


def Above3D(pos, dist=eps) -> Specifier:
    return directional_spec_helper('above', pos, dist, 'height', lambda dist: (0, 0, dist),
                                   lambda self, dx, dy, dz: Vector3D(dx, dy, self.height / 2.0 + dz))


def Below3D(pos, dist=eps) -> Specifier:
    return directional_spec_helper('below', pos, dist, 'height', lambda dist: (0, 0, dist),
                                   lambda self, dx, dy, dz: Vector3D(dx, dy, -self.height / 2.0 - dz))


def directional_spec_helper(syntax, pos, dist, axis, to_components, make_offset) -> Specifier:
    extras = set()
    d_type = underlyingType(dist)
    if d_type is float or d_type is int:
        offset_vec = to_components(dist)
    elif d_type is Vector3D:
        offset_vec = dist
    else:
        raise RuntimeParseError(f'"{syntax} X by D" with D not a number or vector3d')

    if isinstance(pos, Oriented):
        val = lambda self: pos.position + rotate_euler_v3d(make_offset(self, *offset_vec), pos.orientation)
        new = DelayedArgument({axis}, val)
    else:
        pos = toType(pos, Vector3D)
        val = lambda self: pos + make_offset(self, *offset_vec)
        new = DelayedArgument({axis, 'orientation'}, val)
    return Specifier('position', new, optionals=extras)


def OnTopOf(thing: Union[Point3D, Vector3D, Object, Rectangle3DRegion], dist: float = eps, strict: bool = False) -> Specifier:
    if isinstance(thing, Rectangle3DRegion):
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


def OnTopOfStrict(thing: Union[Point3D, Vector3D, Object, Rectangle3DRegion], dist=eps) -> Specifier:
    return OnTopOf(thing, dist, True)


def LeftRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    if isinstance(obj, Object):
        new = PointInRegionDistribution(left_plane(obj, min_amount, max_amount))
    elif isinstance(obj, Point3D):
        new = PointInRegionDistribution(HalfSpaceRegion(obj.position + Vector3D(-min_amount, 0, 0), Vector3D(-1, 0, 0), max_amount - min_amount))
    else:
        raise TypeError(f'Object {obj} of type {type(obj)} not supported by Ahead. Only supports Point3D and Object types')

    return Specifier('position', new)


def RightRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0) -> Specifier:
    if isinstance(obj, Object):
        new = PointInRegionDistribution(right_plane(obj, min_amount, max_amount))
    elif isinstance(obj, Point3D):
        new = PointInRegionDistribution(HalfSpaceRegion(obj.position + Vector3D(min_amount, 0, 0), Vector3D(1, 0, 0), max_amount - min_amount))
    else:
        raise TypeError(f'Object {obj} of type {type(obj)} not supported by Ahead. Only supports Point3D and Object types')

    return Specifier('position', new)


def AheadRough(obj: Union[Object, Point3D], min_amount: float = 0.0, max_amount: float = 1000.0):
    if isinstance(obj, Object):
        new = DelayedArgument({'length'}, lambda s: PointInRegionDistribution(front_plane(s, obj, min_amount, max_amount)))
    elif isinstance(obj, Point3D):
        new = DelayedArgument({'length'}, lambda s: PointInRegionDistribution(HalfSpaceRegion(obj.position + Vector3D(0, s.length / 2.0 + min_amount, 0), Vector3D(0, 1, 0), max_amount - min_amount)))
    else:
        raise TypeError(f'Object {obj} of type {type(obj)} not supported by Ahead. Only supports Point3D and Object types')

    return Specifier('position', new)


def left_plane(ref_obj: Object, min_amount: float, max_amount: float) -> HalfSpaceRegion:
    point = ref_obj.position + rotate_euler_v3d(Vector3D(-ref_obj.width / 2.0 - min_amount, 0.0, 0.0), ref_obj.orientation)
    normal = rotate_euler_v3d(Vector3D(-1, 0, 0), ref_obj.orientation)
    return HalfSpaceRegion(point, normal, max_amount - min_amount)


def right_plane(ref_obj: Object, min_amount: float, max_amount: float) -> HalfSpaceRegion:
    point = ref_obj.position + rotate_euler_v3d(Vector3D(ref_obj.width / 2.0 + min_amount, 0.0, 0.0), ref_obj.orientation)
    normal = rotate_euler_v3d(Vector3D(1, 0, 0), ref_obj.orientation)
    return HalfSpaceRegion(point, normal, max_amount - min_amount)


def front_plane(obj_to_place: Object, ref_obj: Object, min_amount: float, max_amount: float) -> HalfSpaceRegion:
    offset_amount = obj_to_place.length / 2.0 + ref_obj.length / 2.0 + min_amount
    point = ref_obj.position + rotate_euler_v3d(Vector3D(0, offset_amount, 0.0), ref_obj.orientation)
    normal = rotate_euler_v3d(Vector3D(0, 1, 0), ref_obj.orientation)
    return HalfSpaceRegion(point, normal, max_amount - min_amount)


def on_top_of_rect(obj_to_place: Object, r: Rectangle3DRegion, dist: float, strict: bool = False) -> Rectangle3DRegion:
    offset = rotate_euler_v3d(Vector3D(0, 0, dist + obj_to_place.height / 2.0), r.rot)
    if strict:
        # assert r.width > obj_to_place.width and r.length > obj_to_place.length
        return Rectangle3DRegion(r.width - obj_to_place.width, r.length - obj_to_place.length, r.origin + offset, r.rot)

    return Rectangle3DRegion(r.width, r.length, r.origin + offset, r.rot)


def top_surface_region(obj_to_place: Object, ref_obj: Object, dist: float, strict: bool = False) -> Rectangle3DRegion:
    ref_top_surface = ref_obj.position + rotate_euler_v3d(Vector3D(0, 0, ref_obj.height / 2.0), ref_obj.orientation)
    rotated_offset = rotate_euler_v3d(Vector3D(0, 0, dist + obj_to_place.height / 2.0), ref_obj.orientation)
    region_pos = rotated_offset + ref_top_surface

    if strict:
        # assert ref_obj.width > obj_to_place.width and ref_obj.length > obj_to_place.length
        return Rectangle3DRegion(ref_obj.width - obj_to_place.width, ref_obj.length - obj_to_place.length, region_pos, ref_obj.orientation)
    return Rectangle3DRegion(ref_obj.width, ref_obj.length, region_pos, ref_obj.orientation)


def Following3D(field: VectorField3D, dist: float, from_pt):
    assert isinstance(field, VectorField3D)

    from_pt = toType(from_pt, Vector3D)
    dist = float(dist)

    pos = field.follow_from(from_pt, dist)
    orientation = field[pos]
    val = OrientedPoint3D(position=pos, orientation=orientation)

    return Specifier('position', val, optionals={'orientation'})
