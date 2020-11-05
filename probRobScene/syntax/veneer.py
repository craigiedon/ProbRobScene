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

activity = 0
evaluatingRequirement = False
allObjects = []  # ordered for reproducibility
globalParameters = {}
externalParameters = []  # ordered for reproducibility
pendingRequirements = {}
inheritedReqs = []  # TODO improve handling of these?


def isActive():
    """Are we in the middle of compiling a Scenic module?

    The 'activity' global can be >1 when Scenic modules in turn import other
    Scenic modules."""
    return activity > 0


def activate():
    """Activate the veneer when beginning to compile a Scenic module."""
    global activity
    activity += 1
    assert not evaluatingRequirement


def deactivate():
    """Deactivate the veneer after compiling a Scenic module."""
    global activity, allObjects, globalParameters, externalParameters
    global pendingRequirements, inheritedReqs
    activity -= 1
    assert activity >= 0
    assert not evaluatingRequirement
    allObjects = []
    globalParameters = {}
    externalParameters = []
    pendingRequirements = {}
    inheritedReqs = []


def registerObject(obj):
    """Add a Scenic object to the global list of created objects.

    This is called by the Object constructor."""
    if activity > 0:
        assert not evaluatingRequirement
        assert isinstance(obj, Constructible)
        allObjects.append(obj)
    elif evaluatingRequirement:
        raise RuntimeParseError('tried to create an object inside a requirement')


def registerExternalParameter(value):
    """Register a parameter whose value is given by an external sampler."""
    if activity > 0:
        assert isinstance(value, ExternalParameter)
        externalParameters.append(value)


### Primitive statements and functions

def require(reqID, req, line, prob=1):
    """Function implementing the require statement."""
    if evaluatingRequirement:
        raise RuntimeParseError('tried to create a requirement inside a requirement')
    # the translator wrapped the requirement in a lambda to prevent evaluation,
    # so we need to save the current values of all referenced names; throw in
    # the ego object too since it can be referred to implicitly
    assert reqID not in pendingRequirements
    pendingRequirements[reqID] = (req, getAllGlobals(req), line, prob)


def getAllGlobals(req, restrictTo=None):
    """Find all names the given lambda depends on, along with their current bindings."""
    namespace = req.__globals__
    if restrictTo is not None and restrictTo is not namespace:
        return {}
    externals = inspect.getclosurevars(req)
    assert not externals.nonlocals  # TODO handle these
    globs = dict(externals.builtins)
    for name, value in externals.globals.items():
        globs[name] = value
        if inspect.isfunction(value):
            subglobs = getAllGlobals(value, restrictTo=namespace)
            for n, v in subglobs.items():
                if n in globs:
                    assert v is globs[n]
                else:
                    globs[n] = v
    return globs


def resample(dist):
    """The built-in resample function."""
    return dist.clone() if isinstance(dist, Distribution) else dist


def verbosePrint(msg, verbosity):
    """Built-in function printing a message when the verbosity is >0."""
    if verbosity >= 1:
        indent = '  ' * activity if verbosity >= 2 else '  '
        print(indent + msg)


def param(*quotedParams, **params):
    """Function implementing the param statement."""
    if evaluatingRequirement:
        raise RuntimeParseError('tried to create a global parameter inside a requirement')
    for name, value in params.items():
        globalParameters[name] = to_distribution(value)
    assert len(quotedParams) % 2 == 0, quotedParams
    it = iter(quotedParams)
    for name, value in zip(it, it):
        globalParameters[name] = to_distribution(value)


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

def FieldAt(X, Y):
    """The '<VectorField> at <vector>' operator."""
    if not isinstance(X, VectorField):
        raise RuntimeParseError('"X at Y" with X not a vector field')
    Y = toVector(Y, '"X at Y" with Y not a vector')
    return X[Y]


def RelativeTo(X, Y):
    """The 'X relative to Y' polymorphic operator.

    Allowed forms:
        F relative to G (with at least one a field, the other a field or heading)
        <vector> relative to <oriented point> (and vice versa)
        <vector> relative to <vector>
        <heading> relative to <heading>
    """
    xf, yf = isA(X, VectorField), isA(Y, VectorField)
    if xf or yf:
        if xf and yf and X.valueType != Y.valueType:
            raise RuntimeParseError('"X relative to Y" with X, Y fields of different types')
        field_type = X.valueType if xf else Y.valueType
        error = '"X relative to Y" with field and value of different types'

        def helper(context):
            pos = context.position.toVector()
            xp = X[pos] if xf else toType(X, field_type, error)
            yp = Y[pos] if yf else toType(Y, field_type, error)
            return xp + yp

        return DelayedArgument({'position'}, helper)
    else:
        if isinstance(X, OrientedPoint):  # TODO too strict?
            if isinstance(Y, OrientedPoint):
                raise RuntimeParseError('"X relative to Y" with X, Y both oriented points')
            Y = toVector(Y, '"X relative to Y" with X an oriented point but Y not a vector')
            return X.relativize(Y)
        elif isinstance(Y, OrientedPoint):
            X = toVector(X, '"X relative to Y" with Y an oriented point but X not a vector')
            return Y.relativize(X)
        else:
            X = toTypes(X, (Vector, float), '"X relative to Y" with X neither a vector nor scalar')
            Y = toTypes(Y, (Vector, float), '"X relative to Y" with Y neither a vector nor scalar')
            return evaluateRequiringEqualTypes(lambda: X + Y, X, Y,
                                               '"X relative to Y" with vector and scalar')


def RelativeTo3D(X, Y):
    X = toTypes(X, (Vector3D, float))
    Y = toTypes(Y, (Vector3D, float))
    return evaluateRequiringEqualTypes(lambda: X + Y, X, Y, "X relative to Y with different types")


def OffsetAlong(X, H, Y):
    """The 'X offset along H by Y' polymorphic operator.

    Allowed forms:
        <vector> offset along <heading> by <vector>
        <vector> offset along <field> by <vector>
    """
    X = toVector(X, '"X offset along H by Y" with X not a vector')
    Y = toVector(Y, '"X offset along H by Y" with Y not a vector')
    if isinstance(H, VectorField):
        H = H[X]
    H = toHeading(H, '"X offset along H by Y" with H not a heading or vector field')
    return X.offsetRotated(H, Y)


def RelativePosition(X, Y):
    """The 'relative position of <vector> [from <vector>]' operator."""
    X = toVector(X, '"relative position of X from Y" with X not a vector')
    Y = toVector(Y, '"relative position of X from Y" with Y not a vector')
    return X - Y


def RelativeHeading(X, Y):
    """The 'relative heading of <heading> [from <heading>]' operator.
    """
    X = toHeading(X, '"relative heading of X from Y" with X not a heading')
    Y = toHeading(Y, '"relative heading of X from Y" with Y not a heading')
    return normalize_angle(X - Y)


def ApparentHeading(X, Y):
    """The 'apparent heading of <oriented point> [from <vector>]' operator."""
    if not isinstance(X, OrientedPoint):
        raise RuntimeParseError('"apparent heading of X from Y" with X not an OrientedPoint')
    Y = toVector(Y, '"relative heading of X from Y" with Y not a vector')
    return apparentHeadingAtPoint(X.position, X.heading, Y)


def DistanceFrom(X, Y):
    """The 'distance from <vector> [to <vector>]' operator."""
    X = toVector(X, '"distance from X to Y" with X not a vector')
    Y = toVector(Y, '"distance from X to Y" with Y not a vector')
    return X.distanceTo(Y)


def CanSee(X, Y):
    """The 'X can see Y' polymorphic operator.

    Allowed forms:
        <point> can see <object>
        <point> can see <vector>
    """
    if not isinstance(X, Point):
        raise RuntimeParseError('"X can see Y" with X not a Point')
    if isinstance(Y, Point):
        return X.canSee(Y)
    else:
        Y = toVector(Y, '"X can see Y" with Y not a vector')
        return X.visibleRegion.contains_point(Y)


### Specifiers

def With(prop, val):
    """The 'with <property> <value>' specifier.

    Specifies the given property, with no dependencies.
    """
    return Specifier(prop, val)


def At(pos):
    """The 'at <vector>' specifier.

    Specifies 'position', with no dependencies."""
    pos = toVector(pos, 'specifier "at X" with X not a vector')
    return Specifier('position', pos)


def At3D(pos):
    """
    Specifies the 3d position with no dependencies
    """

    pos = toType(pos, Vector3D)
    return Specifier('position', pos)


def In(region):
    """The 'in/on <region>' specifier.

    Specifies 'position', with no dependencies. Optionally specifies 'heading'
    if the given Region has a preferred orientation.
    """
    region = toType(region, Region, 'specifier "in/on R" with R not a Region')
    extras = {'heading'} if alwaysProvidesOrientation(region) else {}
    return Specifier('position', PointInRegionDistribution(region), optionals=extras)


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


def Beyond(pos, offset, fromPt):
    """The 'beyond X by Y [from Z]' polymorphic specifier.

    Specifies 'position', with no dependencies.

    Allowed forms:
        beyond <vector> by <number> [from <vector>]
        beyond <vector> by <vector> [from <vector>]
    """
    pos = toVector(pos, 'specifier "beyond X by Y" with X not a vector')
    dType = underlyingType(offset)
    if dType is float or dType is int:
        offset = Vector(0, offset)
    elif dType is not Vector:
        raise RuntimeParseError('specifier "beyond X by Y" with Y not a number or vector')
    fromPt = toVector(fromPt, 'specifier "beyond X by Y from Z" with Z not a vector')
    lineOfSight = fromPt.angleTo(pos)
    return Specifier('position', pos.offsetRotated(lineOfSight, offset))


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


def VisibleFrom(base):
    """The 'visible from <Point>' specifier.

    Specifies 'position', with no dependencies.

    This uses the given object's 'visibleRegion' property, and so correctly
    handles the view regions of Points, OrientedPoints, and Objects.
    """
    if not isinstance(base, Point):
        raise RuntimeParseError('specifier "visible from O" with O not a Point')
    return Specifier('position', PointInRegionDistribution(base.visibleRegion))


def OffsetBy3D(offset):
    offset = toType(offset, Vector3D)
    pos = RelativeTo3D(offset, ego()).to_vector_3d()
    return Specifier('position', pos)


def OffsetAlongSpec(direction, offset):
    """The 'offset along X by Y' polymorphic specifier.

    Specifies 'position', with no dependencies.

    Allowed forms:
        offset along <heading> by <vector>
        offset along <field> by <vector>
    """
    return Specifier('position', OffsetAlong(ego(), direction, offset))


def Facing(heading):
    """The 'facing X' polymorphic specifier.

    Specifies 'heading', with dependencies depending on the form:
        facing <number> -- no dependencies;
        facing <field> -- depends on 'position'.
    """
    if isinstance(heading, VectorField):
        return Specifier('heading', DelayedArgument({'position'},
                                                    lambda self: heading[self.position]))
    else:
        heading = toHeading(heading, 'specifier "facing X" with X not a heading or vector field')
        return Specifier('heading', heading)


def Facing3D(direction: Vector3D) -> Specifier:
    orientation = rotation_to_euler(Vector3D(0, 1, 0), direction)
    return Specifier('orientation', orientation)


def FacingToward(pos):
    """The 'facing toward <vector>' specifier.

    Specifies 'heading', depending on 'position'.
    """
    pos = toVector(pos, 'specifier "facing toward X" with X not a vector')
    return Specifier('heading', DelayedArgument({'position'},
                                                lambda self: self.position.angleTo(pos)))


def FacingToward3D(pos):
    pos = toType(pos, Vector3D)
    return Specifier('orientation', DelayedArgument({'position'}, lambda s: rotation_to_euler(Vector3D(0, 1, 0), Vector3D(pos[0], pos[1], s.position[2]) - s.position)))


def ApparentlyFacing(heading, fromPt):
    """The 'apparently facing <heading> [from <vector>]' specifier.

    Specifies 'heading', depending on 'position'.
    """
    heading = toHeading(heading, 'specifier "apparently facing X" with X not a heading')
    fromPt = toVector(fromPt, 'specifier "apparently facing X from Y" with Y not a vector')
    value = lambda self: fromPt.angleTo(self.position) + heading
    return Specifier('heading', DelayedArgument({'position'}, value))


def LeftSpec(pos, dist=0):
    return left_spec_helper('left of', pos, dist, 'width', lambda dist: (dist, 0),
                            lambda self, dx, dy: Vector(-self.width / 2 - dx, dy))


def RightSpec(pos, dist=0):
    return left_spec_helper('right of', pos, dist, 'width', lambda dist: (dist, 0),
                            lambda self, dx, dy: Vector(self.width / 2 + dx, dy))


def Ahead(pos, dist=0):
    return left_spec_helper('ahead of', pos, dist, 'height', lambda dist: (0, dist),
                            lambda self, dx, dy: Vector(dx, self.height / 2.0 + dy))


def Behind(pos, dist=0):
    return left_spec_helper('behind', pos, dist, 'height', lambda dist: (0, dist),
                            lambda self, dx, dy: Vector(dx, -self.height / 2.0 - dy))


eps = 1e-9


# def LeftSpec3D(pos, dist=eps) -> Specifier:
#     return directional_spec_helper(syntax='left of',
#                                    pos=pos,
#                                    dist=dist,
#                                    axis='width',
#                                    to_components=lambda d: (d, 0, 0),
#                                    make_offset=lambda self, dx, dy, dz: Vector3D(-self.width / 2.0 - dx, dy, dz))


# def RightSpec3D(pos, dist=eps) -> Specifier:
#     return directional_spec_helper('right of', pos, dist, 'width', lambda dist: (dist, 0, 0),
#                                    lambda self, dx, dy, dz: Vector3D(self.width / 2.0 + dx, dy, dz))


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


def left_spec_helper(syntax, pos, dist, axis, to_components, make_offset):
    raise NotImplementedError


def Following(field, dist, fromPt):
    """The 'following F [from X] for D' specifier.

    Specifies 'position', and optionally 'heading', with no dependencies.

    Allowed forms:
        following <field> [from <vector>] for <number>
    """
    dist, fromPt = fromPt, dist
    if not isinstance(field, VectorField):
        raise RuntimeParseError('"following F" specifier with F not a vector field')
    fromPt = toVector(fromPt, '"following F from X for D" with X not a vector')
    dist = toScalar(dist, '"following F for D" with D not a number')
    pos = field.followFrom(fromPt, dist)
    heading = field[pos]
    val = OrientedPoint(position=pos, heading=heading)
    return Specifier('position', val, optionals={'heading'})


def Following3D(field: VectorField3D, dist: float, from_pt):
    assert isinstance(field, VectorField3D)

    from_pt = toType(from_pt, Vector3D)
    dist = float(dist)

    pos = field.follow_from(from_pt, dist)
    orientation = field[pos]
    val = OrientedPoint3D(position=pos, orientation=orientation)

    return Specifier('position', val, optionals={'orientation'})
