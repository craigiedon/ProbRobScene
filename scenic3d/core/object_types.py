"""Implementations of the built-in Scenic classes."""
import abc
import collections
import inspect
import itertools as it
import math
import random
import numpy as np

from scenic3d.core.distributions import Samplable, needs_sampling, to_distribution
from scenic3d.core.geometry import averageVectors, hypot, min
from scenic3d.core.lazy_eval import needs_lazy_evaluation, makeDelayedFunctionCall
from scenic3d.core.regions import SphericalRegion, Oriented, intersect_many, PointInRegionDistribution, \
    IntersectionRegion
from scenic3d.core.specifiers import Specifier, PropertyDefault
from scenic3d.core.type_support import toVector, toScalar, toType
from scenic3d.core.utils import areEquivalent, RuntimeParseError, group_by
from scenic3d.core.vectors import Vector, Vector3D, rotate_euler_v3d
from scenic3d.core.plotUtil3d import draw_cube


## Abstract base class

class Constructible(Samplable):
    """Abstract base class for Scenic objects.

    Scenic objects, which are constructed using specifiers, are implemented
    internally as instances of ordinary Python classes. This abstract class
    implements the procedure to resolve specifiers and determine values for
    the properties of an object, as well as several common methods supported
    by objects.
    """

    def evaluateInner(self, context):
        return self

    @classmethod
    def defaults(cls):  # TODO improve so this need only be done once?
        # find all defaults provided by the class or its superclasses
        allDefs = collections.defaultdict(list)
        for sc in inspect.getmro(cls):
            if hasattr(sc, '__annotations__'):
                for prop, value in sc.__annotations__.items():
                    allDefs[prop].append(PropertyDefault.forValue(value))

        # resolve conflicting defaults
        resolvedDefs = {}
        for prop, defs in allDefs.items():
            primary, rest = defs[0], defs[1:]
            spec = primary.resolveFor(prop, rest)
            resolvedDefs[prop] = spec
        return resolvedDefs

    @classmethod
    def withProperties(cls, props):
        assert all(reqProp in props for reqProp in cls.defaults())
        assert all(not needs_lazy_evaluation(val) for val in props.values())
        specs = (Specifier(prop, val) for prop, val in props.items())
        return cls(*specs)

    def __init__(self, *args, **kwargs):
        # Validate specifiers
        name = type(self).__name__
        defs = self.defaults()

        specifiers = list(args) + [Specifier(p, v) for p, v in kwargs.items()]

        properties = group_by(specifiers, lambda s: s.property)
        for p, specs in properties.items():
            if len(specs) == 1:
                properties[p] = specs[0]
            else:
                spec_vals = [s.value for s in specs]
                regs = [sv.region for sv in spec_vals]
                # r1 = spec_vals[0].region
                # r2 = spec_vals[1].region
                delayed_intersection = makeDelayedFunctionCall(intersect_distribution, regs, {})
                intersect_spec = Specifier(p, delayed_intersection)
                properties[p] = intersect_spec
                specifiers.append(intersect_spec)
                for s in specs:
                    specifiers.remove(s)
            # else:
            #     raise RuntimeParseError(f'property "{p}" of {name} specified twice (non combinable)')

        # TODO: dealing with duplicates part using intersections

        optionals = collections.defaultdict(list)
        for s in specifiers:
            relevant_opts = [o for o in s.optionals if o in defs]
            for opt in relevant_opts:
                optionals[opt].append(s)

        # Decide which optionals to use
        optionals_for_spec = collections.defaultdict(set)
        for opt, specs in optionals.items():
            if opt in properties:
                continue  # optionals do not override a primary specification
            if len(specs) > 1:
                raise RuntimeParseError(f'property "{opt}" of {name} specified twice (optionally)')
            assert len(specs) == 1
            spec = specs[0]
            properties[opt] = spec
            optionals_for_spec[spec].add(opt)

        # Add any default specifiers needed
        for prop in filter(lambda x: x not in properties, defs):
            specifiers.append(defs[prop])
            properties[prop] = defs[prop]

        # Topologically sort specifiers
        order = []
        seen, done = set(), set()

        def dfs(spec):
            if spec in done:
                return
            elif spec in seen:
                raise RuntimeParseError(f'specifier for property {spec.property} '
                                        'depends on itself')
            seen.add(spec)
            for dep in spec.requiredProperties:
                child = properties.get(dep)
                if child is None:
                    raise RuntimeParseError(f'property {dep} required by '
                                            f'specifier {spec} is not specified')
                else:
                    dfs(child)
            order.append(spec)
            done.add(spec)

        for spec in specifiers:
            dfs(spec)
        assert len(order) == len(specifiers)

        # Evaluate and apply specifiers
        for spec in order:
            v = to_distribution(spec.value.evaluateIn(self))
            assert not needs_lazy_evaluation(v)
            setattr(self, spec.property, v)
            for opt in optionals_for_spec[spec]:
                assert opt in spec.optionals
                setattr(self, opt, getattr(v, opt))

        # Set up dependencies
        super().__init__(dependencies=[getattr(self, p) for p in properties if needs_sampling(getattr(self, p))])
        self.properties = set(properties)

    def sample_given_dependencies(self, dep_values):
        return self.withProperties({prop: dep_values[getattr(self, prop)]
                                    for prop in self.properties})

    def allProperties(self):
        return {prop: getattr(self, prop) for prop in self.properties}

    def copyWith(self, **overrides):
        props = self.allProperties()
        props.update(overrides)
        return self.withProperties(props)

    def isEquivalentTo(self, other):
        if type(other) is not type(self):
            return False
        return areEquivalent(self.allProperties(), other.allProperties())

    def __str__(self):
        if hasattr(self, 'properties'):
            all_props = {prop: getattr(self, prop) for prop in self.properties}
        else:
            all_props = '<under construction>'
        return f'{type(self).__name__}({all_props})'


def intersect_distribution(*regions) -> PointInRegionDistribution:
    return PointInRegionDistribution(IntersectionRegion(*regions))


class Mutator:
    """An object controlling how the ``mutate`` statement affects an `Object`.

    A `Mutator` can be assigned to the ``mutator`` property of an `Object` to
    control the effect of the ``mutate`` statement. When mutation is enabled
    for such an object using that statement, the mutator's `appliedTo` method
    is called to compute a mutated version.
    """

    def appliedTo(self, obj):
        """Return a mutated copy of the object. Implemented by subclasses."""
        raise NotImplementedError


class Point3D(Constructible):
    position: Vector3D(0, 0, 0)
    visibleDistance: 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corners = (self.position,)
        self.visibleRegion = SphericalRegion(self.position, self.visibleDistance)

    def to_vector_3d(self):
        return self.position.to_vector_3d()

    def sample_given_dependencies(self, dep_values):
        sample = super().sample_given_dependencies(dep_values)
        # TODO: Mutation stuff here?
        return sample

    def __getattr__(self, attr):
        if hasattr(Vector3D, attr):
            return getattr(self.to_vector_3d(), attr)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


class OrientedPoint3D(Point3D, Oriented):
    orientation: Vector3D(0, 0, 0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orientation = toType(self.orientation, Vector3D)

    def to_orientation(self):
        return self.orientation


def rel_pos_3d(rel_pos, reference_pos, reference_orientation):
    pos = reference_pos + rotate_euler_v3d(rel_pos, reference_orientation)
    return OrientedPoint3D(position=pos, orientation=reference_orientation)


class Object(Point3D, Oriented):
    width: 1
    height: 1
    length: 1
    orientation: Vector3D(0, 0, 0)
    allowCollisions: False
    requireVisible: True
    cameraOffset: Vector(0, 0)

    def to_orientation(self):
        return self.orientation

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import scenic3d.syntax.veneer as veneer  # TODO improve?
        veneer.registerObject(self)

        self.hw = hw = self.width / 2
        self.hh = hh = self.height / 2
        self.hl = hl = self.length / 2

        self.radius = hypot(hw, hl, hh)  # circumcircle; for collision detection
        self.inradius = min(hw, hl, hh)  # incircle; for collision detection

        self.left = rel_pos_3d(Vector3D(-hw, 0, 0), self.position, self.orientation)
        self.right = rel_pos_3d(Vector3D(hw, 0, 0), self.position, self.orientation)
        self.front = rel_pos_3d(Vector3D(0, hl, 0), self.position, self.orientation)
        self.back = rel_pos_3d(Vector3D(0, -hl, 0), self.position, self.orientation)

        self.top = rel_pos_3d(Vector3D(0, 0, hh), self.position, self.orientation)
        self.top_front = rel_pos_3d(Vector3D(0, hl, hh), self.position, self.orientation)
        self.top_back = rel_pos_3d(Vector3D(0, -hl, hh), self.position, self.orientation)

        self.bottom = rel_pos_3d(Vector3D(0, 0, -hh), self.position, self.orientation)

        self.front_left = rel_pos_3d(Vector3D(-hw, hl, 0), self.position, self.orientation)
        self.front_right = rel_pos_3d(Vector3D(hw, hl, 0), self.position, self.orientation)
        self.back_left = rel_pos_3d(Vector3D(-hw, -hl, 0), self.position, self.orientation)
        self.back_right = rel_pos_3d(Vector3D(-hw, hl, 0), self.position, self.orientation)

        self.corners = tuple(self.position + rotate_euler_v3d(Vector3D(*offset), self.orientation)
                             for offset in it.product((hw, -hw), (hl, -hl), (hh, -hh)))
        self.forward = rotate_euler_v3d(Vector3D(0, 1, 0), self.orientation)
        # camera = self.position.offsetRotated(self.heading, self.cameraOffset)
        # self.visibleRegion = SectorRegion(camera, self.visibleDistance, self.heading, self.viewAngle)
        self._relations = []

    def dimensions(self) -> Vector3D:
        return Vector3D(self.width, self.length, self.height)

    def show_3d(self, ax, highlight=False):
        if needs_sampling(self):
            raise RuntimeError('tried to show_3d() symbolic Object')

        color = self.color if hasattr(self, 'color') else (1, 0, 0)
        draw_cube(ax, np.array([*self.position]), np.array([self.width, self.length, self.height]),
                  np.array([*self.orientation]), color=color)
        ax.quiver(self.position[0], self.position[1], self.position[2], self.forward[0], self.forward[1],
                  self.forward[2], length=0.2, normalize=True)
