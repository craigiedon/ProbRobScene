"""Implementations of the built-in Scenic classes."""
import collections
import dataclasses
import inspect
import itertools as it
from dataclasses import dataclass, is_dataclass, fields
from typing import List

import numpy as np
from multimethod import multimethod

import probRobScene.core.distributions
from probRobScene.core.distributions import Samplable, needs_sampling
import probRobScene.core.geometry as g
from probRobScene.core.lazy_eval import needs_lazy_evaluation, makeDelayedFunctionCall, evaluate_in, DelayedArgument
from probRobScene.core.plotUtil3d import draw_cube
from probRobScene.core.regions import PointInRegionDistribution, Intersection, Spherical
from probRobScene.core.specifiers import Specifier, PropertyDefault, pd_for_value
from probRobScene.core.utils import areEquivalent, RuntimeParseError, group_by
from probRobScene.core.vectors import Vector3D, rotate_euler_v3d


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

        # Get the inheritance chain for this class
        base_classes = inspect.getmro(cls)
        for sc in base_classes:
            # Ok, so what is going on here is that scenic's parser bakes default arguments for properties DIRECTLY INTO THE TYPE ANNOTATIONS
            # TODO: Figure out how to make this not absolutely wild
            if hasattr(sc, '__annotations__'):
                annots = sc.__annotations__
                for prop, value in annots.items():
                    if isinstance(value, PropertyDefault):
                        allDefs[prop].append(value)
                    elif is_dataclass(sc):
                        field = next(f for f in fields(sc) if f.name == prop)
                        if field.default != dataclasses.MISSING:
                            allDefs[prop].append(field.default)

        # resolve conflicting defaults
        resolvedDefs = {}
        for prop, defs in allDefs.items():
            primary, rest = defs[0], defs[1:]
            # TODO: unbreak what you've broken here
            # spec = primary.resolve_for(prop, rest)
            if isinstance(primary, PropertyDefault):
                spec = primary.resolve_for(prop)
            else:
                spec = Specifier(prop, primary)
            resolvedDefs[prop] = spec
        return resolvedDefs

    @classmethod
    def withProperties(cls, props):
        assert all(reqProp in props for reqProp in cls.defaults())
        assert all(not needs_lazy_evaluation(val) for val in props.values())
        specs = (Specifier(prop, val) for prop, val in props.items())
        return cls(*specs)

    def __init__(self, *args, **kwargs):
        self._conditioned = self
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

        def dfs(spec : Specifier):
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
            v = probRobScene.core.distributions.to_distribution(evaluate_in(spec.value, self))
            assert not needs_lazy_evaluation(v)
            setattr(self, spec.property, v)
            for opt in optionals_for_spec[spec]:
                assert opt in spec.optionals
                setattr(self, opt, getattr(v, opt))

        # Set up dependencies
        self.properties = set(properties)

    def dependencies(self) -> List:
        return [getattr(self, p) for p in self.properties if needs_sampling(getattr(self, p))]

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
    return PointInRegionDistribution(Intersection(regions))


@dataclass(eq=False)
class Point3D(Constructible):
    position: Vector3D = Vector3D(0, 0, 0)
    visibleDistance: float = 50

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __post_init__(self):
        self.corners = (self.position,)
        self.visibleRegion = Spherical(self.position, self.visibleDistance)

    def to_vector_3d(self):
        return self.position

    def sample_given_dependencies(self, dep_values):
        sample = super().sample_given_dependencies(dep_values)
        return sample

    def __getattr__(self, attr):
        if hasattr(self.position, attr):
            return getattr(self.to_vector_3d(), attr)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


@dataclass(eq=False)
class OrientedPoint3D(Point3D):
    orientation: Vector3D = Vector3D(0, 0, 0)


def rel_pos_3d(rel_pos, reference_pos, reference_orientation):
    pos = reference_pos + rotate_euler_v3d(rel_pos, reference_orientation)
    return OrientedPoint3D(position=pos, orientation=reference_orientation)


@dataclass(eq=False)
class Object(Point3D):
    width: float = 1
    height: float = 1
    length: float = 1
    orientation: Vector3D = Vector3D(0, 0, 0)
    allowCollisions: bool = False
    requireVisible: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def hw(self):
        return self.width / 2.0

    @property
    def hh(self):
        return self.height / 2.0

    @property
    def hl(self):
        return self.length / 2.0

    @property
    def radius(self):
        return g.hypot(self.hw, self.hl, self.hh)

    @property
    def inradius(self):
        return g.min(self.hw, self.hl, self.hh)

    @property
    def left(self):
        return rel_pos_3d(Vector3D(-self.hw, 0, 0), self.position, self.orientation)

    @property
    def right(self):
        return rel_pos_3d(Vector3D(self.hw, 0, 0), self.position, self.orientation)

    @property
    def front(self):
        return rel_pos_3d(Vector3D(0, self.hl, 0), self.position, self.orientation)

    @property
    def back(self):
        return rel_pos_3d(Vector3D(0, -self.hl, 0), self.position, self.orientation)

    @property
    def top(self):
        return rel_pos_3d(Vector3D(0, 0, self.hh), self.position, self.orientation)

    @property
    def corners(self):
        return tuple(self.position + rotate_euler_v3d(Vector3D(*offset), self.orientation)
                     for offset in it.product((self.hw, -self.hw), (self.hl, -self.hl), (self.hh, -self.hh)))

    @property
    def top_front(self):
        return rel_pos_3d(Vector3D(0, self.hl, self.hh), self.position, self.orientation)

    @property
    def top_back(self):
        return rel_pos_3d(Vector3D(0, -self.hl, self.hh), self.position, self.orientation)

    @property
    def bottom(self):
        return rel_pos_3d(Vector3D(0, 0, -self.hh), self.position, self.orientation)

    @property
    def front_left(self):
        return rel_pos_3d(Vector3D(-self.hw, self.hl, 0), self.position, self.orientation)

    @property
    def front_right(self):
        return rel_pos_3d(Vector3D(self.hw, self.hl, 0), self.position, self.orientation)

    @property
    def back_left(self):
        return rel_pos_3d(Vector3D(-self.hw, -self.hl, 0), self.position, self.orientation)

    @property
    def back_right(self):
        return rel_pos_3d(Vector3D(-self.hw, self.hl, 0), self.position, self.orientation)

    @property
    def forward(self):
        return rotate_euler_v3d(Vector3D(0, 1, 0), self.orientation)

    @property
    def dimensions(self) -> Vector3D:
        return Vector3D(self.width, self.length, self.height)


def show_3d(o: Object, ax, highlight=False):
    if needs_sampling(o):
        raise RuntimeError('tried to show_3d() symbolic Object')

    color = o.color if hasattr(o, 'color') else (1, 0, 0)
    draw_cube(ax, np.array([*o.position]), np.array([o.width, o.length, o.height]),
              np.array([*o.orientation]), color=color)
    ax.quiver(o.position[0], o.position[1], o.position[2], o.forward[0], o.forward[1],
              o.forward[2], length=0.2, normalize=True)
