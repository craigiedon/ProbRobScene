"""Objects representing regions in space."""
import abc
import itertools
import math
import random
from typing import Optional, Tuple, List, Union, TYPE_CHECKING

import numpy as np
import scipy.spatial
from scipy.linalg import inv
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation as R

from probRobScene.core.distributions import Samplable, RejectionException, needs_sampling, distributionFunction
from probRobScene.core.geometry import cuboid_contains_point, normalize
from probRobScene.core.geometry import sin, cos, hypot, min_and_max
from probRobScene.core.lazy_eval import value_in_context
from probRobScene.core.type_support import toVector
from probRobScene.core.utils import areEquivalent
from probRobScene.core.vectors import Vector, OrientedVector, VectorDistribution, Vector3D, rotate_euler_v3d, rotation_to_euler, reverse_euler, rotate_euler


# class Region(Samplable, abc.ABC):
#     """Abstract class for regions."""
#
#     def __init__(self, name, *dependencies, orientation=None):
#         super().__init__(dependencies)
#         self.name = name
#         self.orientation = orientation
#
#     @abc.abstractmethod
#     def uniform_point_inner(self):
#         """Do the actual random sampling. Implemented by subclasses."""
#         raise NotImplementedError()
#
#     @abc.abstractmethod
#     def contains_point(self, point):
#         """Check if the `Region` contains a point. Implemented by subclasses."""
#         raise NotImplementedError()
#
#     def contains_object(self, obj):
#         return all(self.contains_point(c) for c in obj.corners)
#
#     def __contains__(self, thing):
#         """Check if this `Region` contains an object or vector."""
#         from probRobScene.core.object_types import Object
#         if isinstance(thing, Object):
#             return self.contains_object(thing)
#         vec = toVector(thing, '"X in Y" with X not an Object or a vector')
#         return self.contains_point(vec)
#
#     def __str__(self):
#         return f'<Region {self.name}>'


class PointInRegionDistribution(VectorDistribution):
    """Uniform distribution over points in a Region"""

    def support_interval(self):
        return None, None

    def evaluateInner(self, context):
        r = value_in_context(self.region, context)
        return PointInRegionDistribution(r)

    def __init__(self, region):
        super().__init__(region)
        self.region = region

    def sample_given_dependencies(self, dep_values):
        return dep_values[self.region].uniform_point_inner()

    def __str__(self):
        return f'PointIn({self.region})'


class Intersect(abc.ABC):
    @abc.abstractmethod
    def intersect(self, other):
        raise NotImplementedError()


class BoundingBox(abc.ABC):
    @abc.abstractmethod
    def getAABB(self) -> Tuple[np.array]:
        """Axis-aligned bounding box"""
        raise NotImplementedError()


class Convex(abc.ABC):
    @abc.abstractmethod
    def to_hsi(self) -> HalfspaceIntersection:
        raise NotImplementedError


class Oriented(abc.ABC):
    @abc.abstractmethod
    def to_orientation(self):
        raise NotImplementedError


def orient_along_region(region: Region, vec):
    """Orient the given vector along the region's orientation, if any."""
    if region.orientation is None:
        return vec
    else:
        return OrientedVector(vec.x, vec.y, region.orientation[vec])


class AllRegion(Region, Intersect):
    """Region consisting of all space."""

    def uniform_point_inner(self):
        raise RuntimeError("Should not be sampling from the all region")

    def sample_given_dependencies(self, dep_values):
        raise RuntimeError("Should not be sampling from the all region")

    def evaluateInner(self, context):
        return self

    def intersect(self, other):
        return other

    def contains_point(self, point):
        return True

    def contains_object(self, obj):
        return True

    def __eq__(self, other):
        return type(other) is AllRegion

    def __hash__(self):
        return hash(AllRegion)


class EmptyRegion(Region, Intersect):
    """Region containing no points."""

    def sample_given_dependencies(self, dep_values):
        raise RejectionException(f'sampling from empty Region')

    def evaluateInner(self, context):
        return self

    def intersect(self, other):
        return self

    def uniform_point_inner(self):
        raise RejectionException(f'sampling empty Region')

    def contains_point(self, point):
        return False

    def contains_object(self, obj):
        return False

    def show(self, plt, style=None):
        pass

    def __eq__(self, other):
        return type(other) is EmptyRegion

    def __hash__(self):
        return hash(EmptyRegion)


everywhere = AllRegion('everywhere')
nowhere = EmptyRegion('nowhere')


class SphericalRegion(Region, BoundingBox):
    def __init__(self, center, radius, resolution=32):
        super().__init__('Sphere', center, radius)
        self.center = center.to_vector_3d()
        self.radius = radius
        self.circumsphere = (self.center, self.radius)
        self.resolution = resolution

    def sample_given_dependencies(self, dep_values):
        return SphericalRegion(dep_values[self.center], dep_values[self.radius])

    def evaluateInner(self, context):
        center = value_in_context(self.center, context)
        radius = value_in_context(self.radius, context)
        return SphericalRegion(center, radius)

    def contains_point(self, point):
        point = point.to_vector_3d()
        return point.distanceTo(self.center) <= self.radius

    def uniform_point_inner(self):
        x, y, z = self.center
        u = 2.0 * random.random() - 1.0
        phi = 2.0 * math.pi * random.random()
        r = random.random() ** (1 / 3.)
        x_offset = r * cos(phi) * (1 - u ** 2) ** 0.5
        y_offset = r * sin(phi) * (1 - u ** 2) ** 0.5
        z_offset = r * u
        pt = Vector3D(x + x_offset, y + y_offset, z + z_offset)
        return pt

    def getAABB(self):
        x, y, z = self.center
        r = self.radius
        return np.array((x - r, y - r, z - r)), np.array((x + r, y + r, z + r))

    def isEquivalentTo(self, other):
        if type(other) is not SphericalRegion:
            return False
        return areEquivalent(other.center, self.center) and areEquivalent(other.radius, self.radius)

    def __str__(self):
        return f'SphericalRegion({self.center}, {self.radius})'


class IntersectionRegion(Region):

    def __init__(self, *regions):
        super().__init__('Intersection', *regions)
        self.regions = regions
        # self.r1 = r1
        # self.r2 = r2

    def uniform_point_inner(self):
        raise RuntimeError("Should not be sampling from this region. Prune out")

    def contains_point(self, point):
        raise NotImplementedError

    def sample_given_dependencies(self, dep_values):
        raise NotImplementedError

    def evaluateInner(self, context):
        raise NotImplementedError


class HalfSpaceRegion(Region, Intersect, Convex):
    def to_hsi(self) -> HalfspaceIntersection:
        pos = np.array(self.point + self.normal * (self.dist / 2.0))
        return cube_to_hsi(pos, self.dist * np.ones(3), np.array(self.rot))

    def sample_given_dependencies(self, dep_values):
        return HalfSpaceRegion(dep_values[self.point], dep_values[self.normal], dep_values[self.dist])

    def evaluateInner(self, context):
        point = value_in_context(self.point, context)
        normal = value_in_context(self.normal, context)
        dist = value_in_context(self.dist, context)
        return HalfSpaceRegion(point, normal, dist)

    def intersect(self, other):
        if isinstance(other, CuboidRegion):
            return intersect_cuboid_half_space(other, self)
        if isinstance(other, HalfSpaceRegion):
            return intersect_halfspaces(self, other)
        if isinstance(other, ConvexPolyhedronRegion):
            return intersect_halfspace_convpoly(self, other)
        if isinstance(other, (ConvexPolygon3DRegion, Rectangle3DRegion)):
            return intersect_poly_convex(other, self)
        if isinstance(other, (EmptyRegion, AllRegion)):
            return other.intersect(self)

        raise NotImplementedError("Havent yet filled in the intersection code")

    def __init__(self, point: Vector3D, normal: Vector3D, dist: float = 100.0):
        super().__init__('PlaneCast', point, normal, dist)
        self.point = point
        self.normal = normal
        self.dist = dist
        self.rot = rotation_to_euler(Vector3D(0, 0, 1), self.normal)

    def uniform_point_inner(self) -> Vector3D:
        untransformed_point = Vector3D(np.random.uniform(0, self.dist, 3))
        return rotate_euler_v3d(untransformed_point, self.rot) + self.point

    def contains_point(self, p):
        return np.dot(self.normal, p - self.point) >= 0


class ConvexPolyhedronRegion(Region, Intersect, BoundingBox, Convex):
    def to_hsi(self) -> HalfspaceIntersection:
        return self.hsi

    def sample_given_dependencies(self, dep_values):
        return ConvexPolyhedronRegion(dep_values[self.hsi])

    def evaluateInner(self, context):
        hsi = value_in_context(self.hsi, context)
        return ConvexPolyhedronRegion(hsi)

    def intersect(self, other):
        # Cuboid, Halfspace, convexpoly, empty, any, other
        if isinstance(other, CuboidRegion):
            return intersect_cuboid_convpoly(other, self)
        if isinstance(other, HalfSpaceRegion):
            return intersect_halfspace_convpoly(other, self)
        if isinstance(other, ConvexPolyhedronRegion):
            return intersect_convpolys(self, other)
        if isinstance(other, (ConvexPolygon3DRegion, Rectangle3DRegion)):
            return intersect_poly_convex(other, self)
        if isinstance(other, (EmptyRegion, AllRegion)):
            return other.intersect(self)

        raise NotImplementedError(f"Intersection between {type(self)} and {type(other)} not implemented")

    def __init__(self, hsi: HalfspaceIntersection):
        super().__init__('ConvexPoly', hsi)
        self.hsi = hsi
        convex_hull = ConvexHull(hsi.intersections)
        self.corners = tuple(convex_hull.points[i] for i in convex_hull.vertices)

    def uniform_point_inner(self):
        return Vector3D(*hit_and_run(self.hsi))

    def contains_point(self, point):
        for hs_ineq in self.hsi.halfspaces:
            if np.dot(point, hs_ineq[:-1]) + hs_ineq[-1] > 0:
                return False
        return True

    def getAABB(self):
        cs = np.array(self.corners)
        return np.min(cs, axis=0), np.max(cs, axis=0)


class CuboidRegion(Region, Intersect, BoundingBox, Convex, Oriented):

    def to_orientation(self):
        return self.orientation

    def to_hsi(self) -> HalfspaceIntersection:
        return cube_to_hsi(np.array(self.position), self.dimensions(), np.array(self.orientation))

    def intersect(self, other: Intersect):
        if isinstance(other, CuboidRegion):
            return intersect_cuboid_cuboid(self, other)
        if isinstance(other, HalfSpaceRegion):
            return intersect_cuboid_half_space(self, other)
        if isinstance(other, ConvexPolyhedronRegion):
            return intersect_cuboid_convpoly(self, other)
        if isinstance(other, (ConvexPolygon3DRegion, Rectangle3DRegion)):
            return intersect_poly_convex(other, self)
        if isinstance(other, (EmptyRegion, AllRegion)):
            return other.intersect(self)

        raise NotImplementedError(f'Have forgotten to implement intersection between {type(self)} and {type(other)}')

    def __init__(self, position, orientation, width, length, height):
        super().__init__('Cuboid', position, orientation, width, length, height)
        self.position = position
        self.orientation = orientation
        self.width = width
        self.length = length
        self.height = height

        self.hw = hw = width / 2.0
        self.hl = hl = length / 2.0
        self.hh = hh = height / 2.0

        self.radius = np.linalg.norm((hw, hl, hh))
        self.corners = tuple(self.position + rotate_euler_v3d(Vector3D(*offset), self.orientation)
                             for offset in itertools.product((hw, -hw), (hl, -hl), (hh, -hh)))
        self.circumcircle = (self.position, self.radius)

    def dimensions(self):
        return np.array([self.width, self.length, self.height])

    def contains_point(self, point):
        return cuboid_contains_point(self, point)

    def sample_given_dependencies(self, dep_vals):
        return CuboidRegion(dep_vals[self.position], dep_vals[self.orientation], dep_vals[self.width], dep_vals[self.length], dep_vals[self.height])

    def evaluateInner(self, context):
        position = value_in_context(self.position, context)
        orientation = value_in_context(self.orientation, context)
        width = value_in_context(self.width, context)
        length = value_in_context(self.length, context)
        height = value_in_context(self.height, context)
        return CuboidRegion(position, orientation, width, length, height)

    def uniform_point_inner(self) -> Vector3D:
        hw, hl, hh = self.hw, self.hl, self.hh
        rx = random.uniform(-hw, hw)
        ry = random.uniform(-hl, hl)
        rz = random.uniform(-hh, hh)
        pt = self.position + rotate_euler_v3d(Vector3D(rx, ry, rz), self.orientation)
        return pt

    def getAABB(self):
        xs, ys, zs = zip(*self.corners)
        min_x, max_x = min_and_max(xs)
        min_y, max_y = min_and_max(ys)
        min_z, max_z = min_and_max(zs)
        return (np.array((min_x, min_y, min_z))), (np.array((max_x, max_y, max_z)))

    def isEquivalentTo(self, other):
        if type(other) is not CuboidRegion:
            return False
        return (areEquivalent(other.position, self.position)
                and areEquivalent(other.orientation, self.orientation)
                and areEquivalent(other.width, self.width)
                and areEquivalent(other.length, self.length)
                and areEquivalent(other.height, self.height))

    def __str__(self):
        return f'CuboidRegion({self.position},{self.orientation},{self.width},{self.length},{self.height}'


class PlaneRegion(Region, Intersect):
    def __init__(self, origin: Vector3D, normal: Vector3D, dist=100.0):
        super().__init__('Plane', origin, normal, dist)
        self.origin = origin
        self.normal = normal
        self.dist = dist

    def uniform_point_inner(self):

        # We don't want arbitrarily chosen axis to be exactly aligned with normal
        if 1.0 - np.abs(self.normal, Vector3D(1.0, 0.0, 0.0)) >= 1e-5:
            u = np.cross(self.normal, Vector3D(1.0, 0.0, 0.0))
        else:
            u = np.cross(self.normal, Vector3D(0.0, 1.0, 0.0))

        v = np.cross(self.normal, u)

        a = np.random.uniform(-self.dist / 2.0, self.dist / 2.0)
        b = np.random.uniform(-self.dist / 2.0, self.dist / 2.0)

        offset = a * u + b * v
        return self.origin + offset

    def contains_point(self, point):
        return np.abs(np.dot(self.normal - self.origin, point - self.origin)) <= 1e-8

    def sample_given_dependencies(self, dep_values):
        return PlaneRegion(dep_values[self.origin], dep_values[self.normal], dep_values[self.dist])

    def evaluateInner(self, context):
        origin = value_in_context(self.origin)
        normal = value_in_context(self.normal)
        dist = value_in_context(self.dist)
        return PlaneRegion(origin, normal, dist)

    def intersect(self, other):
        if isinstance(other, PlaneRegion):
            return intersect_planes(self, other)

        raise NotImplementedError("Not yet done intersections with other types")


class Rectangle3DRegion(Region, Convex, Intersect, Oriented):

    def to_orientation(self):
        return self.rot

    def to_hsi(self) -> HalfspaceIntersection:
        hs_origins = np.array([[self.width / 2.0, 0], [0, self.length / 2.0], [-self.width / 2.0, 0], [0, -self.length / 2.0]])
        hs_norms = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])

        hs_ineqs = halfspaces_to_inequalities(hs_norms, hs_origins)
        return HalfspaceIntersection(hs_ineqs, np.array([0.0, 0.0]))

    def __init__(self, width, length, origin, rot):
        super().__init__('Rectangle3D', width, length, origin, rot)
        self.width = width
        self.length = length
        self.origin = origin

        self.rot = rot
        self.rev_rot: Vector3D = reverse_euler(self.rot)
        self.normal = rotate_euler_v3d(Vector3D(0, 0, 1), rot)

        self.w_ax = rotate_euler_v3d(Vector3D(1, 0, 0), rot)
        self.l_ax = rotate_euler_v3d(Vector3D(0, 1, 0), rot)

    def uniform_point_inner(self):
        x = random.uniform(-self.width / 2.0, self.width / 2.0)
        y = random.uniform(-self.length / 2.0, self.length / 2.0)
        flat_point = Vector3D(x, y, 0)
        return rotate_euler_v3d(flat_point, self.rot) + self.origin

    def contains_point(self, point):
        flat_point = rotate_euler_v3d(point - self.origin, self.rev_rot)
        return (np.abs(flat_point.x) <= self.width / 2.0 and
                np.abs(flat_point.y) <= self.length / 2.0 and
                np.abs(flat_point.z) <= 1e-8)  # "Roughly equals" zero

    def sample_given_dependencies(self, dep_values):
        return Rectangle3DRegion(dep_values[self.width],
                                 dep_values[self.length],
                                 dep_values[self.origin],
                                 dep_values[self.rot])

    def evaluateInner(self, context):
        width = value_in_context(self.width, context)
        length = value_in_context(self.length, context)
        origin = value_in_context(self.origin, context)
        rot = value_in_context(self.rot, context)
        return Rectangle3DRegion(width, length, origin, rot)

    def intersect(self, other):
        if isinstance(other, (CuboidRegion, HalfSpaceRegion, ConvexPolyhedronRegion)):
            return intersect_poly_convex(self, other)
        if isinstance(other, Rectangle3DRegion):
            return intersect_rects(self, other)
        if isinstance(other, ConvexPolygon3DRegion):
            raise NotImplementedError("convex poly intersections resulting in lines not supported")
        if isinstance(other, (EmptyRegion, AllRegion)):
            return other.intersect(self)

        raise NotImplementedError("Havent yet filled in the intersection code")


class Line3DRegion(Region, Intersect):
    def __init__(self, origin: Vector3D, direction: Vector3D, dist: float = 100.0):
        super().__init__('Line3DRegion', origin, direction)
        self.origin = origin
        self.direction = direction
        self.dist = dist

    def uniform_point_inner(self):
        t = np.random.uniform(-self.dist / 2.0, self.dist / 2.0)
        return self.origin + t * self.direction

    def contains_point(self, point):
        pv = point - self.origin
        pv = pv / np.linalg.norm(pv)
        dp_unsigned = np.abs(np.dot(pv, self.direction))

        # Looking for dot product to be +-1.0 (with wiggle room)
        return 1.0 - dp_unsigned <= 1e-8

    def sample_given_dependencies(self, dep_values):
        return Line3DRegion(dep_values[self.origin], dep_values[self.direction], dep_values[self.dist])

    def evaluateInner(self, context):
        origin = value_in_context(self.origin)
        direction = value_in_context(self.direction)
        dist = value_in_context(self.dist)
        return Line3DRegion(origin, direction, dist)

    def intersect(self, other):
        raise NotImplementedError


class LineSeg3DRegion(Region, Intersect):
    def __init__(self, start: Vector3D, end: Vector3D):
        super().__init__('LineSeg3DRegion', start, end)
        self.start = start
        self.end = end

    def uniform_point_inner(self):
        t = random.uniform(0.0, 1.0)
        return (1.0 - t) * self.start + t * self.end

    def contains_point(self, point):
        s_e_dir = self.end - self.start / np.linalg.norm(self.end - self.start)

        s_p_dir = point - self.start / np.linalg.norm(point - self.start)
        p_e_dir = self.end - point / np.linalg.norm(self.end - point)

        dp1 = np.dot(s_e_dir, s_p_dir)
        dp2 = np.dot(s_e_dir, p_e_dir)

        return np.abs(dp1 - 1.0) <= 1e-8 and np.abs(dp2 - 1.0) <= 1e-8  # A little bit of wiggle room to be a small epsilon of rounding off of the line

    def sample_given_dependencies(self, dep_values):
        return LineSeg3DRegion(dep_values[self.start], dep_values[self.end])

    def evaluateInner(self, context):
        start = value_in_context(self.start, context)
        end = value_in_context(self.end, context)
        return LineSeg3DRegion(start, end)

    def intersect(self, other):
        if isinstance(other, (EmptyRegion, AllRegion)):
            return other.intersect(self)
        if isinstance(other, ConvexPolyhedronRegion):
            return intersect_lineseg_convex(self, other)

        raise NotImplementedError(f"Intersection between {type(self)} and {type(other)} not yet implemented")


class ConvexPolygon3DRegion(Region, Convex, Intersect, Oriented):

    def to_orientation(self):
        return self.rot

    def to_hsi(self) -> HalfspaceIntersection:
        return self.hsi

    def __init__(self, hsi, origin, rot):
        super().__init__("ConvexPolygon3D", hsi, origin, rot)
        self.hsi = hsi
        self.origin = origin
        self.rot = rot
        self.rev_rot: Vector3D = reverse_euler(self.rot)

        self.normal = rotate_euler_v3d(Vector3D(0, 0, 1), rot)

    def uniform_point_inner(self):
        random_point_flat = Vector3D(*hit_and_run(self.hsi), 0)
        return rotate_euler_v3d(random_point_flat, self.rot) + self.origin

    def contains_point(self, point):
        flat_point = rotate_euler_v3d(point - self.origin, self.rev_rot)
        if np.abs(flat_point[-1]) > 1e-8:
            return False

        for hs_ineq in self.hsi.halfspaces:
            if np.dot(flat_point[:-1], hs_ineq[:-1]) + hs_ineq[-1] > 0:
                return False
        return True

    def sample_given_dependencies(self, dep_values):
        return ConvexPolygon3DRegion(dep_values[self.hsi], dep_values[self.origin], dep_values[self.rot])

    def evaluateInner(self, context):
        hsi = value_in_context(self.hsi, context)
        origin = value_in_context(self.origin, context)
        rot = value_in_context(self.rot, context)
        return ConvexPolygon3DRegion(hsi, origin, rot)

    def intersect(self, other):
        if isinstance(other, (CuboidRegion, HalfSpaceRegion, ConvexPolyhedronRegion)):
            return intersect_poly_convex(self, other)
        if isinstance(other, (Rectangle3DRegion, ConvexPolygon3DRegion)):
            raise NotImplementedError("Intersections which produce lines are not yet supported")
        if isinstance(other, (EmptyRegion, AllRegion)):
            return other.intersect(self)

        raise NotImplementedError("Havent yet filled in the intersection code")


def hit_and_run(hsi: HalfspaceIntersection, num_steps: int = 10) -> np.array:
    current_point = hsi.interior_point

    for i in range(num_steps):
        random_direction = np.random.normal(size=len(current_point))
        random_direction = random_direction / np.linalg.norm(random_direction)

        ts = []
        for hs_ineq in hsi.halfspaces:
            hs_norm = -hs_ineq[:-1]

            point_coherence = np.dot(current_point, hs_norm) - hs_ineq[-1]
            direction_coherence = np.dot(random_direction, hs_norm)
            if np.abs(direction_coherence) > 1e-9:
                ts.append(-point_coherence / direction_coherence)

        ts = np.array(ts)
        assert len(ts) > 0

        if len(ts[ts > 0]) == 0 or len(ts[ts < 0]) == 0:
            raise Exception
        max_t = np.min(ts[ts > 0])
        min_t = np.max(ts[ts < 0])

        current_point = current_point + np.random.uniform(min_t, max_t) * random_direction

    return current_point


class PointSetRegion(Region):
    """Region consisting of a set of discrete points.

    No :obj:`~probRobScene.core.object_types.Object` can be contained in a `PointSetRegion`,
    since the latter is discrete. (This may not be true for subclasses, e.g.
    `GridRegion`.)

    Args:
        name (str): name for debugging
        points (iterable): set of points comprising the region
        kd_tree (:obj:`scipy.spatial.KDTree`, optional): k-D tree for the points (one will
          be computed if none is provided)
        orientation (:obj:`~probRobScene.core.vectors.VectorField`, optional): orientation for
          the region
        tolerance (float, optional): distance tolerance for checking whether a point lies
          in the region
    """

    def __init__(self, name, points, kd_tree=None, orientation=None, tolerance=1e-6):
        super().__init__(name, orientation=orientation)
        self.points = tuple(points)
        for point in self.points:
            if needs_sampling(point):
                raise RuntimeError('only fixed PointSetRegions are supported')
        self.kd_tree = scipy.spatial.cKDTree(self.points) if kd_tree is None else kd_tree
        self.orientation = orientation
        self.tolerance = tolerance

    def uniform_point_inner(self):
        return orient_along_region(self, Vector(*random.choice(self.points)))

    def contains_point(self, point):
        distance, location = self.kd_tree.query(point)
        return distance <= self.tolerance

    def contains_object(self, obj):
        raise NotImplementedError()

    def __eq__(self, other):
        if type(other) is not PointSetRegion:
            return NotImplemented
        return (other.name == self.name
                and other.points == self.points
                and other.orientation == self.orientation)

    def __hash__(self):
        return hash((self.name, self.points, self.orientation))


@distributionFunction
def intersect_many(*regions) -> Region:
    intersection = AllRegion("All")
    for r in regions:
        intersection = intersection.intersect(r)
    return intersection


@distributionFunction
def intersect_poly_convex(c1: Union[ConvexPolygon3DRegion, Rectangle3DRegion], c2: Convex):
    projected_hsis = proj_hsi_to_plane(c2.to_hsi(), c1.normal, c1.origin)

    if projected_hsis is None:
        return EmptyRegion("Empty")

    hsi = intersect_hsis(projected_hsis, c1.to_hsi())

    if hsi is None:
        return EmptyRegion("Empty")
    return ConvexPolygon3DRegion(hsi, c1.origin, c1.rot)


@distributionFunction
def intersect_cuboid_cuboid(c1: CuboidRegion, c2: CuboidRegion) -> Region:
    if c1.contains_object(c2):
        return c2
    if c2.contains_object(c1):
        return c1

    hs_1 = cube_to_hsi(np.array(c1.position, dtype=float), c1.dimensions(), np.array(c1.orientation, dtype=float))
    hs_2 = cube_to_hsi(np.array(c2.position, dtype=float), c2.dimensions(), np.array(c2.orientation, dtype=float))
    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyhedronRegion(hs_intersection)


@distributionFunction
def intersect_cuboid_half_space(c1: CuboidRegion, c2: HalfSpaceRegion) -> Region:
    if c2.contains_object(c1):
        return c1

    # Otherwise we have to actually do the work:
    # So turn the cuboid into some halfspace inequalities, turn the halfspace class into the correct form, find a feasible point, and smoosh them all together
    hs_1 = cube_to_hsi(np.array(c1.position), c1.dimensions(), np.array(c1.orientation)).halfspaces
    hs_2 = halfspaces_to_inequalities(np.array(c2.normal), np.array(c2.point))

    hs_intersection = intersect_hs_ineqs(np.vstack(hs_1, hs_2))

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyhedronRegion(hs_intersection)


@distributionFunction
def intersect_cuboid_convpoly(r1: CuboidRegion, r2: ConvexPolyhedronRegion) -> Region:
    if r1.contains_object(r2):
        return r2
    if r2.contains_object(r1):
        return r1

    hs_1 = cube_to_hsi(np.array(r1.position), r1.dimensions(), np.array(r1.orientation))
    hs_2 = r2.hsi
    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyhedronRegion(hs_intersection)


@distributionFunction
def intersect_halfspaces(r1: HalfSpaceRegion, r2: HalfSpaceRegion) -> Region:
    c1 = r1.to_hsi()
    c2 = r2.to_hsi()

    hs_1 = cube_to_hsi(np.array(c1.position), c1.dimensions(), np.array(c1.orientation))
    hs_2 = cube_to_hsi(np.array(c2.position), c2.dimensions(), np.array(c2.orientation))

    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyhedronRegion(hs_intersection)


@distributionFunction
def intersect_halfspace_convpoly(r1: HalfSpaceRegion, r2: ConvexPolyhedronRegion) -> Region:
    r1_hs = halfspaces_to_inequalities(np.array([r1.normal]), np.array([r1.point]))
    cp_halfspaces = r2.hsi.halfspaces
    hs_intersection = intersect_hs_ineqs(np.vstack((r1_hs, cp_halfspaces)))

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyhedronRegion(hs_intersection)


@distributionFunction
def intersect_convpolys(r1: ConvexPolyhedronRegion, r2: ConvexPolyhedronRegion) -> Region:
    hs_intersection = intersect_hsis(r1.hsi, r2.hsi)

    if hs_intersection is None:
        return EmptyRegion("empty")

    return ConvexPolyhedronRegion(hs_intersection)

@distributionFunction
def intersect_lineseg_convex(r1: LineSeg3DRegion, r2: ConvexPolyhedronRegion) -> Region:
    # Q: How to check for no overlap?
    halfspaces = r2.to_hsi().halfspaces

    l_origin = r1.start
    start_end_dist = np.linalg.norm(r1.end - r1.start)
    l_dir = (r1.end - r1.start) / start_end_dist

    # The original line segment acts as initial parameters
    t_max = start_end_dist
    t_min = 0.0

    for halfspace in halfspaces:
        A, b = halfspace[:3], halfspace[3]

        if np.dot(A, r1.start) > -b and np.dot(A, r1.end) > -b: # Line and convex poly do not intersect
            return EmptyRegion("empty")

        dir_align = np.dot(A, l_dir)
        point_align = np.dot(A, l_origin)
        if np.abs(dir_align) < 1e-8: # Orthogonal
            continue
        if dir_align > 0: # Pointing in same direction
            t_max = np.minimum(t_max, (-b - point_align) / dir_align)
        else:
            t_min = np.maximum(t_min, (-b - point_align) / dir_align)

    return LineSeg3DRegion(l_origin + t_min * l_dir, l_origin + t_max * l_dir)


@distributionFunction
def intersect_rects(r1: Rectangle3DRegion, r2: Rectangle3DRegion) -> Region:
    # if r1.contains_object(r2):
    #     return r2
    # if r2.contains_object(r1):
    #     return r1
    #
    # if r1.normal == r2.normal and r1.origin != r2.origin:  # Parallel
    #     return EmptyRegion("empty")

    # The infinite line between the two planes the rectangles lie on
    line = intersect_planes(PlaneRegion(r1.origin, r1.normal), PlaneRegion(r2.origin, r2.normal))

    t_max = 100.0
    t_min = -100.0

    bounding_axes = [
        (r1.origin, r1.w_ax, r1.width / 2.0),
        (r1.origin, r1.l_ax, r1.length / 2.0),

        (r2.origin, r2.w_ax, r2.width / 2.0),
        (r2.origin, r2.l_ax, r2.length / 2.0),
    ]

    for b_o, b_ax, m, in bounding_axes:
        axis_alignment = np.dot(line.direction, b_ax)
        if np.abs(axis_alignment) > 1e-8:
            dist_along_axis = np.dot((line.origin - b_o), b_ax)
            if axis_alignment > 0:
                t_max = np.minimum(t_max, (m - dist_along_axis) / axis_alignment)
                t_min = np.maximum(t_min, (-m - dist_along_axis) / axis_alignment)
            else:
                t_min = np.maximum(t_min, (m - dist_along_axis) / axis_alignment)
                t_max = np.minimum(t_max, (-m - dist_along_axis) / axis_alignment)

    start = line.origin + t_min * line.direction
    end = line.origin + t_max * line.direction
    return LineSeg3DRegion(start, end)


@distributionFunction
def intersect_planes(p1: PlaneRegion, p2: PlaneRegion):
    if p1.normal == p2.normal and p1.origin == p2.origin:
        return p1

    if p1.normal == p2.normal:
        return EmptyRegion("empty")

    line_dir = Vector3D(*np.cross(p1.normal, p2.normal))

    s1 = np.dot(p1.normal, p1.origin)
    s2 = np.dot(p2.normal, p2.origin)

    n1_sq = np.dot(p1.normal, p1.normal)
    n2_sq = np.dot(p2.normal, p2.normal)
    n1n2 = np.dot(p1.normal, p2.normal)

    a = (s2 * n1n2 - s1 * n1_sq) / (n1n2 * n1n2 - n1_sq * n2_sq)
    b = (s1 * n1n2 - s2 * n1_sq) / (n1n2 * n1n2 - n1_sq * n2_sq)

    line_origin = a * p1.normal + b * p2.normal
    line_origin = Vector3D(*line_origin)

    return Line3DRegion(line_origin, line_dir)


def cube_to_hsi(cube_centre: np.ndarray, cube_dims: np.ndarray, cube_rot: np.ndarray) -> HalfspaceIntersection:
    hs_norms, hs_origins = cube_to_normals(cube_centre, cube_dims, cube_rot)
    hs_ineqs = halfspaces_to_inequalities(hs_norms, hs_origins)
    return HalfspaceIntersection(hs_ineqs, cube_centre)


def cube_to_normals(cube_centre: np.ndarray, cube_dims: np.ndarray, cube_rot: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r = R.from_euler('zyx', cube_rot, degrees=False)
    face_offsets = r.apply(np.vstack([
        np.diag(cube_dims / 2.0),
        np.diag(-cube_dims / 2.0)
    ]))

    halfspace_normals = r.apply(np.vstack([
        np.diag(-np.ones(3)),
        np.diag(np.ones(3))
    ]))

    halfspace_origins = np.tile(cube_centre, (6, 1)) + face_offsets

    return halfspace_normals, halfspace_origins


def halfspaces_to_inequalities(hs_normals: np.ndarray, hs_origins: np.ndarray) -> np.ndarray:
    """ Converts halfspace normals and origins to Ax + b <= 0 format """
    hs_bs = np.array([np.dot(hs_normals[i], hs_origins[i])
                      for i in range(len(hs_normals))]).reshape((-1, 1))
    return np.append(-hs_normals, hs_bs, axis=1)


def feasible_point(hs_ineqs: np.ndarray) -> Optional[np.array]:
    coefficients = np.zeros(hs_ineqs.shape[1])
    coefficients[-1] = -1
    bounds = [(None, None) for _ in range(len(coefficients) - 1)] + [(1e-6, None)]  # Intersection must have non-zero volume
    A = hs_ineqs[:, :-1]
    A_row_norms = np.linalg.norm(A, axis=1).reshape(-1, 1)
    b = -hs_ineqs[:, -1]

    result = linprog(coefficients, A_ub=np.hstack((A, A_row_norms)), b_ub=b, bounds=bounds, method='revised simplex')

    if result.success:
        return result.x[:-1]
    elif result.status == 2:  # Infeasible, empty intersection
        return None
    else:
        raise ValueError("Feasible point finder failed! Likely because problem is unbounded (or too difficult?)")


def intersect_hsis(hs_1: HalfspaceIntersection, hs_2: HalfspaceIntersection) -> Optional[HalfspaceIntersection]:
    combined_halfspaces = np.vstack((hs_1.halfspaces, hs_2.halfspaces))
    return intersect_hs_ineqs(combined_halfspaces)


def intersect_hs_ineqs(hs_ineqs: np.ndarray) -> Optional[HalfspaceIntersection]:
    fsp = feasible_point(hs_ineqs)
    if fsp is None:
        return None
    return HalfspaceIntersection(hs_ineqs, fsp)


def erode_hsis(original: HalfspaceIntersection, eroder: HalfspaceIntersection):
    tightest_bs = []
    for hs in original.halfspaces:
        result = linprog(-hs[:-1], eroder.halfspaces[:, :-1], -eroder.halfspaces[:, -1], bounds=(None, None))
        assert result.status == 0
        tightest_bs.append(hs[-1] - result.fun)

    eroded_halfspaces = np.column_stack((original.halfspaces[:, :-1], tightest_bs))
    return HalfspaceIntersection(eroded_halfspaces, original.interior_point)


def proj_vec_to_plane(v, plane_norm):
    return v - np.dot(v, plane_norm) * plane_norm


def proj_point_to_plane(p, plane_norm, plane_origin):
    v = p - plane_origin
    return plane_origin + proj_vec_to_plane(v, plane_norm)


def project_to_plane_intersection(p, p1_norm, p1_origin, p2_norm, p2_origin):
    """
    KKT conditions:
    [2A^TA  C^T ] [ x ]= [ 2A^Tb]
    [ C      0  ] [ z ]  [ d    ]

    Objective function is ||x - p||^2
    A matrix is np.identity(3)
    Parts of KKT =
       2 * A^TA
       C^T
       C
       0
    A_t_A = np.identity(3)

    C = [p1_n, p2_n]
    d = [p1_n . p1_o, p2_n . p2_o)]
    Constraint p1 is p1_n * x = b1
    Constraint p2 is p2_n * x = b2

    """
    A = np.identity(3)
    C = np.array([p1_norm, p2_norm])
    d = np.array([
        np.dot(p1_norm, p1_origin),
        np.dot(p2_norm, p2_origin)
    ])

    top_rows = np.hstack((2 * A, C.transpose()))
    bottom_rows = np.pad(C, ((0, 0), (0, 2)))
    M = np.vstack((top_rows, bottom_rows))

    rhs = np.concatenate((2 * A.transpose() @ p, d))

    # So solution is M-1 @ rhs
    solution = inv(M) @ rhs
    projected_point = solution[:3]
    return projected_point


def proj_hsi_to_plane(hsi: HalfspaceIntersection, p_norm: Vector3D, p_origin: Vector3D) -> Optional[HalfspaceIntersection]:
    projected_halfspaces = []
    rev_rot = rotation_to_euler(p_norm, Vector3D(0, 0, 1))

    for hs in hsi.halfspaces:
        hs_norm = hs[:-1]
        dp = np.dot(hs_norm, p_norm)
        if 1.0 - np.abs(dp) >= 1e-9:
            hs_origin = -hs[-1] * hs[:-1]
            projected_origin = project_to_plane_intersection(hs_origin, hs_norm, hs_origin, p_norm, p_origin)

            projected_norm = hs_norm - dp * p_norm
            projected_norm = projected_norm / np.linalg.norm(projected_norm)
            aa_norm = rotate_euler(projected_norm, rev_rot)

            new_b = -np.dot(projected_origin - p_origin, projected_norm)

            projected_halfspaces.append(np.append(aa_norm[:-1], new_b))
    projected_halfspaces = np.array(projected_halfspaces)

    proj_feasible_point = feasible_point(projected_halfspaces)
    if proj_feasible_point is None:
        return None

    return HalfspaceIntersection(projected_halfspaces, proj_feasible_point)
