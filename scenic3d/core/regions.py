"""Objects representing regions in space."""
import abc
import itertools
import math
import random
from typing import Optional, Tuple, List

import numpy as np
import scipy.spatial
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation as R

from scenic3d.core.distributions import Samplable, RejectionException, needs_sampling, distributionFunction
from scenic3d.core.geometry import cuboid_contains_point
from scenic3d.core.geometry import sin, cos, hypot, min_and_max
from scenic3d.core.lazy_eval import value_in_context
from scenic3d.core.type_support import toVector
from scenic3d.core.utils import areEquivalent
from scenic3d.core.vectors import Vector, OrientedVector, VectorDistribution, Vector3D, rotate_euler, rotation_to_euler


class Region(Samplable, abc.ABC):
    """Abstract class for regions."""

    def __init__(self, name, *dependencies, orientation=None):
        super().__init__(dependencies)
        self.name = name
        self.orientation = orientation

    @abc.abstractmethod
    def uniform_point_inner(self):
        """Do the actual random sampling. Implemented by subclasses."""
        raise NotImplementedError()

    @abc.abstractmethod
    def contains_point(self, point):
        """Check if the `Region` contains a point. Implemented by subclasses."""
        raise NotImplementedError()

    def contains_object(self, obj):
        return all(self.contains_point(c) for c in obj.corners)

    def __contains__(self, thing):
        """Check if this `Region` contains an object or vector."""
        from scenic3d.core.object_types import Object
        if isinstance(thing, Object):
            return self.contains_object(thing)
        vec = toVector(thing, '"X in Y" with X not an Object or a vector')
        return self.contains_point(vec)

    def __str__(self):
        return f'<Region {self.name}>'


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
    def getAABB(self):
        """Axis-aligned bounding box"""
        raise NotImplementedError()


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
        return (x - r, y - r, z - r), (x + r, y + r, z + r)

    def isEquivalentTo(self, other):
        if type(other) is not SphericalRegion:
            return False
        return areEquivalent(other.center, self.center) and areEquivalent(other.radius, self.radius)

    def __str__(self):
        return f'SphericalRegion({self.center}, {self.radius})'


class HalfSpaceRegion(Region, Intersect):
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
        if isinstance(other, ConvexPolyRegion):
            return intersect_halfspace_convpoly(self, other)
        if isinstance(other, EmptyRegion) or isinstance(other, AllRegion):
            return other.intersect(self)

        raise NotImplementedError("Havent yet filled in the intersection code")

    def __init__(self, point: Vector3D, normal: Vector3D, dist: float = 100.0):
        super().__init__('PlaneCast', point, normal, dist)
        self.point = point
        self.normal = normal
        self.dist = dist
        rot = rotation_to_euler(Vector3D(0, 1, 0), self.normal)
        self.ax_1 = rotate_euler(Vector3D(1, 0, 0), rot)
        self.ax_2 = rotate_euler(Vector3D(0, 0, 1), rot)

    def uniform_point_inner(self) -> Vector3D:
        rx_1 = random.uniform(-self.dist / 2.0, self.dist / 2.0)
        rx_2 = random.uniform(-self.dist / 2.0, self.dist / 2.0)
        rn = random.uniform(0, self.dist)
        plane_point = self.point + rx_1 * self.ax_1 + rx_2 * self.ax_2
        cast_point = plane_point + rn * self.normal
        return cast_point

    def contains_point(self, p):
        return np.dot(self.normal, p - self.point) >= 0

    def to_cuboid_region(self):
        pos = self.point + self.normal * self.dist / 2.0
        orientation = rotation_to_euler(Vector3D(0, 1, 0), self.normal)  # TODO: Confirm this is correct
        return CuboidRegion(pos, orientation, self.dist, self.dist, self.dist)


class ConvexPolyRegion(Region, Intersect, BoundingBox):
    def sample_given_dependencies(self, dep_values):
        return ConvexPolyRegion(dep_values[self.hsi])

    def evaluateInner(self, context):
        hsi = value_in_context(self.hsi, context)
        return ConvexPolyRegion(hsi)

    def intersect(self, other):
        # Cuboid, Halfspace, convexpoly, empty, any, other
        if isinstance(other, CuboidRegion):
            return intersect_cuboid_convpoly(other, self)
        if isinstance(other, HalfSpaceRegion):
            return intersect_halfspace_convpoly(other, self)
        if isinstance(other, ConvexPolyRegion):
            return intersect_convpolys(self, other)
        if isinstance(other, EmptyRegion) or isinstance(other, AllRegion):
            return other.intersect(self)

        raise NotImplementedError

    def __init__(self, hsi: HalfspaceIntersection):
        super().__init__('ConvexPoly', hsi)
        self.hsi = hsi
        convex_hull = ConvexHull(hsi.intersections)
        self.corners = tuple(convex_hull.points[i] for i in convex_hull.vertices)

    def uniform_point_inner(self):
        current_point = self.hsi.interior_point

        for i in range(10):
            random_direction = np.random.normal(size=3)
            random_direction = random_direction / np.linalg.norm(random_direction)

            ts = []
            for hs_ineq in self.hsi.halfspaces:
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

        return Vector3D(*current_point)

    def contains_point(self, point):
        for hs_ineq in self.hsi.halfspaces:
            if np.dot(point, hs_ineq[:-1]) + hs_ineq[-1] > 0:
                return False
        return True

    def getAABB(self):
        cs = np.array(self.corners)
        return np.min(cs, axis=0), np.max(cs, axis=0)


class CuboidRegion(Region, Intersect, BoundingBox):

    def intersect(self, other: Intersect):
        if isinstance(other, CuboidRegion):
            return intersect_cuboid_cuboid(self, other)
        if isinstance(other, HalfSpaceRegion):
            return intersect_cuboid_half_space(self, other)
        if isinstance(other, ConvexPolyRegion):
            return intersect_cuboid_convpoly(self, other)
        if isinstance(other, EmptyRegion) or isinstance(other, AllRegion):
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
        self.corners = tuple(self.position + rotate_euler(Vector3D(*offset), self.orientation)
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
        pt = self.position + rotate_euler(Vector3D(rx, ry, rz), self.orientation)
        return pt

    def getAABB(self):
        xs, ys, zs = zip(*self.corners)
        min_x, max_x = min_and_max(xs)
        min_y, max_y = min_and_max(ys)
        min_z, max_z = min_and_max(zs)
        return (min_x, min_y, min_z), (max_x, max_y, max_z)

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


class PointSetRegion(Region):
    """Region consisting of a set of discrete points.

    No :obj:`~scenic3d.core.object_types.Object` can be contained in a `PointSetRegion`,
    since the latter is discrete. (This may not be true for subclasses, e.g.
    `GridRegion`.)

    Args:
        name (str): name for debugging
        points (iterable): set of points comprising the region
        kd_tree (:obj:`scipy.spatial.KDTree`, optional): k-D tree for the points (one will
          be computed if none is provided)
        orientation (:obj:`~scenic3d.core.vectors.VectorField`, optional): orientation for
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

    return ConvexPolyRegion(hs_intersection)


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

    return ConvexPolyRegion(hs_intersection)


@distributionFunction
def intersect_cuboid_convpoly(r1: CuboidRegion, r2: ConvexPolyRegion) -> Region:
    if r1.contains_object(r2):
        return r2
    if r2.contains_object(r1):
        return r1

    hs_1 = cube_to_hsi(np.array(r1.position), r1.dimensions(), np.array(r1.orientation))
    hs_2 = r2.hsi
    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


@distributionFunction
def intersect_halfspaces(r1: HalfSpaceRegion, r2: HalfSpaceRegion) -> Region:
    c1 = r1.to_cuboid_region()
    c2 = r2.to_cuboid_region()

    hs_1 = cube_to_hsi(np.array(c1.position), c1.dimensions(), np.array(c1.orientation))
    hs_2 = cube_to_hsi(np.array(c2.position), c2.dimensions(), np.array(c2.orientation))

    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


@distributionFunction
def intersect_halfspace_convpoly(r1: HalfSpaceRegion, r2: ConvexPolyRegion) -> Region:
    r1_hs = halfspaces_to_inequalities(np.array([r1.normal]), np.array([r1.point]))
    cp_halfspaces = r2.hsi.halfspaces
    hs_intersection = intersect_hs_ineqs(np.vstack((r1_hs, cp_halfspaces)))

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


@distributionFunction
def intersect_convpolys(r1: ConvexPolyRegion, r2: ConvexPolyRegion) -> Region:
    hs_intersection = intersect_hsis(r1.hsi, r2.hsi)

    if hs_intersection is None:
        return EmptyRegion("empty")

    return ConvexPolyRegion(hs_intersection)


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
    bounds = [(None, None) for _ in range(len(coefficients) - 1)] + [(1e-5, None)]  # Intersection must have non-zero volume
    A = hs_ineqs[:, :-1]
    A_row_norms = np.linalg.norm(A, axis=1).reshape(-1, 1)
    b = -hs_ineqs[:, -1]

    result = linprog(coefficients, A_ub=np.hstack((A, A_row_norms)), b_ub=b, bounds=bounds)

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
