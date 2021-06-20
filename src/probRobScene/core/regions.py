import itertools
from abc import ABC
from dataclasses import dataclass
import random
from inspect import signature
from typing import List, Any, Tuple

from multimethod import multimethod
from scipy.spatial.qhull import HalfspaceIntersection, ConvexHull
import math
import numpy as np

from probRobScene.core.distributions import Samplable
from probRobScene.core.utils import min_and_max
from probRobScene.core.vectors import rotate_euler_v3d, rotation_to_euler, Vector3D, VectorDistribution, reverse_euler


class LazilyEvaluable(ABC):
    pass


class Region(Samplable):
    pass


class Intersect(ABC):
    pass


class Convex(ABC):
    pass


class PointInRegionDistribution(VectorDistribution):
    """Uniform distribution over points in a Region"""

    def sample_given_dependencies(self, dep_values) -> Any:
        return uniform_point_inner(dep_values[self.region])

    def support_interval(self):
        return None, None

    def __init__(self, region):
        self.region = region

    def __str__(self):
        return f'PointIn({self.region})'


@dataclass(frozen=True)
class All(Region):
    pass


@dataclass(frozen=True)
class Empty(Region):
    pass


@dataclass(frozen=True, eq=False)
class Spherical(Region):
    center: Vector3D
    radius: float

    @property
    def circumsphere(self):
        return self.center, self.radius


@dataclass(frozen=True, eq=False)
class Intersection(Region):
    regions: List[Region]

    def __post_init__(self):
        assert len(self.regions) > 1


@dataclass(frozen=True, eq=False)
class HalfSpace(Region, Convex):
    point: Vector3D
    normal: Vector3D
    dist: float = 100.0

    def __post_init__(self):
        assert np.isclose(np.linalg.norm(self.normal), 1.0), f"Normal vector {self.normal} has no-unit magnitude"

    @property
    def rot(self):
        return rotation_to_euler(Vector3D(0, 0, 1), self.normal)


@dataclass(frozen=True, eq=False)
class ConvexPolyhedron(Region, Convex):
    hsi: HalfspaceIntersection

    @property
    def corners(self):
        convex_hull = ConvexHull(self.hsi.intersections)
        return tuple(convex_hull.points[i] for i in convex_hull.vertices)


@dataclass(frozen=True, eq=False)
class ConvexPolygon3D(Region, Convex):
    hsi: HalfspaceIntersection
    origin: Vector3D
    rot: Vector3D

    @property
    def rev_rot(self) -> Vector3D:
        return reverse_euler(self.rot)

    @property
    def normal(self) -> Vector3D:
        return rotate_euler_v3d(Vector3D(0, 0, 1), self.rot)


@dataclass(frozen=True, eq=False)
class Cuboid(Region, Convex):
    position: Vector3D
    orientation: Vector3D
    width: float
    length: float
    height: float

    @property
    def dimensions(self):
        return np.array([self.width, self.length, self.height])

    @property
    def hw(self): return self.width / 2.0

    @property
    def hl(self): return self.length / 2.0

    @property
    def hh(self): return self.height / 2.0

    @property
    def radius(self): return np.linalg.norm((self.hw, self.hl, self.hh))

    @property
    def corners(self):
        return tuple(self.position + rotate_euler_v3d(Vector3D(*offset), self.orientation)
                     for offset in itertools.product((self.hw, -self.hw), (self.hl, -self.hl), (self.hh, -self.hh)))

    @property
    def circumcircle(self):
        return self.position, self.radius


@dataclass(frozen=True, eq=False)
class Rectangle3D(Region, Convex):
    width: float
    length: float
    origin: Vector3D
    rot: Vector3D

    @property
    def rev_rot(self) -> Vector3D:
        return reverse_euler(self.rot)

    @property
    def normal(self) -> Vector3D:
        return rotate_euler_v3d(Vector3D(0, 0, 1), self.rot)

    @property
    def w_ax(self):
        return rotate_euler_v3d(Vector3D(1, 0, 0), self.rot)

    @property
    def l_ax(self):
        return rotate_euler_v3d(Vector3D(0, 1, 0), self.rot)


@dataclass(frozen=True, eq=False)
class Plane(Region):
    origin: Vector3D
    normal: Vector3D


@dataclass(frozen=True, eq=False)
class Line(Region):
    origin: Vector3D
    direction: Vector3D


@dataclass(frozen=True, eq=False)
class LineSeg(Region):
    start: Vector3D
    end: Vector3D


@dataclass(frozen=True, eq=False)
class PointSet(Region):
    points: List[Vector3D]


@multimethod
def uniform_point_inner(r: Spherical):
    x, y, z = r.center
    u = 2.0 * random.random() - 1.0
    phi = 2.0 * math.pi * random.random()
    r = random.random() ** (1 / 3.)
    x_offset = r * np.cos(phi) * (1 - u ** 2) ** 0.5
    y_offset = r * np.sin(phi) * (1 - u ** 2) ** 0.5
    z_offset = r * u
    pt = Vector3D(x + x_offset, y + y_offset, z + z_offset)
    return pt
    pass


@multimethod
def uniform_point_inner(r: HalfSpace):
    untransformed_point = Vector3D(*np.random.uniform(0, r.dist, 3))
    return rotate_euler_v3d(untransformed_point, r.rot) + r.point


@multimethod
def uniform_point_inner(r: ConvexPolyhedron):
    return Vector3D(*hit_and_run(r.hsi))


@multimethod
def uniform_point_inner(r: ConvexPolygon3D):
    random_point_flat = Vector3D(*hit_and_run(r.hsi), 0)
    return rotate_euler_v3d(random_point_flat, r.rot) + r.origin


@multimethod
def uniform_point_inner(r: Cuboid):
    hw, hl, hh = r.hw, r.hl, r.hh
    rx = random.uniform(-hw, hw)
    ry = random.uniform(-hl, hl)
    rz = random.uniform(-hh, hh)
    pt = r.position + rotate_euler_v3d(Vector3D(rx, ry, rz), r.orientation)
    return pt


@multimethod
def uniform_point_inner(r: Rectangle3D):
    x = random.uniform(-r.width / 2.0, r.width / 2.0)
    y = random.uniform(-r.length / 2.0, r.length / 2.0)
    flat_point = Vector3D(x, y, 0)
    return rotate_euler_v3d(flat_point, r.rot) + r.origin


@multimethod
def uniform_point_inner(r: Plane):
    # We don't want arbitrarily chosen axis to be exactly aligned with normal
    if 1.0 - np.abs(r.normal, Vector3D(1.0, 0.0, 0.0)) >= 1e-5:
        u = np.cross(r.normal, Vector3D(1.0, 0.0, 0.0))
    else:
        u = np.cross(r.normal, Vector3D(0.0, 1.0, 0.0))

    v = np.cross(r.normal, u)

    a = np.random.uniform(-r.dist / 2.0, r.dist / 2.0)
    b = np.random.uniform(-r.dist / 2.0, r.dist / 2.0)

    offset = a * u + b * v
    return r.origin + offset


@multimethod
def uniform_point_inner(r: Line):
    t = np.random.uniform(-r.dist / 2.0, r.dist / 2.0)
    return r.origin + t * r.direction


@multimethod
def uniform_point_inner(r: LineSeg):
    t = random.uniform(0.0, 1.0)
    return (1.0 - t) * r.start + t * r.end


@multimethod
def uniform_point_inner(r: PointSet):
    return Vector3D(*random.choice(r.points))


@multimethod
def contains(r: Region, o: Any) -> bool:
    contains_points = [contains_point(r,c) for c in o.corners]
    return all(contains_points)


@multimethod
def contains(r: Region, v: Vector3D) -> bool:
    return contains_point(r, v)


@multimethod
def contains_point(r: All, point: Vector3D) -> bool:
    return True


@multimethod
def contains_point(r: Empty, point: Vector3D) -> bool:
    return False


@multimethod
def contains_point(r: Spherical, point: Vector3D) -> bool:
    return point.distanceTo(r.center) <= r.radius


@multimethod
def contains_point(r: HalfSpace, point: Vector3D) -> bool:
    offset = point - r.point
    dp = np.dot(r.normal, offset)
    return dp >= 0


@multimethod
def contains_point(r: ConvexPolyhedron, point: Vector3D) -> bool:
    for hs_ineq in r.hsi.halfspaces:
        if np.dot(point, hs_ineq[:-1]) + hs_ineq[-1] > 0:
            return False
    return True


@multimethod
def contains_point(r: ConvexPolygon3D, point: Vector3D) -> bool:
    flat_point = rotate_euler_v3d(point - r.origin, r.rev_rot)
    if np.abs(flat_point[-1]) > 1e-8:
        return False

    for hs_ineq in r.hsi.halfspaces:
        if np.dot(flat_point[:-1], hs_ineq[:-1]) + hs_ineq[-1] > 0:
            return False
    return True


@multimethod
def contains_point(r: Cuboid, point: Vector3D) -> bool:
    diff = point - r.position
    x, y, z = rotate_euler_v3d(diff, reverse_euler(r.orientation))
    return abs(x) <= r.hw and abs(y) <= r.hl and abs(z) <= r.hh


@multimethod
def contains_point(r: Plane, point: Vector3D) -> bool:
    return np.abs(np.dot(r.normal - r.origin, point - r.origin)) <= 1e-8


@multimethod
def contains_point(r: Rectangle3D, point: Vector3D) -> bool:
    flat_point = rotate_euler_v3d(point - r.origin, r.rev_rot)
    return (np.abs(flat_point.x) <= r.width / 2.0 and
            np.abs(flat_point.y) <= r.length / 2.0 and
            np.abs(flat_point.z) <= 1e-8)  # "Roughly equals" zero


@multimethod
def contains_point(r: Line, point: Vector3D) -> bool:
    pv = point - r.origin
    pv = pv / np.linalg.norm(pv)
    dp_unsigned = np.abs(np.dot(pv, r.direction))

    # Looking for dot product to be +-1.0 (with wiggle room)
    return 1.0 - dp_unsigned <= 1e-8


@multimethod
def contains_point(r: LineSeg, point: Vector3D) -> bool:
    s_e_dir = r.end - r.start / np.linalg.norm(r.end - r.start)

    s_p_dir = point - r.start / np.linalg.norm(point - r.start)
    p_e_dir = r.end - point / np.linalg.norm(r.end - point)

    dp1 = np.dot(s_e_dir, s_p_dir)
    dp2 = np.dot(s_e_dir, p_e_dir)

    return np.abs(dp1 - 1.0) <= 1e-8 and np.abs(dp2 - 1.0) <= 1e-8  # A little bit of wiggle room to be a small epsilon of rounding off of the line


@multimethod
def contains_point(r: PointSet, point: Vector3D) -> bool:
    distance, location = r.kd_tree.query(point)
    return distance <= r.tolerance


@multimethod
def AABB(r: Spherical) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z = r.center
    return np.array((x, y, z)) - r.radius, np.array((x, y, z)) + r.radius


@multimethod
def AABB(r: ConvexPolyhedron) -> Tuple[np.ndarray, np.ndarray]:
    cs = np.array(r.corners)
    return np.min(cs, axis=0), np.max(cs, axis=0)


@multimethod
def AABB(r: Cuboid) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys, zs = zip(*r.corners)
    min_x, max_x = min_and_max(xs)
    min_y, max_y = min_and_max(ys)
    min_z, max_z = min_and_max(zs)
    return (np.array((min_x, min_y, min_z))), (np.array((max_x, max_y, max_z)))


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