from abc import ABC
from dataclasses import dataclass
import random
from typing import List

from multimethod import multimethod
from scipy.spatial.qhull import HalfspaceIntersection
import math
import numpy as np


# from probRobScene.core.vectors import Vector3D
from probRobScene.core.geometry import cos, sin
from probRobScene.core.regions import hit_and_run
from probRobScene.core.vectors import rotate_euler_v3d


class LazilyEvaluable(ABC):
    pass


def evaluate_inner(x: LazilyEvaluable, context):
    var_names = vars(x).keys()

    return
    # look for all the constructor properties of the dataclass?


@dataclass(frozen=True, eq=False)
class Vector3D:
    x: float
    y: float
    z: float


class Region(ABC):
    pass


@dataclass(frozen=True, eq=False)
class All(Region):
    pass


@dataclass(frozen=True, eq=False)
class Empty(Region):
    pass


@dataclass(frozen=True, eq=False)
class Spherical(Region):
    center: Vector3D
    radius: float


@dataclass(frozen=True, eq=False)
class Intersection(Region):
    regions: List[Region]


@dataclass(frozen=True, eq=False)
class HalfSpace(Region):
    point: Vector3D
    normal: Vector3D
    dist: float = 100.0


@dataclass(frozen=True, eq=False)
class ConvexPolyhedron(Region):
    hsi: HalfspaceIntersection


@dataclass(frozen=True, eq=False)
class ConvexPolygon3D(Region):
    pass


@dataclass(frozen=True, eq=False)
class Cuboid(Region):
    position: Vector3D
    orientation: Vector3D
    width: float
    length: float
    height: float


@dataclass(frozen=True, eq=False)
class Rectangle3D(Region):
    width: float
    length: float
    origin: Vector3D
    rot: Vector3D


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
    x_offset = r * cos(phi) * (1 - u ** 2) ** 0.5
    y_offset = r * sin(phi) * (1 - u ** 2) ** 0.5
    z_offset = r * u
    pt = Vector3D(x + x_offset, y + y_offset, z + z_offset)
    return pt
    pass


@multimethod
def uniform_point_inner(r: HalfSpace):
    untransformed_point = Vector3D(np.random.uniform(0, r.dist, 3))
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
    return orient_along_region(r, Vector(*random.choice(r.points)))


@multimethod
def print_thing(r: Region):
    print("Congrats! Its a region")


@multimethod
def print_thing(f: float):
    print("Just a number...I guess thats still fine...")


if __name__ == "__main__":
    r = All()
    s = Line(Vector3D(1, 1, 1), Vector3D(2, 2, 2))
    f = 20.0

    print(evaluate_inner(r, []))
    print(evaluate_inner(s, []))

    print_thing(r)
    print_thing(s)
    print_thing(f)
