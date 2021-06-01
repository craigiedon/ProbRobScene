from abc import ABC
from dataclasses import dataclass
from typing import List

from multimethod import multimethod
from scipy.spatial.qhull import HalfspaceIntersection


# from probRobScene.core.vectors import Vector3D
class LazilyEvaluable(ABC):
    pass


def evaluate_inner(x: LazilyEvaluable, context):
    var_names = vars(x).keys()

    return
    # look for all the constructor properties of the dataclass?


@dataclass
class Vector3D:
    x: float
    y: float
    z: float


class Region(ABC):
    pass


@dataclass
class All(Region):
    pass


@dataclass
class Empty(Region):
    pass


@dataclass
class Spherical(Region):
    center: Vector3D
    radius: float


@dataclass
class Intersection(Region):
    regions: List[Region]


@dataclass
class HalfSpace(Region):
    point: Vector3D
    normal: Vector3D
    dist: float = 100.0


@dataclass
class ConvexPolyhedron(Region):
    hsi: HalfspaceIntersection


@dataclass
class ConvexPolygon3D(Region):
    pass


@dataclass
class Cuboid(Region):
    position: Vector3D
    orientation: Vector3D
    width: float
    length: float
    height: float


@dataclass
class Rectangle3D(Region):
    width: float
    length: float
    origin: Vector3D
    rot: Vector3D


@dataclass
class Plane(Region):
    origin: Vector3D
    normal: Vector3D


@dataclass
class Line(Region):
    origin: Vector3D
    direction: Vector3D


@dataclass
class LineSeg(Region):
    start: Vector3D
    end: Vector3D


@dataclass
class PointSet(Region):
    points: List[Vector3D]


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
