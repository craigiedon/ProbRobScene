"""Utility functions for geometric computation."""
import math

import numpy as np

from probRobScene.core.distributions import distributionFunction, monotonicDistributionFunction
from probRobScene.core.utils import min_and_max


@distributionFunction
def sin(x):
    return math.sin(x)


@distributionFunction
def cos(x):
    return math.cos(x)


@distributionFunction
def normalize(x):
    return x / np.linalg.norm(x)


@monotonicDistributionFunction
def hypot(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@monotonicDistributionFunction
def max(*args):
    return __builtins__['max'](*args)


@monotonicDistributionFunction
def min(*args):
    return __builtins__['min'](*args)


def averageVectors(a, b, weight=0.5):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    aw, bw = 1.0 - weight, weight
    return ax * aw + bx * bw, ay * aw + by * bw


def rotate_vector(vector, angle):
    x, y = vector
    c, s = cos(angle), sin(angle)
    return (c * x) - (s * y), (s * x) + (c * y)


def radialToCartesian(point, radius, heading):
    angle = heading + (math.pi / 2.0)
    rx, ry = radius * cos(angle), radius * sin(angle)
    return point[0] + rx, point[1] + ry


def headingOfSegment(pointA, pointB):
    ax, ay = pointA
    bx, by = pointB
    return math.atan2(by - ay, bx - ax) - (math.pi / 2.0)


def viewAngleToPoint(point, base, heading):
    x, y = base
    ox, oy = point
    a = math.atan2(oy - y, ox - x) - (heading + (math.pi / 2.0))
    if a < -math.pi:
        a += math.tau
    elif a > math.pi:
        a -= math.tau
    assert -math.pi <= a <= math.pi
    return a


def apparentHeadingAtPoint(point, heading, base):
    x, y = base
    ox, oy = point
    a = (heading + (math.pi / 2.0)) - math.atan2(oy - y, ox - x)
    if a < -math.pi:
        a += math.tau
    elif a > math.pi:
        a -= math.tau
    assert -math.pi <= a <= math.pi
    return a


def cuboids_intersect(cuboid_a, cuboid_b):
    # Quick bounding circle check
    dx, dy, dz = cuboid_a.position - cuboid_b.position
    rr = cuboid_a.radius + cuboid_b.radius
    if (dx * dx) + (dy * dy) + (dz * dz) > (rr * rr):
        return False

    if cube_edge_separates(cuboid_a, cuboid_b) or cube_edge_separates(cuboid_b, cuboid_a):
        return False

    return True


def cube_edge_separates(cuboid_a, cuboid_b):
    from probRobScene.core.vectors import reverse_euler
    from probRobScene.core.vectors import rotate_euler_v3d
    base = cuboid_a.position.to_vector_3d()

    # A reversal of the first one's rotation
    rot = reverse_euler(cuboid_a.orientation)

    # Take each of cube_b's corners, then get the relative vector from the position of cube_a to each of these corners
    # Then take each of these relative vectors and undo the rotation of A
    rc = [rotate_euler_v3d(corner - base, rot) for corner in cuboid_b.corners]
    xs, ys, zs = zip(*rc)

    min_x, max_x = min_and_max(xs)
    min_y, max_y = min_and_max(ys)
    min_z, max_z = min_and_max(zs)

    if max_x < -cuboid_a.hw or min_x > cuboid_a.hw:
        return True
    if max_y < -cuboid_a.hl or min_y > cuboid_a.hl:
        return True
    if max_z < -cuboid_a.hh or min_z > cuboid_a.hh:
        return True

    return False
