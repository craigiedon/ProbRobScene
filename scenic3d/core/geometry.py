"""Utility functions for geometric computation."""
import abc
import math
import warnings

import numpy as np
import pypoly2tri
import shapely.geometry
import shapely.ops

import scenic3d.core.utils as utils
from scenic3d.core.distributions import (needsSampling, distributionFunction,
                                         monotonicDistributionFunction)
from scenic3d.core.lazy_eval import needs_lazy_evaluation


@distributionFunction
def sin(x):
    return math.sin(x)


@distributionFunction
def cos(x):
    return math.cos(x)


@monotonicDistributionFunction
def hypot(x, y, z):
    return math.hypot(x, y, z)


@monotonicDistributionFunction
def max(*args):
    return __builtins__['max'](*args)


@monotonicDistributionFunction
def min(*args):
    return __builtins__['min'](*args)


def normalizeAngle(angle):
    while angle > math.pi:
        angle -= math.tau
    while angle < -math.pi:
        angle += math.tau
    assert -math.pi <= angle <= math.pi
    return angle


def addVectors(a, b):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    return ax + bx, ay + by


def subtractVectors(a, b):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    return ax - bx, ay - by


def averageVectors(a, b, weight=0.5):
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    aw, bw = 1.0 - weight, weight
    return ax * aw + bx * bw, ay * aw + by * bw


def rotateVector(vector, angle):
    x, y = vector
    c, s = cos(angle), sin(angle)
    return (c * x) - (s * y), (s * x) + (c * y)


def min_and_max(iterable):
    min_v = float('inf')
    max_v = float('-inf')
    for val in iterable:
        if val < min_v:
            min_v = val
        if val > max_v:
            max_v = val
    return min_v, max_v


def radialToCartesian(point, radius, heading):
    angle = heading + (math.pi / 2.0)
    rx, ry = radius * cos(angle), radius * sin(angle)
    return (point[0] + rx, point[1] + ry)


def positionRelativeToPoint(point, heading, offset):
    ro = rotateVector(offset, heading)
    return addVectors(point, ro)


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
    assert -math.pi <= a and a <= math.pi
    return a


def apparentHeadingAtPoint(point, heading, base):
    x, y = base
    ox, oy = point
    a = (heading + (math.pi / 2.0)) - math.atan2(oy - y, ox - x)
    if a < -math.pi:
        a += math.tau
    elif a > math.pi:
        a -= math.tau
    assert -math.pi <= a and a <= math.pi
    return a


def circumcircleOfAnnulus(center, heading, angle, minDist, maxDist):
    m = (minDist + maxDist) / 2.0
    g = (maxDist - minDist) / 2.0
    h = m * math.sin(angle / 2.0)
    h2 = h * h
    d = math.sqrt(h2 + (m * m))
    r = math.sqrt(h2 + (g * g))
    return radialToCartesian(center, d, heading), r


def pointIsInCone(point, base, heading, angle):
    va = viewAngleToPoint(point, base, heading)
    return (abs(va) <= angle / 2.0)


def polygonUnion(polys, tolerance=0.05, holeTolerance=0.002):
    union = shapely.ops.unary_union(list(polys))
    assert union.is_valid, union
    if tolerance > 0:
        if isinstance(union, shapely.geometry.MultiPolygon):
            polys = [cleanPolygon(poly, tolerance, holeTolerance) for poly in union]
            union = shapely.ops.unary_union(polys)
        else:
            union = cleanPolygon(union, tolerance, holeTolerance)
    assert union.is_valid, union
    return union


def cleanPolygon(poly, tolerance, holeTolerance):
    exterior = cleanChain(poly.exterior.coords, tolerance)
    assert len(exterior) >= 3
    ints = []
    for interior in poly.interiors:
        interior = cleanChain(interior.coords, tolerance)
        if len(interior) >= 3:
            hole = shapely.geometry.Polygon(interior)
            if hole.area > holeTolerance:
                ints.append(interior)
    newPoly = shapely.geometry.Polygon(exterior, ints)
    assert newPoly.is_valid, newPoly
    return newPoly


def cleanChain(chain, tolerance, angleTolerance=0.008):
    if len(chain) <= 2:
        return chain
    closed = (tuple(chain[0]) == tuple(chain[-1]))
    tol2 = tolerance * tolerance
    # collapse hooks
    chain = np.array(chain)
    a, b = chain[-1], chain[0]
    newChain = []
    for c in chain[1:]:
        dx, dy = c[0] - a[0], c[1] - a[1]
        if (dx * dx) + (dy * dy) > tol2:
            newChain.append(b)
            a = b
            b = c
        else:
            b = c
    if len(newChain) == 0:
        return newChain
    newChain.append(newChain[0] if closed else c)
    return newChain


#: Whether to warn when falling back to pypoly2tri for triangulation
givePP2TWarning = True


class TriangulationError(RuntimeError):
    """Signals that the installed triangulation libraries are insufficient.

    Specifically, raised when pypoly2tri hits the recursion limit trying to
    triangulate a large polygon.
    """
    pass


def triangulatePolygon(polygon):
    """Triangulate the given Shapely polygon.

    Note that we can't use ``shapely.ops.triangulate`` since it triangulates
    point sets, not polygons (i.e., it doesn't respect edges). We need an
    algorithm for triangulation of polygons with holes (it doesn't need to be a
    Delaunay triangulation).

    We currently use the GPC library (wrapped by the ``Polygon3`` package) if it is
    installed. Since it is not free for commercial use, we don't require it as a
    dependency, falling back on the BSD-compatible ``pypoly2tri`` as needed. In this
    case we issue a warning, since GPC is more robust and handles large polygons.
    The warning can be disabled by setting `givePP2TWarning` to ``False``.

    Args:
        polygon (shapely.geometry.Polygon): Polygon to triangulate.

    Returns:
        A list of disjoint (except for edges) triangles whose union is the
        original polygon.
    """
    try:
        import Polygon
        return triangulatePolygon_gpc(polygon)
    except ImportError:
        pass
    if givePP2TWarning:
        warnings.warn('Using pypoly2tri for triangulation; for non-commercial use, consider'
                      ' installing the faster Polygon3 library (pip install Polygon3)')
    return triangulatePolygon_pypoly2tri(polygon)


def triangulatePolygon_pypoly2tri(polygon):
    polyline = []
    for c in polygon.exterior.coords[:-1]:
        polyline.append(pypoly2tri.shapes.Point(c[0], c[1]))
    cdt = pypoly2tri.cdt.CDT(polyline)
    for i in polygon.interiors:
        polyline = []
        for c in i.coords[:-1]:
            polyline.append(pypoly2tri.shapes.Point(c[0], c[1]))
        cdt.AddHole(polyline)
    try:
        cdt.Triangulate()
    except RecursionError:  # polygon too big for pypoly2tri
        raise TriangulationError('pypoly2tri unable to triangulate large polygon; for '
                                 'non-commercial use, try "pip install Polygon3"')

    triangles = list()
    for t in cdt.GetTriangles():
        triangles.append(shapely.geometry.Polygon([
            t.GetPoint(0).toTuple(),
            t.GetPoint(1).toTuple(),
            t.GetPoint(2).toTuple()
        ]))
    return triangles


def triangulatePolygon_gpc(polygon):
    import Polygon as PolygonLib
    poly = PolygonLib.Polygon(polygon.exterior.coords)
    for interior in polygon.interiors:
        poly.addContour(interior.coords, True)
    tristrips = poly.triStrip()
    triangles = []
    for strip in tristrips:
        a, b = strip[:2]
        for c in strip[2:]:
            tri = shapely.geometry.Polygon((a, b, c))
            triangles.append(tri)
            a = b
            b = c
    return triangles


def plotPolygon(polygon, plt, style='r-'):
    def plotRing(ring):
        x, y = ring.xy
        plt.plot(x, y, style)

    if isinstance(polygon, shapely.geometry.MultiPolygon):
        polygons = polygon
    else:
        polygons = [polygon]
    for polygon in polygons:
        plotRing(polygon.exterior)
        for ring in polygon.interiors:
            plotRing(ring)


# class RotatedRectangle:
#     """mixin providing collision detection for rectangular objects and regions"""
#
#     def containsPoint(self, point):
#         diff = point - self.position.toVector()
#         x, y = diff.rotatedBy(-self.heading)
#         return abs(x) <= self.hw and abs(y) <= self.hh
#
#     def intersects(self, rect):
#         if not isinstance(rect, RotatedRectangle):
#             raise RuntimeError(f'tried to intersect RotatedRectangle with {type(rect)}')
#         # Quick check by bounding circles
#         dx, dy = rect.position.toVector() - self.position.toVector()
#         rr = self.radius + rect.radius
#         if (dx * dx) + (dy * dy) > (rr * rr):
#             return False
#         # Check for separating line parallel to our edges
#         if self.edgeSeparates(self, rect):
#             return False
#         # Check for separating line parallel to rect's edges
#         if self.edgeSeparates(rect, self):
#             return False
#         return True
#
#     @staticmethod
#     def edgeSeparates(rectA, rectB):
#         """Whether an edge of rectA separates it from rectB"""
#         base = rectA.position.toVector()
#         rot = -rectA.heading
#         rc = [(corner - base).rotatedBy(rot) for corner in rectB.corners]
#         x, y = zip(*rc)
#         minx, maxx = min_and_max(x)
#         miny, maxy = min_and_max(y)
#         if maxx < -rectA.hw or rectA.hw < minx:
#             return True
#         if maxy < -rectA.hh or rectA.hh < miny:
#             return True
#         return False
#
#     @property
#     def polygon(self):
#         if any(needsSampling(c) or needsLazyEvaluation(c) for c in self.corners):
#             return None  # can only convert fixed Regions to Polygons
#         return self.getConstantPolygon()
#
#     @utils.cached
#     def getConstantPolygon(self):
#         assert not any(needsSampling(c) or needsLazyEvaluation(c) for c in self.corners)
#         # TODO refactor???
#         corners = [(x, y) for x, y in self.corners]  # convert Vectors to tuples
#         return shapely.geometry.Polygon(corners)


class Polygonable(abc.ABC):
    @abc.abstractmethod
    def polygon(self):
        pass


# class Intersectable(abc.ABC):
#     @abc.abstractmethod
#     def intersets(self):
#         pass

def cuboid_contains_point(obj, point):
    from scenic3d.core.vectors import rotate_euler
    from scenic3d.core.vectors import reverse_euler
    diff = point - obj.position
    x, y, z = rotate_euler(diff, reverse_euler(obj.orientation))
    return abs(x) <= obj.hw and abs(y) <= obj.hl and abs(z) <= obj.hh


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
    from scenic3d.core.vectors import reverse_euler
    from scenic3d.core.vectors import rotate_euler
    base = cuboid_a.position.to_vector_3d()

    # A reversal of the first one's rotation
    rot = reverse_euler(cuboid_a.orientation)

    # Take each of cube_b's corners, then get the relative vector from the position of cube_a to each of these corners
    # Then take each of these relative vectors and undo the rotation of A
    rc = [rotate_euler(corner - base, rot) for corner in cuboid_b.corners]
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


def get_constant_polygon(cuboid):
    assert not any(needsSampling(c) or needs_lazy_evaluation(c) for c in cuboid.corners)
    corners = [(x, y, z) for x, y, z in cuboid.corners]  # convert Vectors to tuples
    return shapely.geometry.Polygon(corners)
