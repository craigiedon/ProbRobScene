import abc
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull, HalfspaceIntersection

from scenic3d.core.regions import CuboidRegion, HalfSpaceRegion, EmptyRegion, Region, ConvexPolyRegion


class Intersect(abc.ABC):
    @abc.abstractmethod
    def intersect(self, other):
        raise NotImplementedError()


def intersect_cuboid_cuboid(c1: CuboidRegion, c2: CuboidRegion) -> Region:
    if c1.contains_object(c2):
        return c2
    if c2.contains_object(c1):
        return c1

    hs_1 = cube_to_hsi(c1.position, c1.dimensions(), c1.orientation)
    hs_2 = cube_to_hsi(c2.position, c2.dimensions(), c2.orientation)
    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


def intersect_cuboid_half_space(c1: CuboidRegion, c2: HalfSpaceRegion) -> Region:
    if c2.contains_object(c1):
        return c1

    # Otherwise we have to actually do the work:
    # So turn the cuboid into some halfspace inequalities, turn the halfspace class into the correct form, find a feasible point, and smoosh them all together
    hs_1 = cube_to_hsi(c1.position, c1.dimensions(), c1.orientation).halfspaces
    hs_2 = halfspaces_to_inequalities(c2.normal, c2.point)

    hs_intersection = intersect_hs_ineqs(np.vstack(hs_1, hs_2))

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


def intersect_cuboid_convpoly(r1: CuboidRegion, r2: ConvexPolyRegion) -> Region:
    if r1.contains_object(r2):
        return r2
    if r2.contains_object(r1):
        return r1

    hs_1 = cube_to_hsi(r1.position, r1.dimensions(), r1.orientation)
    hs_2 = r2.hsi
    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


def intersect_halfspaces(r1: HalfSpaceRegion, r2: HalfSpaceRegion) -> Region:
    c1 = r1.to_cuboid_region()
    c2 = r2.to_cuboid_region()

    hs_1 = cube_to_hsi(c1.position, c1.dimensions(), c1.orientation)
    hs_2 = cube_to_hsi(c2.position, c2.dimensions(), c2.orientation)

    hs_intersection = intersect_hsis(hs_1, hs_2)

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


def intersect_halfspace_convpoly(r1: HalfSpaceRegion, r2: ConvexPolyRegion) -> Region:
    r1_hs = halfspaces_to_inequalities(np.array([r1.normal]), np.array([r1.point]))
    cp_halfspaces = r2.hsi.halfspaces
    hs_intersection = intersect_hs_ineqs(np.vstack((r1_hs, cp_halfspaces)))

    if hs_intersection is None:
        return EmptyRegion("Empty")

    return ConvexPolyRegion(hs_intersection)


def intersect_convpolys(r1: ConvexPolyRegion, r2: ConvexPolyRegion) -> Region:
    hs_intersection = intersect_hsis(r1.hsi, r2.hsi)

    if hs_intersection is None:
        return EmptyRegion("empty")

    return ConvexPolyRegion(hs_intersection)


"""
class ExRegA(Intersect):
    def intersect(self, other):
        if isinstance(other, ExRegA):
            return 1
        if isinstance(other, ExRegB):
            return 2
        if isinstance(other, ExRegC):
            return 3


class ExRegB(Intersect):
    def intersect(self, other: Intersect):
        if isinstance(other, ExRegB):
            return 1
        if isinstance(other, ExRegC):
            return 2
        if isinstance(other, ExRegA):
            return other.intersect(self)
        raise ValueError("Unsupported type")


class ExRegC(Intersect):
    def intersect(self, other):
        pass
"""


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
    bounds = [(None, None) for i in range(len(coefficients) - 1)] + [(1e-5, None)]  # Intersection must have non-zero volume
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
