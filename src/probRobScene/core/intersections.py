from typing import Any, Union, Optional, Tuple
import numpy as np
from scipy.linalg import inv

from multimethod import multimethod
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from scipy.spatial.transform import Rotation as R

from probRobScene.core.distributions import distributionFunction
from probRobScene.core.regions import All, Region, Empty, HalfSpace, Cuboid, ConvexPolyhedron, ConvexPolygon3D, Vector3D, Rectangle3D, Convex, contains, LineSeg, Plane, Line
from probRobScene.core.vectors import rotation_to_euler, rotate_euler


# @distributionFunction
@multimethod
def intersect(r1: All, r2) -> Region:
    return r2


@multimethod
def intersect(r1: All, r2: Rectangle3D) -> Region:
    return r2


# @distributionFunction
@multimethod
def intersect(r1, r2: All) -> Region:
    return r1


# @distributionFunction
@multimethod
def intersect(r1: Empty, r2) -> Region:
    return r1


# @distributionFunction
@multimethod
def intersect(r1, r2: Empty) -> Region:
    return r2


# @distributionFunction
@multimethod
def intersect(r1: HalfSpace, r2: Cuboid) -> Region:
    pass


# @distributionFunction
@multimethod
def intersect(r1: HalfSpace, r2: HalfSpace) -> Region:
    hs_intersection = intersect(to_hsi(r1), to_hsi(r2))

    if hs_intersection is None:
        return Empty()

    return ConvexPolyhedron(hs_intersection)


# @distributionFunction
@multimethod
def intersect(r1: Convex, r2: Convex) -> Region:
    hs_intersection = intersect(to_hsi(r1), to_hsi(r2))

    if hs_intersection is None:
        return Empty()

    return ConvexPolyhedron(hs_intersection)


# @distributionFunction
@multimethod
def intersect(r1: Convex, r2: Cuboid) -> Region:
    if contains(r1, r2):
        return r2

    if contains(r2, r1):
        return r1

    hs_intersection = intersect(to_hsi(r1), to_hsi(r2))

    if hs_intersection is None:
        return Empty()

    return ConvexPolyhedron(hs_intersection)


# @distributionFunction
@multimethod
def intersect(r1: Cuboid, r2: Convex) -> Region:
    return intersect(r2, r1)


# @distributionFunction
@multimethod
def intersect(hs_1: HalfspaceIntersection, hs_2: HalfspaceIntersection) -> Optional[HalfspaceIntersection]:
    combined_halfspaces = np.vstack((hs_1.halfspaces, hs_2.halfspaces))
    return intersect_hs_ineqs(combined_halfspaces)


# @distributionFunction
@multimethod
def intersect(c1: Union[ConvexPolygon3D, Rectangle3D], c2: Convex) -> Region:
    projected_hsis = proj_hsi_to_plane(to_hsi(c2), c1.normal, c1.origin)

    if projected_hsis is None:
        return Empty()

    hsi = intersect(projected_hsis, to_hsi(c1))

    if hsi is None:
        return Empty()

    return ConvexPolygon3D(hsi, c1.origin, c1.rot)


@multimethod
def intersect(c1: Convex, c2: Union[ConvexPolygon3D, Rectangle3D]) -> Region:
    return intersect(c2, c1)


# @distributionFunction
@multimethod
def intersect(r1: Rectangle3D, r2: Rectangle3D) -> Region:
    # if r1.contains_object(r2):
    #     return r2
    # if r2.contains_object(r1):
    #     return r1
    #
    # if r1.normal == r2.normal and r1.origin != r2.origin:  # Parallel
    #     return EmptyRegion("empty")

    # The infinite line between the two planes the rectangles lie on
    line = intersect(Plane(r1.origin, r1.normal), Plane(r2.origin, r2.normal))

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
    return LineSeg(start, end)


# @distributionFunction
@multimethod
def intersect(r1: LineSeg, r2: ConvexPolyhedron) -> Region:
    # Q: How to check for no overlap?
    halfspaces = to_hsi(r2).halfspaces

    l_origin = r1.start
    start_end_dist = np.linalg.norm(r1.end - r1.start)
    l_dir = (r1.end - r1.start) / start_end_dist

    # The original line segment acts as initial parameters
    t_max = start_end_dist
    t_min = 0.0

    for halfspace in halfspaces:
        A, b = halfspace[:3], halfspace[3]

        if np.dot(A, r1.start) > -b and np.dot(A, r1.end) > -b:  # Line and convex poly do not intersect
            return Empty()

        dir_align = np.dot(A, l_dir)
        point_align = np.dot(A, l_origin)
        if np.abs(dir_align) < 1e-8:  # Orthogonal
            continue
        if dir_align > 0:  # Pointing in same direction
            t_max = np.minimum(t_max, (-b - point_align) / dir_align)
        else:
            t_min = np.maximum(t_min, (-b - point_align) / dir_align)

    return LineSeg(l_origin + t_min * l_dir, l_origin + t_max * l_dir)


# @distributionFunction
@multimethod
def intersect(p1: Plane, p2: Plane) -> Union[Empty, Line]:
    if p1.normal == p2.normal and p1.origin == p2.origin:
        return p1

    if p1.normal == p2.normal:
        return Empty()

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

    return Line(line_origin, line_dir)


@distributionFunction
def intersect_many(*regions) -> Region:
    intersection = All()
    for r in regions:
        intersection = intersect(intersection, r)
    return intersection


@distributionFunction
def intersect_hs_ineqs(hs_ineqs: np.ndarray) -> Optional[HalfspaceIntersection]:
    fsp = feasible_point(hs_ineqs)
    if fsp is None:
        return None
    return HalfspaceIntersection(hs_ineqs, fsp)


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


# @distributionFunction
@multimethod
def to_hsi(r: HalfSpace) -> HalfspaceIntersection:
    pos: Vector3D = r.point + r.normal * (r.dist / 2.0)
    return to_hsi(Cuboid(pos, r.rot, *(r.dist * np.ones(3))))


# @distributionFunction
@multimethod
def to_hsi(r: Union[ConvexPolyhedron, ConvexPolygon3D]) -> HalfspaceIntersection:
    return r.hsi


# @distributionFunction
@multimethod
def to_hsi(r: Cuboid) -> HalfspaceIntersection:
    hs_norms, hs_origins = cube_to_normals(np.array(r.position), np.array(r.dimensions), np.array(r.orientation))
    hs_ineqs = halfspaces_to_inequalities(hs_norms, hs_origins)
    in_point = np.array(r.position)
    return HalfspaceIntersection(hs_ineqs, in_point)


# @distributionFunction
@multimethod
def to_hsi(r: Rectangle3D) -> HalfspaceIntersection:
    hs_origins = np.array([[r.width / 2.0, 0], [0, r.length / 2.0], [-r.width / 2.0, 0], [0, -r.length / 2.0]])
    hs_norms = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])

    hs_ineqs = halfspaces_to_inequalities(hs_norms, hs_origins)
    return HalfspaceIntersection(hs_ineqs, np.array([0.0, 0.0]))


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
