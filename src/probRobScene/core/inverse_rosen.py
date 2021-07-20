import time
from itertools import permutations, product
from typing import Optional, Any, Tuple, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.spatial import HalfspaceIntersection, ConvexHull, convex_hull_plot_2d

from probRobScene.core.good_lattice_point import centred_normalize, good_lattice_point
from probRobScene.core.intersections import feasible_point, project_to_plane_intersection


def in_hull_multi(points: np.ndarray, hull: ConvexHull) -> np.ndarray:
    hsi_eqns = hull.equations
    x_At = np.matmul(points, hsi_eqns[:, :-1].transpose())
    lhs = x_At + hsi_eqns[:, -1]
    return np.all(lhs <= np.zeros(hsi_eqns.shape[0]), axis=1)


def inv_rosen_given_dependencies(hypercube_design: np.ndarray, convex_hsis: List[np.ndarray]) -> np.ndarray:
    assert len(hypercube_design) == len(convex_hsis)

    axis_order = [0, 1, 2]
    transformed_points = []
    for i in range(len(convex_hsis)):
        transformed_point = inv_cdf_convex_2d_single(hypercube_design[i], convex_hsis[i], axis_order)
        transformed_points.append(transformed_point)

    print(transformed_points)

    return np.array(transformed_points)


def inv_rosen(hypercube_design: np.ndarray, convex_hsi: np.ndarray) -> np.ndarray:
    fp = feasible_point(convex_hsi)
    convex_set = HalfspaceIntersection(convex_hsi, fp)
    hull = ConvexHull(convex_set.intersections)
    n_dims = hypercube_design.shape[1]
    transformed_designs = []
    for axis_order in permutations(range(n_dims)):
        # print(axis_order)
        if n_dims == 3:
            transformed_points = inv_cdf_convex_3d(hypercube_design, convex_hsi, axis_order)
        elif n_dims == 2:
            transformed_points = inv_cdf_convex_2d(hypercube_design, convex_hsi, axis_order)
        else:
            raise ValueError(f"Currently only 2d and 3d shapes supported. This has {n_dims} dimensions")
        transformed_designs.append(transformed_points)

    g_num = 5
    ccds = [central_composite_discrepancy(hull, tp, g_num) for tp in transformed_designs]

    best_ccd, best_design = min(zip(ccds, transformed_designs), key=lambda x: x[0])
    return best_design


def central_composite_discrepancy(hull: ConvexHull, exp_design: np.ndarray, axis_division: int) -> float:
    lb, ub = hull_bounds(hull)
    hull_vol = hull.volume
    num_points = exp_design.shape[0]
    n_dims = exp_design.shape[1]

    integral_part = 0.0
    spacings = [centred_normalize(np.arange(1, axis_division + 1), axis_division) * (ub[i] - lb[i]) + lb[i] for i in
                range(n_dims)]
    subregion_centres = np.array(list(product(*spacings)))
    subregion_area = np.product([np.abs(ub[i] - lb[i]) / axis_division for i in range(n_dims)])

    for x in subregion_centres:
        partition_sum = 0.0
        x_partitions = partition_around_point(hull, x)
        assert len(x_partitions) > 0

        for partition in x_partitions:
            in_hull_mask = in_hull_multi(exp_design, partition)
            points_in_hull_prop = len(exp_design[in_hull_mask]) / num_points
            partition_vol_prop = partition.volume / hull_vol
            p_val = (points_in_hull_prop - partition_vol_prop) ** 2
            partition_sum += p_val
        integral_part += subregion_area * (1.0 / (2 ** n_dims)) * partition_sum

    ccd = ((1.0 / hull_vol) * integral_part) ** 0.5
    # print(ccd)
    return ccd


def partition_around_point(hull: ConvexHull, centre_point: np.ndarray) -> List[ConvexHull]:
    assert centre_point.shape[0] == len(hull.points[0])
    n_dims = centre_point.shape[0]
    base_ineqs = hull.equations

    # Go along each axis and make two halfspace slices
    partition_hulls = []
    combos = list(product([1, -1], repeat=n_dims))
    for partition_signs in product([1, -1], repeat=n_dims):
        new_ineqs = []
        for i, (norm_sign, p_val) in enumerate(zip(partition_signs, centre_point)):
            # Add in new halfspace inequalities at each of the axes
            new_ineq = np.zeros(n_dims + 1)
            new_ineq[i] = norm_sign
            new_ineq[-1] = -p_val * norm_sign
            new_ineqs.append(new_ineq)
        partition_ineqs = np.vstack((base_ineqs, np.array(new_ineqs)))
        fp = feasible_point(partition_ineqs)
        if fp is not None:
            partition_set = HalfspaceIntersection(partition_ineqs, fp)
            partition_hulls.append(ConvexHull(partition_set.intersections))

    return partition_hulls


def approx_volume(hull: ConvexHull, num_xs: int, num_ys: int) -> float:
    bounds = (np.min(hull.points), np.max(hull.points))

    cell_area = (bounds[1] - bounds[0]) * (bounds[1] - bounds[0]) / (num_xs * num_ys)
    # print(cell_area)
    grid = np.mgrid[1:num_xs + 1, 1:num_ys + 1].swapaxes(0, 2)
    normed_grid = centred_normalize(grid, num_xs)
    bounds_grid = (bounds[1] - bounds[0]) * normed_grid + bounds[0]

    reshaped = bounds_grid.reshape(-1, 2)
    # print(in_hull_multi(reshaped, hull))
    convex_mask = in_hull_multi(reshaped, hull).reshape(num_xs, num_ys)
    # convex_mask = np.apply_along_axis(lambda p: in_hull(p, convex_hull), 1, reshaped).reshape(num_xs, num_ys)

    convex_hull_plot_2d(hull)
    ax = plt.gca()
    inside_points = bounds_grid[convex_mask]
    # print(inside_points.shape)

    # Print out what you know the volume to be, and what convex hull computes it as
    true_volume = hull.volume
    approx_volume = cell_area * inside_points.shape[0]
    # print("True Volume: ", true_volume)
    print("Approx Volume: ", approx_volume)
    plt.xlim(0, 6)
    plt.ylim(0, 6)

    plt.scatter(bounds_grid[:, :, 0], bounds_grid[:, :, 1], c='r', alpha=0.5)  # , s=0.5)
    plt.scatter(inside_points[:, 0], inside_points[:, 1], c='g', alpha=1.0)
    plt.show()

    return approx_volume


def intersect_line_convex_2d(line_dir: np.ndarray, line_origin: np.ndarray, conv_poly_hsis: np.ndarray) -> Any:
    # TODO: What if they don't intersect?
    t_max = np.inf
    t_min = -np.inf

    for halfspace in conv_poly_hsis:
        A, b = halfspace[:-1], halfspace[-1]

        dir_align = np.dot(A, line_dir)
        point_align = np.dot(A, line_origin)
        if np.abs(dir_align) < 1e-8:  # Orthogonal
            continue
        if dir_align > 0:  # Pointing in same direction
            t_max = np.minimum(t_max, (-b - point_align) / dir_align)
        else:
            t_min = np.maximum(t_min, (-b - point_align) / dir_align)

    return line_origin + t_min * line_dir, line_origin + t_max * line_dir


def cdf_x0_convex(halfspaces: np.ndarray, x0: float, axis: int) -> float:
    full_point = feasible_point(halfspaces)
    full_convex = HalfspaceIntersection(halfspaces, full_point)
    full_hull = ConvexHull(full_convex.intersections)

    x0_bound = np.zeros(halfspaces.shape[1])
    x0_bound[-1] = -x0
    x0_bound[axis] = 1.0

    bounded_hsis = np.vstack((halfspaces, x0_bound))

    try:
        bounded_point = feasible_point(bounded_hsis)
    except ValueError:
        # print(f"Some sort of feasibility problem at x0: {x0}")
        return 0.0

    if bounded_point is None:
        return 0.0

    bounded_convex = HalfspaceIntersection(bounded_hsis, bounded_point)
    bounded_hull = ConvexHull(bounded_convex.intersections)

    return bounded_hull.volume / full_hull.volume


def inv_cdf_x1_given_x0_convex(convex_hsis: np.ndarray, u: float, given_x0: float, axis: int) -> float:
    x0_line_origin = np.array([0.0, 0.0])
    x0_line_origin[(axis + 1) % len(x0_line_origin)] = given_x0

    x0_line_dir = np.array(([0.0, 0.0]))
    x0_line_dir[axis] = 1.0

    line_start, line_end = intersect_line_convex_2d(x0_line_dir, x0_line_origin, convex_hsis)

    return u * line_end[axis] + (1.0 - u) * line_start[axis]


def inv_cdf_convex_2d_single(unit_point: np.ndarray, convex_hsi: np.ndarray, axis_order: List[int]) -> np.ndarray:
    convex_set = HalfspaceIntersection(convex_hsi, feasible_point(convex_hsi))
    convex_hull = ConvexHull(convex_set.intersections)
    ch_lb, ch_ub = hull_bounds(convex_hull)

    transformed_x0 = brentq(lambda x0: cdf_x0_convex(convex_hsi, x0, axis_order[0]) - unit_point[axis_order[0]], ch_lb[axis_order[0]], ch_ub[axis_order[0]])
    transformed_x1 = inv_cdf_x1_given_x0_convex(convex_hsi, unit_point[axis_order[1]], transformed_x0, axis_order[1])

    # If the "axis_order" differed from the true cartesian "xyz" ordering, then we must rearrange axes first
    ordered_transformed = np.array(unpermute([transformed_x0, transformed_x1], axis_order)).transpose()

    return ordered_transformed


def inv_cdf_convex_2d(unit_square_design: np.ndarray, convex_hsis: np.ndarray, axis_order: List[int]) -> np.ndarray:
    convex_set = HalfspaceIntersection(convex_hsis, feasible_point(convex_hsis))
    convex_hull = ConvexHull(convex_set.intersections)
    ch_lb, ch_ub = hull_bounds(convex_hull)

    transformed_x0 = [
        brentq(lambda x0: cdf_x0_convex(convex_hsis, x0, axis_order[0]) - target, ch_lb[axis_order[0]],
               ch_ub[axis_order[0]]) for target in unit_square_design[:, axis_order[0]]]
    transformed_x1 = []

    for i, x0 in enumerate(transformed_x0):
        x1 = inv_cdf_x1_given_x0_convex(convex_hsis, unit_square_design[i, axis_order[1]], x0, axis_order[1])
        transformed_x1.append(x1)

    # If the "axis_order" differed from the true cartesian "xyz" ordering, then we must rearrange axes first
    ordered_transformed = np.array(unpermute([transformed_x0, transformed_x1], axis_order)).transpose()

    return ordered_transformed


def inv_cdf_convex_3d(unit_cube_design: np.ndarray, convex_hsis: np.ndarray, axis_order: Sequence[int]) -> np.ndarray:
    convex_set = HalfspaceIntersection(convex_hsis, feasible_point(convex_hsis))
    convex_hull = ConvexHull(convex_set.intersections)
    lb_3d, ub_3d = hull_bounds(convex_hull)

    # F(X0)
    x0_start = time.perf_counter()
    transformed_x0 = [brentq(lambda x0: cdf_x0_convex(convex_hsis, x0, axis_order[0]) - target, lb_3d[axis_order[0]],
                             ub_3d[axis_order[0]], xtol=1e-4) for target in unit_cube_design[:, axis_order[0]]]

    # F(X1 | X0)
    transformed_x1 = []

    x1x2_start = time.perf_counter()
    transformed_x2 = []
    for i, x0 in enumerate(transformed_x0):
        # F(X1 | X0)
        proj_hsi_2d = proj_hsi_axis_plane(convex_set, axis_order[0], x0)
        hull_2d = ConvexHull(proj_hsi_2d.intersections)
        lb_2d, ub_2d = hull_bounds(hull_2d)

        if axis_order[0] < axis_order[1]:
            proj_ax = axis_order[1] - 1
        else:
            proj_ax = axis_order[1]

        x1 = brentq(lambda x0: cdf_x0_convex(proj_hsi_2d.halfspaces, x0, proj_ax) - unit_cube_design[i, axis_order[1]],
                    lb_2d[proj_ax], ub_2d[proj_ax], xtol=1e-4)

        transformed_x1.append(x1)

        # F(X2 | X0, X1)
        if axis_order[0] < axis_order[2]:
            line_ax = axis_order[2] - 1
        else:
            line_ax = axis_order[2]

        x2 = inv_cdf_x1_given_x0_convex(proj_hsi_2d.halfspaces, unit_cube_design[i, axis_order[2]], x1, line_ax)
        transformed_x2.append(x2)

    ordered_transformed = np.array(unpermute([transformed_x0, transformed_x1, transformed_x2], axis_order)).transpose()

    return ordered_transformed


def inv_cdf_convex_3d_single(unit_point: np.ndarray, convex_hsi: np.ndarray, axis_order: List[int]) -> np.ndarray:
    convex_set = HalfspaceIntersection(convex_hsi, feasible_point(convex_hsi))
    convex_hull = ConvexHull(convex_set.intersections)
    lb_3d, ub_3d = hull_bounds(convex_hull)

    # F(X0)
    transformed_x0 = brentq(lambda x0: cdf_x0_convex(convex_hsi, x0, axis_order[0]) - unit_point[axis_order[0]], lb_3d[axis_order[0]], ub_3d[axis_order[0]], xtol=1e-4)
    proj_hsi_2d = proj_hsi_axis_plane(convex_set, axis_order[0], transformed_x0)
    hull_2d = ConvexHull(proj_hsi_2d.intersections)
    lb_2d, ub_2d = hull_bounds(hull_2d)

    if axis_order[0] < axis_order[1]:
        proj_ax = axis_order[1] - 1
    else:
        proj_ax = axis_order[1]

    # F(X1 | X0)
    transformed_x1 = brentq(lambda x0: cdf_x0_convex(proj_hsi_2d.halfspaces, x0, proj_ax) - unit_point[axis_order[1]], lb_2d[proj_ax], ub_2d[proj_ax], xtol=1e-4)

    # F(X2 | X0, X1)
    if axis_order[0] < axis_order[2]:
        line_ax = axis_order[2] - 1
    else:
        line_ax = axis_order[2]

    transformed_x2 = inv_cdf_x1_given_x0_convex(proj_hsi_2d.halfspaces, unit_point[axis_order[2]], transformed_x1, line_ax)

    ordered_transformed = np.array(unpermute([transformed_x0, transformed_x1, transformed_x2], axis_order)).transpose()

    return ordered_transformed


def unpermute(items: Sequence[Any], permuted_order: Sequence[int]) -> Sequence[Any]:
    return list(zip(*sorted(zip(permuted_order, items))))[1]


def norm_vecs(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=1)[:, np.newaxis]


def hull_bounds(hull: ConvexHull) -> Tuple[Any, Any]:
    mins = np.min(hull.points, axis=0)
    maxs = np.max(hull.points, axis=0)

    return mins, maxs


def proj_hsi_axis_plane(hsi: HalfspaceIntersection, axis: int, plane_offset: float) -> Optional[HalfspaceIntersection]:
    projected_halfspaces = []
    p_norm = np.zeros(3)
    p_norm[axis] = 1.0

    p_origin = np.zeros(3)
    p_origin[axis] = plane_offset

    for hs in hsi.halfspaces:
        hs_norm = hs[:-1]
        dp = np.dot(hs_norm, p_norm)
        if 1.0 - np.abs(dp) >= 1e-9:
            hs_origin = -hs[-1] * hs[:-1]
            projected_origin = project_to_plane_intersection(hs_origin, hs_norm, hs_origin, p_norm, p_origin)

            projected_norm = hs_norm - dp * p_norm
            projected_norm = projected_norm / np.linalg.norm(projected_norm)

            new_b = -np.dot(projected_origin - p_origin, projected_norm)

            less_dim_norm = np.concatenate((projected_norm[:axis], projected_norm[axis + 1:]))

            projected_halfspaces.append(np.append(less_dim_norm, new_b))
    projected_halfspaces = np.array(projected_halfspaces)

    proj_feasible_point = feasible_point(projected_halfspaces)
    if proj_feasible_point is None:
        return None

    return HalfspaceIntersection(projected_halfspaces, proj_feasible_point)


def run():
    cube_hsi = np.array([
        [1.0, 0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -1.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -1.0],
        [0.0, 0.0, -1.0, 0.0],
    ])

    A = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5]
    ])

    b = np.array([[
        -2.0,
        0.5,
        -1.0,
        0.0,
        -3.0,
        0.0,
        -2.3,
        1.3
    ]])

    hsis = np.hstack((norm_vecs(A), b.transpose()))

    simple_hsi = np.array([
        [1.0, 0.0, 0.0, -4.0],
        [-1.0, 0.0, 0.0, 2.0],
        [0.0, 1.0, 0.0, -5.0],
        [0.0, -1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, -6.0],
        [0.0, 0.0, -1.0, 3.0],
    ])

    cube_points = good_lattice_point(3, 13)
    inv_rosen(cube_points, hsis)


if __name__ == "__main__":
    run()
