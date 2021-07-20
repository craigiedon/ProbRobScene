# Load in a PRS scenario file with "cube on table"
import itertools
from typing import Any, List, Optional, Set, Union, Mapping

import numpy as np
from multimethod import multimethod
import matplotlib.pyplot as plt

from probRobScene import scenario_from_string, scenario_from_file
from probRobScene.core.distributions import RejectionException, sample_all, DefaultIdentityDict, sample, needs_sampling, Samplable, Range
from probRobScene.core.geometry import cuboids_intersect
from probRobScene.core.intersections import to_hsi
from probRobScene.core.object_types import Object, show_3d
from probRobScene.core.plotUtil3d import draw_cube
from probRobScene.core.regions import Region, ConvexPolyhedron, Rectangle3D, PointInRegionDistribution, Cuboid, HalfSpace, Plane, ConvexPolygon3D, PointSet, LineSeg, Line, AABB
from probRobScene.core.scenarios import Scenario, Scene
from probRobScene.core.vectors import rotate_euler, Vector3D
from scipy.spatial import HalfspaceIntersection, ConvexHull

from good_lattice_point import good_lattice_point
from inverse_rosen import inv_rosen, inv_rosen_given_dependencies, inv_cdf_convex_2d_single, inv_cdf_convex_3d_single, central_composite_discrepancy


@multimethod
def unif_region_design(r: Region, n: int) -> np.ndarray:
    raise NotImplementedError(f"Not yet implemented uniform designs for region type: {type(r)}")


@multimethod
def unif_region_design(r: ConvexPolyhedron, unit_point: np.ndarray) -> Vector3D:
    assert unit_point.shape == (3,)
    region_design = inv_cdf_convex_3d_single(unit_point, r.hsi.halfspaces, [0, 1, 2])
    return Vector3D(*region_design)


@multimethod
def unif_region_design(r: Union[Rectangle3D, ConvexPolygon3D], unit_point: np.ndarray) -> Vector3D:
    assert unit_point.shape == (2,)
    flat_design = inv_cdf_convex_2d_single(unit_point, to_hsi(r).halfspaces, [0, 1])
    flat_3d = np.append(flat_design, 0)
    exp_design = rotate_euler(flat_3d, r.rot) + r.origin
    return Vector3D(*exp_design)


@multimethod
def unif_design_given_deps(d: PointInRegionDistribution, unit_design: np.ndarray, dep_values: List[Mapping]):
    transformed_points = []
    for i, unit_point in enumerate(unit_design):
        concrete_region = dep_values[i][d.region]
        transformed_points.append(unif_region_design(concrete_region, unit_point))

    return transformed_points


@multimethod
def unif_design_given_deps(d: Range, unit_design: np.ndarray, dep_values: List[Mapping]):
    assert unit_design.ndim == 1 or (unit_design.ndim == 2 and unit_design.shape[1] == 1)
    transformed_points = []
    for i, t in enumerate(unit_design.reshape(-1)):
        low = dep_values[i][d.low]
        high = dep_values[i][d.high]
        transformed_points.append((1.0 - t) * low + t * high)

    return transformed_points


@multimethod
def unif_design_given_deps(x: Any, unit_design: np.ndarray, dep_values: List[Mapping]) -> List:
    transformed_points = [x.sample_given_dependencies(dep_vi) for dep_vi in dep_values]
    return transformed_points


@multimethod
def inv_trans(r: ConvexPolyhedron, unit_point: np.ndarray) -> np.ndarray:
    return inv_cdf_convex_3d_single(unit_point, r.hsi.halfspaces, [0, 1, 2])


def d_tree(objs: List[Samplable]) -> List[Samplable]:
    assert len(objs) == len((set(objs)))

    visited = set()
    ordered_deps = []

    def visit(o: Samplable):
        c_obj = o
        if hasattr(o, "_conditioned"):
            c_obj = o._conditioned

        if o in visited:
            return

        for d in c_obj.dependencies():
            visit(d)

        visited.add(c_obj)
        ordered_deps.append(c_obj)

    for o in objs:
        visit(o)

    return ordered_deps


def try_unif_design(scenario: Scenario, n: int) -> List[Scene]:
    sub_designs: List[Mapping[Samplable, Any]] = [DefaultIdentityDict() for i in range(n)]

    ordered_deps = d_tree(scenario.objects)
    d_dims = dep_dims(ordered_deps)
    print(d_dims)

    unit_design = good_lattice_point(np.sum(d_dims), n)

    for i, d in enumerate(ordered_deps):
        start_ix = int(np.sum(d_dims[:i]))
        end_ix = int(start_ix + d_dims[i])
        dep_design = unif_design_given_deps(d, unit_design[:, start_ix:end_ix], sub_designs)

        for x in dep_design:
            assert not needs_sampling(dep_design)

        for j in range(n):
            sub_designs[j][d] = dep_design[j]

    for d_inst in sub_designs:
        obj_inst = [d_inst[o] for o in scenario.objects]
        ns = [needs_sampling(o) for o in obj_inst]
        assert not any(ns)

        collidable = [o for o in obj_inst if not o.allowCollisions]

        for i, vi in enumerate(collidable):
            for j, vj in enumerate(collidable[:i]):
                if cuboids_intersect(vi, vj):
                    print(f"Collision between {i} : {vi} and {j} : {vj}")

    scenes = [Scene(scenario.workspace, tuple(d_inst[o] for o in scenario.objects), []) for d_inst in sub_designs]

    return scenes


def dep_dims(ordered_dependencies: List[Samplable]) -> List[int]:
    total_dims = []
    for o in ordered_dependencies:
        if isinstance(o, PointInRegionDistribution):
            if isinstance(o.region, (ConvexPolyhedron, ConvexPolyhedron, Cuboid, HalfSpace)):
                total_dims.append(3)
            elif isinstance(o.region, (ConvexPolygon3D, Rectangle3D, Plane)):
                total_dims.append(2)
            elif isinstance(o.region, (Line, LineSeg, PointSet)):
                total_dims.append(1)
            else:
                raise NotImplementedError
        elif isinstance(o, Range):
            total_dims.append(1)
        else:
            total_dims.append(0)
    return total_dims


def run():
    n = 20
    scenario = scenario_from_file("../../../scenarios/syntEx2.prs")
    scenes = try_unif_design(scenario, n)
    print(scenes)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for i, scene in enumerate(scenes):
        for obj in scene.objects:
            if not hasattr(obj, 'model_name') or obj.model_name != "Table":
                show_3d(obj, ax)

    w_min_corner, w_max_corner = AABB(scenes[0].workspace)
    w_dims = w_max_corner - w_min_corner

    draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)

    total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()

    show_3d(scenes[0].objects[0], ax)

    plt.show()
    # d = d_tree(scenario.objects)
    # d_dims = dep_dims(d)
    # print(d)
    # scenes = unif_design_from_scenario(scenario, n)
    #
    # sample_scenes = [scenario.generate()[0] for i in range(n)]
    #
    # # Extract random "design"
    # random_design = np.array([s.objects[1].position for s in sample_scenes])
    # print(random_design.shape)
    # cube_reg = scenario.objects[1]._conditioned._dependencies[0].region
    # flattened_random = rotate_euler(random_design - cube_reg.origin, cube_reg.rev_rot)
    # print("Flat: ", flattened_random)
    # cube_hull = ConvexHull(cube_reg.to_hsi().intersections)
    # print(cube_hull)
    # random_ccd = central_composite_discrepancy(cube_hull, flattened_random[:, :2], 5)
    # print("Random CCD: ", random_ccd)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # print("Num objects:", len(self.objects))
    #
    # w_min_corner, w_max_corner = scenes[0].workspace.getAABB()
    # w_dims = w_max_corner - w_min_corner
    #
    # draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)
    #
    # total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)
    #
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    #
    # plt.tight_layout()
    #
    # scenes[0].objects[0].show_3d(ax)
    # for scene in scenes:
    #     for obj in scene.objects[1:]:
    #         obj.show_3d(ax)
    # plt.show()


if __name__ == "__main__":
    run()
