from copy import copy
import time
from typing import List

import probRobScene.core.distributions
from probRobScene.core.intersections import intersect_many, erode_hsis, to_hsi, intersect
from probRobScene.core.object_types import Object
from probRobScene.core.regions import PointInRegionDistribution, Intersection, Intersect, Convex, ConvexPolyhedron, Cuboid
from probRobScene.core.scenarios import Scenario
from probRobScene.core.vectors import Vector3D
from probRobScene.core.distributions import needs_sampling


def prune(s: Scenario, verbosity: int = 1) -> None:
    if verbosity >= 1:
        print('  Pruning scenario...')
        start_time = time.time()

    for o in s.objects:
        prune_obj(o, s)

    if verbosity >= 1:
        total_time = time.time() - start_time
        print(f'  Pruned scenario in {total_time:.4g} seconds.')


def prune_obj(o: Object, scenario: Scenario):
    if not isinstance(o.position, PointInRegionDistribution):
        return o

    pruned_pos = o.position
    if isinstance(o.position.region, Intersection):
        r_intersected = intersect_many(*o.position.region.regions)
        pruned_pos = PointInRegionDistribution(r_intersected)

    # With a fixed orientation, we can precisely erode the outer workspace, otherwise we must approximate with a default orientation
    if not needs_sampling(o.orientation):
        eroded_container = erode_container(scenario.workspace, o.dimensions, o.orientation)
    else:
        eroded_container = erode_container(scenario.workspace, o.dimensions, Vector3D(0, 0, 0))

    new_base = intersect_container(eroded_container, pruned_pos.region)
    o.position._conditioned = PointInRegionDistribution(new_base)

@probRobScene.core.distributions.distributionFunction
def intersect_container(container: Convex, obj_pos_region: Intersect):
    new_base = intersect(obj_pos_region, container)
    return new_base


@probRobScene.core.distributions.distributionFunction
def erode_container(container: Convex, obj_dims: Vector3D, obj_rot: Vector3D):
    o_hsi = to_hsi(Cuboid(Vector3D(0.0, 0.0, 0.0), obj_rot, *obj_dims))
    container_hsi = to_hsi(container)
    return ConvexPolyhedron(erode_hsis(container_hsi, o_hsi))
