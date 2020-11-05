import math
import time
from typing import List

import numpy as np

from probRobScene.core.distributions import (Samplable, MethodDistribution, OperatorDistribution,
                                             support_interval, underlying_function, distributionFunction, condition_to, needs_sampling)
from probRobScene.core.geometry import normalize_angle
from probRobScene.core.object_types import Object
from probRobScene.core.regions import PointInRegionDistribution, Intersect, cube_to_hsi, Region, EmptyRegion, erode_hsis, Convex, ConvexPolyhedronRegion, IntersectionRegion, intersect_many
from probRobScene.core.utils import InvalidScenarioError
from probRobScene.core.vectors import VectorField, PolygonalVectorField, VectorMethodDistribution, Vector3D


def prune(scenario, verbosity=1):
    if verbosity >= 1:
        print('  Pruning scenario...')
        start_time = time.time()

    prune_containment(scenario, verbosity)

    if verbosity >= 1:
        total_time = time.time() - start_time
        print(f'  Pruned scenario in {total_time:.4g} seconds.')


def prune_containment(scenario, verbosity):
    prunable : List[Object] = []
    for obj in scenario.objects:
        if isinstance(obj.position, PointInRegionDistribution):
            r = obj.position.region
            if isinstance(r, (Intersect, IntersectionRegion)):
                prunable.append(obj)

    for obj in prunable:
        if isinstance(obj.position.region, IntersectionRegion):
            r_intersected = intersect_many(*obj.position.region.regions)
            condition_to(obj.position, PointInRegionDistribution(r_intersected))

        if not needs_sampling(obj.to_orientation()):
            eroded_container = erode_container(scenario.workspace, obj.dimensions(), obj.to_orientation())
        else:
            eroded_container = erode_container(scenario.workspace, obj.dimensions(), Vector3D(0, 0, 0))

        new_base = intersect_container(eroded_container, obj.position._conditioned.region)
        new_pos = PointInRegionDistribution(new_base)
        condition_to(obj.position, new_pos)
        # TODO: Calculate volume of convex poly? Probs not possible with infinite vol halfspaces...


@distributionFunction
def intersect_container(container: Convex, obj_pos_region: Intersect):
    new_base = obj_pos_region.intersect(container)
    return new_base


@distributionFunction
def erode_container(container: Convex, obj_dims: Vector3D, obj_rot: Vector3D):
    o_hsi = cube_to_hsi(np.zeros(3), np.array(obj_dims), np.array(obj_rot))
    container_hsi = container.to_hsi()
    return ConvexPolyhedronRegion(erode_hsis(container_hsi, o_hsi))