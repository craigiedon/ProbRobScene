"""Scenario and scene objects."""

import random
from dataclasses import dataclass
from typing import Any, List, Tuple, Mapping

from probRobScene.core.distributions import Samplable, RejectionException, needs_sampling, sample_all
from probRobScene.core.geometry import cuboids_intersect
from probRobScene.core.lazy_eval import needs_lazy_evaluation
from probRobScene.core.object_types import Object, show_3d
from probRobScene.core.plotUtil3d import draw_cube
from probRobScene.core.regions import AABB, contains
from probRobScene.core.utils import areEquivalent, InvalidScenarioError, RuntimeParseError
import numpy as np


class Scene:
    def __init__(self, workspace, objects, params):
        self.workspace = workspace
        self.objects = tuple(objects)
        self.params = params

    def show_3d(self, block=True):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # print("Num objects:", len(self.objects))
        for obj in self.objects:
            show_3d(obj, ax)

        w_min_corner, w_max_corner = AABB(self.workspace)
        w_dims = w_max_corner - w_min_corner

        draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)

        total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)

        ax.set_xlim(total_min, total_max)
        ax.set_ylim(total_min, total_max)
        ax.set_zlim(total_min, total_max)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.show(block=block)


def has_static_bounds(obj) -> bool:
    static_pos = not needs_sampling(obj.position)
    static_corners = [not needs_sampling(corner) for corner in obj.corners]
    return static_pos and all(static_corners)
    # return not (needs_sampling(obj.position) or any(needs_sampling(corner) for corner in obj.corners))


@dataclass(frozen=True)
class Scenario:
    """A compiled Scenic scenario, from which scenes can be sampled."""
    workspace: Any
    objects: List[Object]
    params: List
    requirements: List
    requirement_deps: List

    def __post_init__(self):
        assert self.workspace is not None
        self.validate()

    @property
    def dependencies(self):
        param_deps = tuple(p for p in self.params.values() if isinstance(p, Samplable))
        return tuple(self.objects) + param_deps + tuple(self.requirement_deps)

    def isEquivalentTo(self, other):
        if type(other) is not Scenario:
            return False
        return (areEquivalent(other.workspace, self.workspace)
                and areEquivalent(other.objects, self.objects)
                and areEquivalent(other.params, self.params)
                and areEquivalent(other.requirements, self.requirements)
                and other.external_sampler == self.external_sampler)

    def validate(self):
        """Make some simple static checks for inconsistent built-in requirements."""
        objects = self.objects
        static_bounds = [has_static_bounds(obj) for obj in objects]
        for i in range(len(objects)):
            oi = objects[i]
            # skip objects with unknown positions or bounding boxes
            if not static_bounds[i]:
                continue
            # Require object to be contained in the workspace/valid region
            container = self.workspace
            if not needs_sampling(container) and not contains(container, oi):
                contains(container, oi)
                raise InvalidScenarioError(f'Object at {oi.position} does not fit in container')
            for j in range(i):
                oj = objects[j]
                if not static_bounds[j]:
                    continue
                if cuboids_intersect(oi, oj):
                    raise InvalidScenarioError(f'Object at {oi.position} intersects'
                                               f' object at {oj.position}')

    def generate(self, max_iterations=2000, verbosity=0, feedback=None):
        active_reqs = [req for req, prob in self.requirements if random.random() <= prob]

        sample, iterations = rejection_sample(self.objects, self.dependencies, self.workspace,
                                              active_reqs, max_iterations, verbosity)

        # obtained a valid sample; assemble a scene from it
        sampled_objects = tuple(sample[obj] for obj in self.objects)
        sampled_params = {}
        for param, value in self.params.items():
            sampled_value = sample[value] if isinstance(value, Samplable) else value
            assert not needs_lazy_evaluation(sampled_value)
            sampled_params[param] = sampled_value
        scene = Scene(self.workspace, sampled_objects, sampled_params)
        return scene, iterations


def rejection_sample(objects, dependencies, workspace, active_reqs, max_iterations, verbosity) -> Tuple[Mapping, int]:
    for i in range(max_iterations):
        try:
            sample = sample_all(dependencies)
        except RejectionException as e:
            if verbosity >= 2:
                print(f'Rejected sample {i} because of: {e}')
            continue

        valid, reason = is_valid_sample(sample, objects, workspace, active_reqs)
        if not valid:
            print(f'Rejected sample {i} because of: {reason}')
            continue

        return sample, i

    raise RejectionException(f'failed to generate scenario in {max_iterations} iterations')


def is_valid_sample(sample: Mapping[Object, Object], objects: List[Object], workspace, active_reqs) -> Tuple[bool, str]:
    obj_samples = [sample[o] for o in objects]
    ns = [needs_sampling(o) for o in obj_samples]
    assert not any(ns)

    collidable = [o for o in obj_samples if not o.allowCollisions]

    for o in obj_samples:
        if not contains(workspace, o):
            return False, 'object containment'

    if any(cuboids_intersect(vi, vj) for (i, vi) in enumerate(collidable) for vj in collidable[:i]):
        return False, 'object intersection'

    for (j, req) in enumerate(active_reqs):
        if not req(sample):
            return False, f'user-specified requirement {j}'

    return True, "valid"
