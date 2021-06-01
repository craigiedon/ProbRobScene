"""Scenario and scene objects."""

import random

from probRobScene.core.distributions import Samplable, RejectionException, needs_sampling, sample_all
from probRobScene.core.external_params import ExternalSampler
from probRobScene.core.geometry import cuboids_intersect
from probRobScene.core.lazy_eval import needs_lazy_evaluation
from probRobScene.core.plotUtil3d import draw_cube
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
            obj.show_3d(ax)

        w_min_corner, w_max_corner = self.workspace.getAABB()
        w_dims = w_max_corner - w_min_corner

        draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)

        total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)

        # ax.set_xlim(total_min, total_max)
        # ax.set_ylim(total_min, total_max)
        # ax.set_zlim(total_min, total_max)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        plt.tight_layout()
        plt.show(block=block)


def has_static_bounds(obj):
    return not (needs_sampling(obj.position) or any(needs_sampling(corner) for corner in obj.corners))


class Scenario:
    """A compiled Scenic scenario, from which scenes can be sampled."""

    def __init__(self, workspace,
                 objects,
                 params, external_params,
                 requirements, requirement_deps):
        assert workspace is not None

        if needs_sampling(workspace):
            raise RuntimeParseError('workspace region must be fixed')

        self.workspace = workspace
        self.objects = tuple(objects)
        self.params = dict(params)
        self.externalParams = tuple(external_params)
        self.requirements = tuple(requirements)
        self.external_sampler = ExternalSampler.forParameters(self.externalParams, self.params)
        # dependencies must use fixed order for reproducibility
        param_deps = tuple(p for p in self.params.values() if isinstance(p, Samplable))
        self.dependencies = self.objects + param_deps + tuple(requirement_deps)
        self.validate()

    def isEquivalentTo(self, other):
        if type(other) is not Scenario:
            return False
        return (areEquivalent(other.workspace, self.workspace)
                and areEquivalent(other.objects, self.objects)
                and areEquivalent(other.params, self.params)
                and areEquivalent(other.externalParams, self.externalParams)
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
            if not needs_sampling(container) and not container.contains_object(oi):
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
                                              active_reqs, self.external_sampler, max_iterations, verbosity)

        # obtained a valid sample; assemble a scene from it
        sampled_objects = tuple(sample[obj] for obj in self.objects)
        sampled_params = {}
        for param, value in self.params.items():
            sampled_value = sample[value] if isinstance(value, Samplable) else value
            assert not needs_lazy_evaluation(sampled_value)
            sampled_params[param] = sampled_value
        scene = Scene(self.workspace, sampled_objects, sampled_params)
        return scene, iterations

    def reset_external_sampler(self):
        """Reset the scenario's external sampler, if any.

        If the Python random seed is reset before calling this function, this
        should cause the sequence of generated scenes to be deterministic."""
        self.external_sampler = ExternalSampler.forParameters(self.externalParams, self.params)


def rejection_sample(objects, dependencies, workspace, active_reqs, external_sampler, max_iterations, verbosity):
    for i in range(max_iterations):
        sample, rejection_reason = try_sample(external_sampler, dependencies, objects, workspace, active_reqs)
        if sample is not None:
            return sample, i
        if verbosity >= 2:
            print(f'  Rejected sample {i} because of: {rejection_reason}')
    raise RejectionException(f'failed to generate scenario in {max_iterations} iterations')


def try_sample(external_sampler, dependencies, objects, workspace, active_reqs):
    try:
        if external_sampler is not None:
            external_sampler.sample(external_sampler.rejectionFeedback)
        sample = sample_all(dependencies)
    except RejectionException as e:
        return None, e

    obj_samples = [sample[o] for o in objects]
    collidable = [o for o in obj_samples if not o.allowCollisions]

    for o in obj_samples:
        if not workspace.contains_object(o):
            return None, 'object containment'

    if any(cuboids_intersect(vi, vj) for (i, vi) in enumerate(collidable) for vj in collidable[:i]):
        return None, 'object intersection'

    for (i, req) in enumerate(active_reqs):
        if not req(sample):
            return None, f'user-specified requirement {i}'

    return sample, None

# def zoom_around(workspace: Region, plt, objects, expansion=2):
#     """Zoom the schematic around the specified objects"""
#     positions = (obj.position for obj in objects)
#     x, y = zip(*positions)
#     minx, maxx = min_and_max(x)
#     miny, maxy = min_and_max(y)
#     sx = expansion * (maxx - minx)
#     sy = expansion * (maxy - miny)
#     s = max(sx, sy) / 2.0
#     s += max(max(obj.width, obj.height) for obj in objects)  # TODO improve
#     cx = (maxx + minx) / 2.0
#     cy = (maxy + miny) / 2.0
#     plt.xlim(cx - s, cx + s)
#     plt.ylim(cy - s, cy + s)
