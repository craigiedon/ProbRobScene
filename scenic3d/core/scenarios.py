"""Scenario and scene objects."""

import random

from scenic3d.core.distributions import Samplable, RejectionException, needsSampling
from scenic3d.core.external_params import ExternalSampler
from scenic3d.core.geometry import cuboids_intersect
from scenic3d.core.lazy_eval import needs_lazy_evaluation
from scenic3d.core.utils import areEquivalent, InvalidScenarioError
from scenic3d.core.workspaces import Workspace


class Scene:
    """A scene generated from a Scenic scenario.

    Attributes:
        objects (tuple(:obj:`~scenic3d.core.object_types.Object`)): All objects in the
          scene. The ``ego`` object is first.
        ego_object (:obj:`~scenic3d.core.object_types.Object`): The ``ego`` object.
        params (dict): Dictionary mapping the name of each global parameter to its value.
        workspace (:obj:`~scenic3d.core.workspaces.Workspace`): Workspace for the scenario.
    """

    def __init__(self, workspace, objects, ego_object, params):
        self.workspace = workspace
        self.objects = tuple(objects)
        self.egoObject = ego_object
        self.params = params

    def show_3d(self, block=True):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        self.workspace.show_3d(ax)

        print("Num objects:", len(self.objects))
        for obj in self.objects:
            obj.show_3d(ax, highlight=(obj is self.egoObject))

        plt.show(block=block)

    def show(self, zoom=None, block=True):
        """Render a schematic of the scene for debugging."""
        import matplotlib.pyplot as plt
        # display map
        self.workspace.show(plt)
        # draw objects
        for obj in self.objects:
            obj.show(self.workspace, plt, highlight=(obj is self.egoObject))
        # zoom in if requested
        if zoom is not None:
            self.workspace.zoomAround(plt, self.objects, expansion=zoom)
        plt.show(block=block)


def has_static_bounds(obj):
    return not (needsSampling(obj.position) or any(needsSampling(corner) for corner in obj.corners))


class Scenario:
    """A compiled Scenic scenario, from which scenes can be sampled."""

    def __init__(self, workspace,
                 objects, ego_object,
                 params, external_params,
                 requirements, requirement_deps):
        if workspace is None:
            workspace = Workspace()  # default empty workspace
        self.workspace = workspace
        ordered = []
        for obj in objects:
            ordered.append(obj)
            if obj is ego_object:  # make ego the first object
                ordered[0], ordered[-1] = ordered[-1], ordered[0]
        assert ordered[0] is ego_object
        self.objects = tuple(ordered)
        self.egoObject = ego_object
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
        static_visibility = not needsSampling(self.egoObject.visibleRegion)
        static_bounds = [has_static_bounds(obj) for obj in objects]
        for i in range(len(objects)):
            oi = objects[i]
            # skip objects with unknown positions or bounding boxes
            if not static_bounds[i]:
                continue
            # Require object to be contained in the workspace/valid region
            container = self.workspace.region
            if not needsSampling(container) and not container.contains_object(oi):
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
        scene = Scene(self.workspace, sampled_objects, self.egoObject, sampled_params)
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
        sample = Samplable.sampleAll(dependencies)
    except RejectionException as e:
        return None, e

    obj_samples = (sample[o] for o in objects)
    collidable = (o for o in obj_samples if not o.allowCollisions)

    if any(not workspace.region.contains_object(o) for o in obj_samples):
        return None, 'object containment'

    if any(cuboids_intersect(vi, vj) for (i, vi) in collidable for vj in collidable[:i]):
        return None, 'object intersection'

    if any(not req(sample) for req in active_reqs):
        return None, 'user-specified requirement'

    return sample, None
