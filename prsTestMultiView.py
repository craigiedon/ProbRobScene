from typing import List

import probRobScene
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from probRobScene.core.object_types import show_3d
from probRobScene.core.plotUtil3d import draw_cube
from probRobScene.core.regions import AABB
from probRobScene.core.scenarios import Scene

scenario = probRobScene.scenario_from_file("scenarios/pointDepTest.prs")

max_generations = 30
rejections_per_scene = []

sample_scenes: List[Scene] = []
for i in range(max_generations):
    print(f"Generation {i}")
    ex_world, used_its = scenario.generate(verbosity=2)
    rejections_per_scene.append(used_its)
    sample_scenes.append(ex_world)
#


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
w_min_corner, w_max_corner = AABB(scenario.workspace)
w_dims = w_max_corner - w_min_corner
total_min, total_max = np.min(w_min_corner), np.max(w_max_corner)
ax.set_xlim(total_min, total_max)
ax.set_ylim(total_min, total_max)
ax.set_zlim(total_min, total_max)

draw_cube(ax, (w_max_corner + w_min_corner) * 0.5, w_dims, np.zeros(3), color='purple', alpha=0.03)

for scene in sample_scenes:
    for o in scene.objects:
        show_3d(o, ax)
plt.show()

avg_rejections = np.average(rejections_per_scene)
print(avg_rejections)
