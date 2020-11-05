import probRobScene
import numpy as np

scenario = probRobScene.scenarioFromFile("scenarios/tableCubeNaive.scenic")

max_generations = 1
rejections_per_scene = []
for i in range(max_generations):
    print(f"Generation {i}")
    ex_world, used_its = scenario.generate(verbosity=2)
    rejections_per_scene.append(used_its)
    ex_world.show_3d()
#
avg_rejections = np.average(rejections_per_scene)
print(avg_rejections)