import scenic3d
from scenic3d.core.object_types import Point3D
from scenic3d.core.vectors import Vector3D
from scenic3d.syntax.veneer import With

scenario = scenic3d.scenarioFromFile("scenarios/simpleCube.scenic")

for i in range(1):
    ex_world, used_its = scenario.generate(verbosity=2)
    ex_world.show_3d()

