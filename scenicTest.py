import scenic3d
from scenic3d.core.object_types import Point3D
from scenic3d.core.vectors import Vector3D
from scenic3d.syntax.veneer import With

# p3d = Point3D(With("position", Vector3D(1.0, 2.0, 3.0)))
scenario = scenic3d.scenarioFromFile("scenarios/simpleCube.scenic")

for i in range(5):
    ex_world, used_its = scenario.generate()
    ex_world.show_3d()
