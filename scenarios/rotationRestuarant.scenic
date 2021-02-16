from model import *

table = Table on Vector3D(0,-1.15,0), with width 1.8, with length 0.8, with height 0.8, with color "0.9"
chair = DiningChair on Vector3D(1.15, -1.15, 0), facing Vector3D(1, 0, 0)

r = Robot on Vector3D(0, -0.3, 0)
belt = CircularConveyorBelt on Vector3D(0, 1.0, 0)

plate = Plate on Vector3D(belt.x, belt.y - 0.75, belt.z+0.081)
cylinder_1 = Cylinder completely on plate
cylinder_2 = Cylinder completely on plate
cylinder_3 = Cylinder completely on plate

camera = Camera at Vector3D(table.x + (-0.1, 0.1), table.y + (-0.1, 0.1), (1.9, 2.1)), facing Vector3D(0, 0, -1)
