"""Scenic model for manipulation scenarios in CoppeliaSim"""

import math

width = 3
length = 3
height = 3
workspace = Workspace(CuboidRegion(Vector3D(0,0,0), Vector3D(0,0,0), width, length, height))

class Cube:
    position: Vector3D((-width / 2, width / 2), (-length / 2, length / 2), (-height / 2, height / 2))
    width: 0.2
    height: 0.2
    length: 0.2
    color: 'blue'
    primitiveShapeType: 'CUBOID'


class Tray:
    width: 0.5
    length: 0.75
    height: 0.2
    color: '0.75'
    allowCollisions: True
    # usable_area: RectangularRegion(self.position, self.heading, self.width - 0.05, self.height - 0.05)
