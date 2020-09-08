"""Scenic model for manipulation scenarios in CoppeliaSim"""

import math

width = 5
length = 5
height = 5
workspace = Workspace(CuboidRegion(Vector3D(0,0,0), Vector3D(0,0,0), width, length, height))

class Cube:
    width: 0.2
    height: 0.2
    length: 0.2
    color: 'blue'
    primitiveShapeType: 'CUBOID'
    allowCollisions: False


class Tray:
    width: 0.5
    length: 0.75
    height: 0.2
    color: '0.75'
    allowCollisions: False
    # usable_area: RectangularRegion(self.position, self.heading, self.width - 0.05, self.height - 0.05)


class Table:
    width: 2.0
    length: 1.0
    height: 1.0
    color: 'brown'
