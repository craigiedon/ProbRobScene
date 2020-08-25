"""Scenic model for manipulation scenarios in CoppeliaSim"""

import math

width = 3
height = 3
workspace = Workspace(RectangularRegion(0 @ 0, 0, width, height))

class Cube:
    position: Point in workspace
    position3d: Point3D at Vector3D((-1.5, 1.5),(-1.5, 1.5),(-1.5, 1.5))
    heading: (-math.pi, math.pi)
    width: 0.2
    height: 0.2
    color: 'blue'
    primitiveShapeType: 'CUBOID'


class Tray:
    position: Point in workspace
    width: 0.5
    height: 0.75
    usable_area: RectangularRegion(self.position, self.heading, self.width - 0.05, self.height - 0.05)
    color: '0.75'
    allowCollisions: True
