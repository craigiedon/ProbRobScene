from model import *

"""
table = Table on Vector3D(0,0,0), with width 2.5, with length 1.2, with height 0.8
r1 = Robot on (top back table).position - Vector3D(0.4, 0, 0), with color "0.5"
r2 = Robot on (top back table).position + Vector3D(0.4, 0, 0)

tr_1 = Tray completely on table, ahead of r1, left of (top table) by 0.1, facing Vector3D(1, 0, 0)
tr_2 = Tray completely on table, right of (top table) by 0.1

c1 = Cube completely on tr_1

camera = Camera at Vector3D(table.x + (-0.1, 0.1), table.y + (-0.1, 0.1), (3, 4)), facing Vector3D(0, 0, -1)
"""

c1 = Cube on floor, with width 0.5, with length 2.5, facing Vector3D(1, 0, 0)
c2 = ToyCube completely on c1, with color 'green'
