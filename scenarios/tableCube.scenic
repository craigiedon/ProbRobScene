from model import *

table = Table on Vector3D(0,0,0), with width 1.8, with length 0.8, with height 0.8
r1 = Robot on (top back table).position - Vector3D(0.4, 0, 0), with color "0.5"
r2 = Robot on (top back table).position + Vector3D(0.4, 0, 0)

tr_1 = Tray completely on table, ahead of r1 by 0.1, left of (top table) by 0.2
tr_2 = Tray completely on table, ahead of r2 by 0.1, right of (top table) by 0.2

c1 = Cube completely on tr_1

camera = Camera at Vector3D(table.x + (-0.1, 0.1), table.y + (-0.1, 0.1), (1.9, 2.1)), facing Vector3D(0, 0, -1)

