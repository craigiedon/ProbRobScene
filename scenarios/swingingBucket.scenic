from model import *

table = Table on Vector3D(0,0,0), with width 1.8, with length 0.8, with height 0.8, with color "0.9"
r = Robot on (top back table).position - Vector3D(0.4, 0, 0), with color "0.5"

tray = Tray completely on table, ahead of r by 0.1, left of (top table) by 0.2

cube_1 = Cube completely on tray
cube_2 = Cube on cube_1
cube_3 = Cube on cube_2

bucket = RopeBucket on Vector3D(r.x, r.y+0.6, r.z - 0.3)

camera = Camera at Vector3D(table.x + (-0.1, 0.1), table.y + (-0.1, 0.1), (1.9, 2.1)), facing Vector3D(0, 0, -1)
