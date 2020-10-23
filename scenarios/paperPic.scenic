from model import *

table = Table on Vector3D(0,0,0)

t1 = Tray completely on table, with width (0.5, 1.0), with length (0.5, 1.0)
t2 = Tray completely on table, with width (0.5, 1.0), with length (0.5, 1.0)
c1 = ToyCube completely on t1, with color "green"
c2 = ToyCube completely on t2, with color "blue"

r1 = Robot on Vector3D(1.6, 0.25, 0), facing toward table
r2 = Robot on Vector3D(-1.4, 0.1, 0), facing toward table
r3 = Robot on Vector3D(0.2, 1.5, 0), facing toward table
r4 = Robot on Vector3D(0.4, -1.2, 0), facing toward table
