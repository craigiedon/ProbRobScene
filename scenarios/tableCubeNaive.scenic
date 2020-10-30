from model import *

table = Table on Vector3D(0,0,0), with width 1.8, with length 0.8, with height 0.8
r1 = Robot on (top back table).position - Vector3D(0.4, 0, 0), with color "0.5"
r2 = Robot on (top back table).position + Vector3D(0.4, 0, 0)

tr_1 = Tray on table
tr_2 = Tray on table

c1 = Cube on tr_1

camera = Camera at Vector3D(table.x + (-0.1, 0.1), table.y + (-0.1, 0.1), (1.9, 2.1)), facing Vector3D(0, 0, -1)

# Ahead / Left of reqs

require (right tr_1).x + 0.2 < table.x
require (back tr_1).y - 0.1 > r1.y

require (left tr_2).x - 0.2 > table.x
require (back tr_2).y - 0.1 > r2.y

# Table containment reqs
require (left tr_1).x > (left table).x

require (front tr_1).y < (front table).y
require (back tr_1).y > (back table).y

require (right tr_2).x < (right table).x
require (front tr_2).y < (front table).y
require (back tr_2).y > (back table).y

# Cube Containment Reqs
require (front c1).y < (front tr_1).y
require (back c1).y > (back tr_1).y
require (left c1).x > (left tr_1).x
require (right c1).x < (right tr_1).x