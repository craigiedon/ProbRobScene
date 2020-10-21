from model import *

"""

- Robot arm manipulator with
    - a starting postition
    - a robot type
    - certain degrees of freedom
    - a set of joint positions


- The trays should be *on* the table
- Tray should have workable *region* with in it which is a proportion of the actual width, length etc.
- The tray should be somewhere in front of first robot (with noise).
- The second tray should be somewhere in front of the second robot.
- There should be three cubes *in* the tray region, but also restricted to not touch each other.
- There should be the ability to randomly sample colours from a vector

"""

"""
table = Table on floor
t_1 = Tray on table
t_2 = Tray on table
c1 = Cube on t_1
c2 = Cube on t_1
c3 = Cube on t_1

robot = Robot ahead of table by 1.1, facing toward table
"""

cup_table = Table on floor, with width 0.75, with length 0.75, with height 0.8
bowl_table = Table on floor, aheadRough of cup_table from 0.6 to 0.8, with width 0.6, with length 0.6, with height 0.5
bowl = Bowl completely on bowl_table
robot = Robot on cup_table, facing Vector3D(0, -1, 0)
c1 = Cup aheadRough of robot
c2 = Cup completely on cup_table
c3 = Cup completely on cup_table


# table = Table on floor
# robot = Robot aheadRough of table

# c1 = BCube on floor
# c2 = BCube on floor, aheadRough of c1, with color 'red'
# c3 = BCube on floor, aheadRough of c1, with color 'red'
# table = Table on floor
# robot = Robot on floor, aheadRough of table
# tray_1 = Tray on table
# tray_2 = Tray on table

# robot = BCube in CuboidRegion(Vector3D(width / 2.0, length / 2.0, height / 2.0), Vector3D(0, 0, 0), width * 2, length * 2, height * 2)
# c1 = BCube in CuboidRegion(Vector3D(1,1,1), Vector3D(0,0,0), 0.2, 0.2, 0.2)
# c2 = BCube in CuboidRegion(Vector3D((0, 2), 2, 2), Vector3D(0.2, 0.4, 0.1), 3, 3, 3)

# require (c2.position is in CuboidRegion(Vector3D(2,2,2), Vector3D(0,0,0), 1,1,1))