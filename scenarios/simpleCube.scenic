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


cup_table = Table on Vector3D(0,0,0), with width 0.75, with length 0.75, with height 0.8
bowl_table = Table on floor, aheadRough of cup_table from 0.6 to 0.8, with width 0.6, with length 0.6, with height 0.5
bowl = Bowl completely on bowl_table
robot = Robot completely on cup_table, facing Vector3D(0, -1, 0)
c1 = Cup completely on cup_table, aheadRough of robot by 0.1
c2 = Cup completely on cup_table, aheadRough of robot by 0.1
c3 = Cup completely on cup_table, aheadRough of robot by 0.1
