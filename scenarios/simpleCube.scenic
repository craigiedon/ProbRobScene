from model import *

"""

- ego should be a robot arm manipulator with
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


table = Table on floor
t_1 = Tray on table
t_2 = Tray on table
c1 = Cube on t_1
c2 = Cube on t_1
c3 = Cube on t_1

ego = Robot ahead of table by 1.5, facing toward table

