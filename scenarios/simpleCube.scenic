from model import *

"""
ego should be a robot arm manipulator, with a starting postition, certain degrees of freedom, a robot type,
and a set of joint positions

There should be a table (with width,length, height, heading, position)

There should be a tray with dimensions, position etc., but also a workable *region* with in it which is a proportion of the actual
width, length etc. The tray should be somewhere in front of first robot (with noise). The second tray should be somewhere in front
of the second robot. The trays should be *on* the table

There should be three cubes *in* the tray region, but also restricted to not touch each other. There should be the ability
to randomly sample colours from a vector

"""


# So this works: when the "on" is referencing something with a fixed position
table = Table on floor
tray = Tray on table
ego = Cube on tray
c1 = Cube on tray
c2 = Cube on tray

c3 = Robot ahead of (bottom table) by 1.0, facing toward table

# But if I do
# table = Table in workspace
# It doesn't work, as the point distribution gets nested and fails to get sampled? Why?
