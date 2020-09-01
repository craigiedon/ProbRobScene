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


t = Tray
c1 = Cube in t.usable_area

ego = Cube ahead of (front of t) by 0.3, facing toward t, with color 'green'

p3d  = Point3D beyond c1.position3d by 0.5 from t.position3d
c2 = Cube with position3d p3d
