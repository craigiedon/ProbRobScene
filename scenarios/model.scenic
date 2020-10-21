"""Scenic model for manipulation scenarios in CoppeliaSim"""

import math

width = 3
length = 3
height = 5
workspace = CuboidRegion(Vector3D(width / 2.0, length / 2.0, height / 2.0), Vector3D(0,0,0), width, length, height)
floor = Rectangle3DRegion(width, length, Vector3D(width / 2.0, length / 2.0, 0.0), Vector3D(0, 0, 0))

class Cube:
    width: 0.02
    height: 0.02
    length: 0.02
    color: 'blue'
    shape_type: 'CUBOID'

class BCube:
    width: 0.5
    height: 0.5
    length: 0.5
    color: 'blue'


class Tray:
    width: 0.4522
    length: 0.2364
    height: 0.028964
    color: '0.75'
    model_name: "Tray"


class Table:
    width: 2.0
    length: 2.0
    height: 0.9
    model_name: "Table"
    color: 'brown'

class Bowl:
    width: 0.258534
    length: 0.258564
    height: 0.115678
    model_name: "Bowl"
    color: 'purple'

class Cup:
    width: 0.09
    length: 0.09
    height: 0.1233
    model_name: "Cup"
    color: "blue"

class Robot:
    width: 0.298174
    length: 0.733159
    height: 1.025562
    color: 'white'