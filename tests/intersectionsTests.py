import unittest

from multimethod import DispatchError

from probRobScene.core.intersections import intersect
from probRobScene.core.regions import All, Empty, Spherical, Cuboid, Intersection, Rectangle3D
from probRobScene.core.vectors import Vector3D


class TestIntersections(unittest.TestCase):
    """
    Regions:
        Halfspace
        ConvexPolyhedron
        ConvexPolygon3D
        Cuboid
        Rectangle3D
        Plane
        Line
        LineSeg
        PointSet
    """

    def test_all_all(self):
        it = intersect(All(), All())
        self.assertEqual(All(), it)

    def test_all_empty(self):
        it = intersect(All(), Empty())
        self.assertEqual(Empty(), it)

    def test_all_others(self):
        s = Spherical(Vector3D(0, 0, 0), 3)
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)

        it_1 = intersect(All(), s)
        it_2 = intersect(All(), c)

        self.assertEqual(it_1, s)
        self.assertEqual(it_2, c)

    def test_others_all(self):
        s = Spherical(Vector3D(0, 0, 0), 3)
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)

        it_1 = intersect(s, All())
        it_2 = intersect(c, All())

        self.assertEqual(it_1, s)
        self.assertEqual(it_2, c)

    def test_empty_all(self):
        self.assertEqual(Empty(), intersect(Empty(), All()))

    def test_empty_empty(self):
        self.assertEqual(Empty(), intersect(Empty(), Empty()))

    def test_empty_others(self):
        s = Spherical(Vector3D(0, 0, 0), 3)
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)
        self.assertEqual(Empty(), intersect(Empty(), s))
        self.assertEqual(Empty(), intersect(Empty(), c))

    def test_others_empty(self):
        s = Spherical(Vector3D(0, 0, 0), 3)
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)
        self.assertEqual(Empty(), intersect(s, Empty()))
        self.assertEqual(Empty(), intersect(c, Empty()))

    def test_spherical_others(self):
        s = Spherical(Vector3D(0, 0, 0), 3)
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)
        self.assertRaises(DispatchError, intersect, s, c)

    def test_others_spherical(self):
        s = Spherical(Vector3D(0, 0, 0), 3)
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)
        self.assertRaises(DispatchError, intersect, c, s)

    def test_intersection_others(self):
        inter = Intersection([Cuboid(Vector3D(0,0,0), Vector3D(0,0,0), 1, 1, 1), Rectangle3D(1,1,Vector3D(0,0,0), Vector3D(0,0,0))])
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)
        self.assertRaises(DispatchError, intersect, c, inter)

    def test_others_intersection(self):
        inter = Intersection([Cuboid(Vector3D(0,0,0), Vector3D(0,0,0), 1, 1, 1), Rectangle3D(1,1,Vector3D(0,0,0), Vector3D(0,0,0))])
        c = Cuboid(Vector3D(0, 0, 0), Vector3D(0, 0, 0), 1, 1, 1)
        self.assertRaises(DispatchError, intersect, inter, c)


if __name__ == '__main__':
    unittest.main()
