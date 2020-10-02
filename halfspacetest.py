import numpy as np
from scipy.spatial import ConvexHull

from scenic3d.core.intersect import cube_to_hsi, intersect_hsis
from scenic3d.core.plotUtil3d import draw_polyhedron
import matplotlib.pyplot as plt

hs_1 = cube_to_hsi(1 * np.ones(3), np.ones(3), np.zeros(3))
ch_1 = ConvexHull(hs_1.intersections)

hs_2 = cube_to_hsi(2.0 * np.ones(3), np.ones(3), np.zeros(3))
ch_2 = ConvexHull(hs_2.intersections)

hs_intersection = intersect_hsis(hs_1, hs_2)
ch_intersection = ConvexHull(hs_intersection.intersections)

print("intersection Points: ", ch_intersection.points)
print("intersection Simplices: ", ch_intersection.simplices)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
draw_polyhedron(ax, ch_1.points, ch_1.simplices, color='g', alpha=0.3)
draw_polyhedron(ax, ch_2.points, ch_2.simplices, color='b', alpha=0.3)
draw_polyhedron(ax, ch_intersection.points, ch_intersection.simplices, color='r')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
plt.show()
