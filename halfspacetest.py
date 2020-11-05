from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.linalg import inv

from probRobScene.core.regions import cube_to_hsi, intersect_hsis, erode_hsis, feasible_point
from probRobScene.core.plotUtil3d import draw_polyhedron, draw_polygon_3d
import matplotlib.pyplot as plt
from probRobScene.core.regions import PlaneRegion
from probRobScene.core.vectors import Vector3D, reverse_euler, rotate_euler_v3d, rotate_euler

if __name__ == "__main__":
    hs_1 = cube_to_hsi(np.array([0.0, 0.0, 0.0]), np.ones(3), np.array([0.4, np.pi / 2.5, np.pi / 2.0]))
    ch_1 = ConvexHull(hs_1.intersections)

    # hs_2 = cube_to_hsi(0.0 * np.ones(3), np.array([0.1, 0.1, 0.4]), np.array([0.0, 0.0, np.pi / 2.0]))
    # ch_2 = ConvexHull(hs_2.intersections)

    p = PlaneRegion(Vector3D(0, 0.0, 0.60), Vector3D(0.0, 0.5, 1.0))

    hs_proj = proj_hsi_to_plane(hs_1, p)
    ch_proj = ConvexHull(hs_proj.intersections)

    #
    # hs_eroded = erode_hsis(hs_1, hs_2)
    # ch_eroded = ConvexHull(hs_eroded.intersections)

    # hs_2 = cube_to_hsi(2.0 * np.ones(3), np.ones(3), np.zeros(3))
    # ch_2 = ConvexHull(hs_2.intersections)
    #
    # hs_intersection = intersect_hsis(hs_1, hs_2)
    # ch_intersection = ConvexHull(hs_intersection.intersections)
    #
    # print("intersection Points: ", ch_intersection.points)
    # print("intersection Simplices: ", ch_intersection.simplices)
    #
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax_bound = 1
    ax.set_xlim(-ax_bound, ax_bound)
    ax.set_ylim(-ax_bound, ax_bound)
    ax.set_zlim(-ax_bound, ax_bound)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    draw_polyhedron(ax, ch_1.points, ch_1.simplices, color='g', alpha=0.3)
    draw_polygon_3d(ax, ch_proj.points, ch_proj.vertices,
                    pos=p.origin, rot=p.rot, color='r', alpha=0.8)
    draw_polygon_3d(ax, np.array([[-ax_bound, -ax_bound], [ax_bound, -ax_bound], [ax_bound, ax_bound], [-ax_bound, ax_bound]]), np.array([0, 1, 2, 3]),
                    pos=p.origin, rot=p.rot, color='b', alpha=0.1)
    # draw_polyhedron(ax, ch_2.points, ch_2.simplices, color='b', alpha=0.3)
    # draw_polyhedron(ax, ch_eroded.points, ch_eroded.simplices, color='r', alpha=0.3)

    plt.show()
