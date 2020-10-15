import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from scenic3d.core.regions import HalfSpaceRegion
from scenic3d.core.vectors import offset_beyond, Vector3D

if __name__ == "__main__":
    v = HalfSpaceRegion(Vector3D(1, 2, 3), Vector3D(0, 1, 0), dist=10.0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    robot = np.array([0.3, 0.4, 0.2])
    cube = np.array([0.5, 0.6, -0.2])

    # draw_cube(ax, robot, 0.1 * np.ones(3), np.zeros(3), 'g')
    # draw_cube(ax, cube, 0.1 * np.ones(3), np.zeros(3), 'b')
    # new_pos = offset_beyond(Vector3D(*cube), Vector3D(0.4, 0.0, 0.0), Vector3D(*robot))
    # draw_cube(ax, new_pos, 0.1 * np.ones(3), np.zeros(3), 'y')

    ax.quiver(*v.point, *v.normal, color='b')
    ax.quiver(*v.point, *v.ax_1, color='r')
    ax.quiver(*v.point, *v.ax_2, color='g')

    for i in range(200):
        s = v.uniform_point_inner()
        if v.contains_point(s):
            ax.scatter(*s, color='g')
        else:
            ax.scatter(*s, color='r')

    ax.set_xlim(-v.dist, v.dist)
    ax.set_ylim(-v.dist, v.dist)
    ax.set_zlim(-v.dist, v.dist)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()
