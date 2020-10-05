import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from scenic3d.core.vectors import offset_beyond, Vector3D


def draw_cube(ax, pos: np.array, size: np.array, rot: np.array, color: str = 'b'):
    r = R.from_euler('zyx', rot, degrees=False)

    global_bounds = np.transpose(np.stack([pos - size / 2.0, pos + size / 2.0]))

    for i in range(3):
        surface_combo = np.array(np.meshgrid(*np.delete(global_bounds, i, 0)))

        for j in range(2):
            static_dim = np.repeat(global_bounds[i][j], 4).reshape(2, 2)
            full_surface = np.insert(surface_combo, i, static_dim, axis=0)

            # Local rotation: need to rearrange axis, and undo translation before applying
            rotated = r.apply(np.moveaxis(full_surface, 0, 2).reshape(4, 3) - pos).reshape(2, 2, 3) + pos
            rotated = np.moveaxis(rotated, 2, 0)

            ax.plot_surface(*rotated, alpha=0.5, color=color, edgecolor='black')


def draw_polyhedron(ax, points: np.array, faces: np.array, color: str = 'b', alpha=1.0):
    for face in faces:
        face_points = np.array([points[i] for i in face])
        ax.add_collection3d(Poly3DCollection(face_points, color=color, edgecolor='black', alpha=alpha))


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    robot = np.array([0.3, 0.4, 0.2])
    cube = np.array([0.5, 0.6, -0.2])

    draw_cube(ax, robot, 0.1 * np.ones(3), np.zeros(3), 'g')
    draw_cube(ax, cube, 0.1 * np.ones(3), np.zeros(3), 'b')

    new_pos = offset_beyond(Vector3D(*cube), Vector3D(0.4, 0.0, 0.0), Vector3D(*robot))

    draw_cube(ax, new_pos, 0.1 * np.ones(3), np.zeros(3), 'y')

    ax.quiver(cube[0], cube[1], cube[2], new_pos[0] - cube[0], new_pos[1] - cube[1], new_pos[2] - cube[2])

    ax.set_xlim(-0.0, 1.0)
    ax.set_ylim(-0.0, 1.0)
    ax.set_zlim(-0.0, 1.0)

    plt.show()
