import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def draw_cube(ax, pos: np.array, size: np.array, rot: np.array, color: str = 'b'):
    r = R.from_euler('zyx', rot, degrees=True)

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


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    draw_cube(ax, np.array([5, 0, 0]), np.array([1.0, 1.0, 5.0]), np.array([45.0, 0.0, 0.0]), 'b')
    draw_cube(ax, np.array([0, 0, 0]), np.array([1.0, 1.0, 5.0]), np.array([0.0, 45.0, 0.0]), 'r')
    draw_cube(ax, np.array([1, 1, 0]), np.array([1.0, 1.0, 5.0]), np.array([0.0, 0.0, 45.0]), 'g')

    ax.set_xlim(-5.0, 5.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_zlim(-5.0, 5.0)

    plt.show()
