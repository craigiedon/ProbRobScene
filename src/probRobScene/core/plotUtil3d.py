import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from multimethod import multimethod
from scipy.spatial.qhull import HalfspaceIntersection, ConvexHull
from scipy.spatial.transform import Rotation as R

from probRobScene.core.distributions import needs_sampling
from probRobScene.core.vectors import offset_beyond, Vector3D, rotate_euler_v3d, rotation_to_euler
from numpy.random import uniform as unif



def draw_cube(ax, pos: np.array, size: np.array, rot: np.array, color: str = 'b', alpha: float = 0.5):
    r = R.from_euler('zyx', rot, degrees=False)

    """
    3 x 2 bounds:
     [[x_min, x_max],
      [y_min, y_max],
      [z_min, z_max]]
    """
    global_bounds = np.transpose(np.stack([pos - size / 2.0, pos + size / 2.0]))

    for i in range(3):
        # For each face of cube, the corner offsets only involve variation along two of the axes
        bound_pair = np.delete(global_bounds, i, 0)
        surface_combo = np.array(np.meshgrid(*bound_pair))

        for j in range(2):
            static_dim = np.repeat(global_bounds[i][j], 4).reshape(2, 2)
            full_surface = np.insert(surface_combo, i, static_dim, axis=0)

            # Local rotation: need to rearrange axis, and undo translation before applying
            rotated = r.apply(np.moveaxis(full_surface, 0, 2).reshape(4, 3) - pos).reshape(2, 2, 3) + pos
            rotated = np.moveaxis(rotated, 2, 0)

            ax.plot_surface(*rotated, alpha=alpha, color=color, edgecolor='black')


def draw_polyhedron(ax, points: np.array, faces: np.array, color: str = 'b', alpha=1.0):
    for face in faces:
        face_points = np.array([points[i] for i in face])
        ax.add_collection3d(Poly3DCollection(face_points, color=color, edgecolor='black', alpha=alpha))


def draw_convex(ax, hsi: HalfspaceIntersection, color: str = 'b', alpha=0.5):
    hull = ConvexHull(hsi.intersections)
    draw_polyhedron(ax, hull.points, hull.simplices, color=color, alpha=alpha)


def draw_line(ax, origin: np.array, direction: np.array, color: str = 'b', alpha: float = 0.5):
    ax.quiver(origin[0], origin[1], origin[2], direction[0], direction[1],
              direction[2], length=2.0, normalize=True, color=color, alpha=alpha)


def draw_line_seg(ax, start: np.array, end: np.array, color: str = 'b', alpha: float = 0.5):
    direction = end - start
    length = np.linalg.norm(direction)
    direction /= length
    # print(direction)
    # print("start: ", start)
    # print("end: ", end)
    # print("direction: ", direction)

    ax.quiver(*start, *direction, length=length, arrow_length_ratio=0.1, color=color, alpha=alpha, linewidths=4.0)


def draw_plane(ax, origin: np.array, normal: np.array, color: str = 'b', alpha: float = 0.5, size=1.0):
    norm_rot = np.array(rotation_to_euler(Vector3D(0.0, 0.0, 1.0), normal))
    draw_rect_3d(ax, origin, size, size, norm_rot, color, alpha)


def draw_rect_3d(ax, pos: np.array, width: float, length: float, rot: np.array, color: str = 'b', alpha: float = 0.5):
    r = R.from_euler('zyx', rot, degrees=False)
    hw = np.array([width / 2.0, 0.0, 0.0])
    hl = np.array([0.0, length / 2.0, 0.0])
    origin_bounds = np.array([
        -hw - hl,
        -hw + hl,
        hw + hl,
        hw - hl
    ])
    rotated_bounds = r.apply(origin_bounds)
    surface_points = rotated_bounds + pos
    draw_polyhedron(ax, surface_points, np.array([[0, 1, 2], [2, 3, 0]]), color, alpha)


def draw_polygon_3d(ax, points: np.array, vertice_indices: np.array, pos: np.array = None, rot: np.array = None, color: str = 'b', alpha: float = 1.0):
    ordered_verts = np.array([np.append(points[i], 0) for i in vertice_indices])
    r = R.from_euler('zyx', rot) if rot is not None else R.from_euler('zyx', np.zeros(3))
    pos = pos if pos is not None else np.zeros(3)

    rotated_verts = r.apply(ordered_verts)
    translated_verts = rotated_verts + pos
    X, Y, Z = translated_verts.transpose()
    ax.plot_trisurf(X, Y, Z, color=color, alpha=alpha, edgecolor='black')


if __name__ == "__main__":
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.0, 1.0)
    ax.set_ylim(-0.0, 1.0)
    ax.set_zlim(-0.0, 1.0)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    c1 = CuboidRegion(Vector3D(0.5, 0.5, 0.5), Vector3D(0,0,0), 0.4, 0.3, 0.2)
    l_1 = LineSeg3DRegion(Vector3D(0.5, 0.5, 0.0), Vector3D(0.5, 0.5, 1.0))
    l_i = intersect_lineseg_convex(l_1, c1)
    draw_convex(ax, c1.to_hsi(), alpha=0.1)
    draw_line_seg(ax, l_1.start, l_1.end, color='r', alpha=0.25)
    draw_line_seg(ax, l_i.start, l_i.end, color='g', alpha=1.0)
    # n = np.array([0.0, 0.5, 0.5])
    # n = n / np.linalg.norm(n)
    #
    # r1 = Rectangle3DRegion(1.0, 1.0, Vector3D(unif(0.2, 0.5), unif(0.2, 0.5), unif(0.2, 0.5)), Vector3D(np.pi / unif(2.0, 6.0), np.pi / unif(2.0, 6.0), np.pi / unif(2.0, 6.0)))
    # r2 = Rectangle3DRegion(1.0, 1.0, Vector3D(0.5, 0.5, 0.5), Vector3D(0.0, np.pi / 4.0, 0.0))
    #
    # draw_rect_3d(ax, r1.origin, r1.width, r1.length, r1.rot, color='b', alpha=0.2)
    # draw_rect_3d(ax, r2.origin, r2.width, r2.length, r2.rot, color='r', alpha=0.2)
    #
    # l_1 = intersect_rects(r1, r2)
    # draw_line_seg(ax, l_1.start, l_1.end, color='g', alpha=1.0)
    plt.show()
