from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.backend.utils import script_call
from pyrep.const import PrimitiveShape
from pyrep.objects import Camera, Shape, VisionSensor
from pyrep.backend import sim
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt


def grasp(pr, gripper, close: bool) -> None:
    if close:
        pos = 0.1
    else:
        pos = 0.9

    actuated = False
    i = 0
    while not actuated:
        actuated = gripper.actuate(pos, 0.01)
        pr.step()
        i += 1
    pr.step()

    return


def move_above_object(pr, agent, target_obj, z_offset=0.05):
    pos = target_obj.get_position()
    pos[2] = pos[2] + z_offset

    path = agent.get_path(position=pos, euler=[-np.pi, 0.0, np.pi / 2.0], ignore_collisions=True)  # , euler=orient)
    # path.visualize()

    done = False
    while not done:
        done = path.step()
        pr.step()
    pr.step()
    return


def move_to_pos(pr, agent, pos, z_offset=0.0):
    path = agent.get_path(position=pos + np.array([0.0, 0.0, z_offset]), euler=[-np.pi, 0, np.pi/2])
    done = False
    while not done:
        done = path.step()
        pr.step()
    pr.step()


def location_from_depth_cam(pr, d_cam, target_obj):
    depth_cam.set_entity_to_render(target_obj.get_handle()) # Cheat at blob extraction - just use mask)
    pr.step()

    p_angle = d_cam.get_perspective_angle()
    clip_near = d_cam.get_near_clipping_plane()
    clip_far = d_cam.get_far_clipping_plane()
    cam_res = d_cam.get_resolution()
    d_im = np.array(d_cam.capture_depth())

    mask_locs = np.flip(np.transpose(np.where(d_im <= 1.0)), axis=1)
    lower_bound = mask_locs[0]
    upper_bound = mask_locs[-1]
    depth_pixel_loc = (lower_bound + upper_bound) / 2.0

    # print("Depth pixel loc: ", depth_pixel_loc)
    # print("D Image", d_im)
    depth_raw = d_im[int(depth_pixel_loc[1]), int(depth_pixel_loc[0])]
    depth = clip_near + (clip_far - clip_near) * depth_raw
    # print("Depth Raw: ", depth_raw)
    # print("Depth", depth)

    coord = [0, 0, depth]
    bX = depth_pixel_loc[0] / cam_res[0]
    bY = depth_pixel_loc[1] / cam_res[1]

    # print("bX: ", bX, "bY: ", bY)

    coord[0] = depth * np.tan(np.deg2rad(p_angle * 0.5)) * (0.5 - bX) * 2.0
    coord[1] = depth * np.tan(np.deg2rad(p_angle * 0.5)) * (0.5 - bY) * 2.0

    # print("Coordinates: ", coord)

    cam_matrix = np.reshape(d_cam.get_matrix(), (3, 4))

    # print("Cam Matrix: ", cam_matrix)
    world_coord = np.matmul(cam_matrix, np.append(coord, 1))
    return world_coord

    # Shape.create(type=PrimitiveShape.CUBOID, color=[0,0,1], position=world_coord, size=[0.02, 0.02, 0.02], respondable=False, static=True, visible_edges=True, mass=0.01)


# def run():
pr = PyRep()
pr.launch("", headless=False, responsive_ui=True)

scene_view = Camera('DefaultCamera')
scene_view.set_position([3.45, 0.18, 2.0])
scene_view.set_orientation(np.array([180, -70, 90]) * np.pi / 180.0)

depth_cam = VisionSensor.create([256, 256], position=[0, 0, 2.0], orientation=np.array([0.0, 180.0, -90.0]) * np.pi / 180.0 )

# Import Robots
pr.import_model("assets/Panda.ttm")
pr.import_model("assets/Panda.ttm")


# Table Creation
table = pr.import_model("assets/Table.ttm")
table_script = sim.sim_scripttype_customizationscript

script_call("table_length_callback@Table", table_script, floats=[1.2])
script_call("table_width_callback@Table", table_script, floats=[2.5])
script_call("table_height_callback@Table", table_script, floats=[0.8])

table_bounds = table.get_bounding_box()
print("Table bounds: ", table_bounds)
table_pos = table.get_position()
table_top_z = table_bounds[-1]

# Trays Creation

in_tray = pr.import_model("assets/Tray.ttm")
in_tray.set_position([0.1, -0.4, table_top_z], relative_to=table)

out_tray = pr.import_model("assets/Tray.ttm")
out_tray.set_position([0.1, 0.4, table_top_z], relative_to=table)

c1_color = [0.5, 0.0, 0.0]
c2_color = [0.0, 0.5, 0.0]
c3_color = [0.0, 0.0, 0.5]
cube_dims = [0.05, 0.05, 0.05]

c1 = Shape.create(type=PrimitiveShape.CUBOID, color=c1_color, size=cube_dims, visible_edges=True, mass=0.01)
c1.set_position([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0.01 + cube_dims[-1] * 0.5], relative_to=in_tray)
# depth_cam.set_entity_to_render(c1.get_handle())

# c2 = Shape.create(type=PrimitiveShape.CUBOID, color=c2_color, size=cube_dims)
# c2.set_position([0.10, 0.0, 0.01 + cube_dims[-1] * 0.5], relative_to=in_tray)

# c3 = Shape.create(type=PrimitiveShape.CUBOID, color=c3_color, size=cube_dims)
# c3.set_position([-0.10, 0.0, 0.01 + cube_dims[-1] * 0.5], relative_to=in_tray)

pr.step_ui()

panda_1 = Panda(0)
gripper_1 = PandaGripper(0)

panda_2 = Panda(1)
gripper_2 = PandaGripper(1)

panda_1.set_position([-0.5, -0.4, table_top_z + 0.07], relative_to=table) # Panda-base is (inexplicably!) slightly above bottom of panda
panda_2.set_position([-0.5, 0.4, table_top_z + 0.07], relative_to=table)


print("Robot joint angles: ", panda_1.get_joint_positions())
print("Robot joint velocities: ", panda_1.get_joint_velocities())

pr.start()
pr.step()

cube_pos_from_im = location_from_depth_cam(pr, depth_cam, c1)
move_to_pos(pr, panda_1, cube_pos_from_im, z_offset=-0.02)
grasp(pr, gripper_1, True)
gripper_1.grasp(c1)


lift_pos = [np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), table_pos[2] + table_top_z + 0.1]
move_to_pos(pr, panda_1, lift_pos)
gripper_1.release()
grasp(pr, gripper_1, False)
lift_pos[1] -= 0.45
move_to_pos(pr, panda_1, lift_pos)

cube_pos_from_im = location_from_depth_cam(pr, depth_cam, c1)
move_to_pos(pr, panda_2, cube_pos_from_im, z_offset=-0.02)
grasp(pr, gripper_2, True)
gripper_2.grasp(c1)

move_above_object(pr, panda_2, out_tray, z_offset=0.1)
gripper_2.release()
grasp(pr, gripper_2, False)

pr.stop()


pr.shutdown()

print("Phew! We got home nice and safe...")
