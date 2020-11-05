import numpy as np
from wrappers.setupFuncs import top_of


def grasp(pr, gripper, close: bool) -> None:
    if close:
        pos = 0.1
    else:
        pos = 0.9

    actuated = False
    i = 0
    while not actuated:
        actuated = gripper.actuate(pos, 0.1)
        # print("Open amount: ", gripper.get_open_amount())
        pr.step()
    pr.step()

    return


def force_grasp(pr, gripper, close: bool) -> None:
    joints = gripper.joints
    print("joints: ", len(joints))

    # gripper.set_joint_forces([5.0, 5.0])
    gripper.set_joint_target_velocities([-0.04] * gripper._num_joints)


def move_above_object(pr, agent, target_obj, z_offset=0.00, ig_cols=False):
    pos = top_of(target_obj)
    pos[2] += z_offset

    path = agent.get_path(position=pos, euler=[-np.pi, 0.0, np.pi / 2.0], ignore_collisions=ig_cols)  # , euler=orient)
    # path.visualize()

    done = False
    while not done:
        done = path.step()
        pr.step()
    pr.step()
    return


def move_to_pos(pr, agent, pos, z_offset=0.0, ig_cols=False):
    path = agent.get_path(position=pos + np.array([0.0, 0.0, z_offset]), euler=[-np.pi, 0, np.pi/2], ignore_collisions=ig_cols)
    done = False
    while not done:
        done = path.step()
        pr.step()
    pr.step()


def location_from_depth_cam(pr, d_cam, target_obj):
    # Cheat at blob extraction - just use mask)
    d_cam.set_entity_to_render(target_obj.get_handle())
    pr.step()

    p_angle = d_cam.get_perspective_angle()
    clip_near = d_cam.get_near_clipping_plane()
    clip_far = d_cam.get_far_clipping_plane()
    cam_res = d_cam.get_resolution()
    d_im = np.array(d_cam.capture_depth())

    # Get bounding box of depth mask and take x/y location of object as centre
    mask_locs = np.flip(np.transpose(np.where(d_im <= 1.0)), axis=1)
    lower_bound = mask_locs[0]
    upper_bound = mask_locs[-1]
    depth_pixel_loc = (lower_bound + upper_bound) / 2.0

    # Raw depth is between 0/1, convert to world depth with reference to camera clip plane
    depth_raw = d_im[int(depth_pixel_loc[1]), int(depth_pixel_loc[0])]
    depth = clip_near + (clip_far - clip_near) * depth_raw

    coord = [0, 0, depth]
    bX = depth_pixel_loc[0] / cam_res[0]
    bY = depth_pixel_loc[1] / cam_res[1]

    # Use tan trigonometry to get appropriate x-y projections
    coord[0] = depth * np.tan(np.deg2rad(p_angle * 0.5)) * (0.5 - bX) * 2.0
    coord[1] = depth * np.tan(np.deg2rad(p_angle * 0.5)) * (0.5 - bY) * 2.0

    # Coord just assumes non transformed camera, so incorporate rotations and translations
    cam_matrix = np.reshape(d_cam.get_matrix(), (3, 4))

    world_coord = np.matmul(cam_matrix, np.append(coord, 1))
    return world_coord
