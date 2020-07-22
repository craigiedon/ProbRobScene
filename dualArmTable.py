from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects import Shape
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.backend.utils import script_call
from pyrep.const import PrimitiveShape
from pyrep.backend import sim
from multiprocessing import Process
import numpy as np


def grasp(pr, gripper, close: bool) -> None:
    if close:
        pos = 0.05
    else:
        pos = 0.9

    actuated = False
    i = 0
    while not actuated:
        actuated = gripper.actuate(pos, 0.1)
        print("Actuation steps: {}".format(i))
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


def move_to_pos(pr, agent, pos):
    path = agent.get_path(position=pos, euler=[-np.pi, 0, np.pi/2])
    done = False
    while not done:
        done = path.step()
        pr.step()
    pr.step()


# def run():
pr = PyRep()
pr.launch("", headless=False, responsive_ui=True)

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
c1.set_position([0, 0, 0.01 + cube_dims[-1] * 0.5], relative_to=in_tray)

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

# panda_1.set_joint_positions([-0.8, 0.7, 3.305009599330333e-08, -0.8726646304130554, -1.1409618139168742e-07, 1.2217304706573486, 0.7853981256484985])

# move_above_object(panda_1, c1)
# move_above_object(panda_1, c2)

move_above_object(pr, panda_1, c1, z_offset=0.00)
grasp(pr, gripper_1, True)
gripper_1.grasp(c1)


lift_pos = panda_1.get_tip().get_position()
lift_pos[2] += 0.1
move_to_pos(pr, panda_1, lift_pos)

lift_pos[1] += 0.45

move_to_pos(pr, panda_1, lift_pos)

gripper_1.release()
grasp(pr, gripper_1, False)

lift_pos[1] -= 0.45

move_to_pos(pr, panda_1, lift_pos)

move_above_object(pr, panda_2, c1, z_offset=0.0)
grasp(pr, gripper_2, True)
gripper_2.grasp(c1)

move_above_object(pr, panda_2, out_tray, z_offset=0.1)
gripper_2.release()
grasp(pr, gripper_2, False)


for i in range(1000):
    if (i % 100) == 0:
        print("step", i)
    pr.step()


# print("Grasp 3 happened")
# grasp(pr, gripper_2, False)
# print("Grasp 4 happened")

pr.stop()


pr.shutdown()

print("Phew! We got home nice and safe...")
