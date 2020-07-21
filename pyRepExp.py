from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.backend.utils import script_call
from pyrep.backend import sim
from multiprocessing import Process
import numpy as np

PROCESSES = 1


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

pr.step_ui()

panda_1 = Panda(0)
gripper_1 = PandaGripper(0)

panda_2 = Panda(1)
gripper_2 = PandaGripper(1)

panda_1.set_position([0.0, -0.6, table_top_z], relative_to=table)
panda_1.set_joint_positions([0.001, 0.17, 0.003, -0.9, -0.11, 1.22, 0.79])

print("joint positions: ", panda_1.get_joint_positions())

panda_2.set_position([0.0, 0.6, table_top_z], relative_to=table)

# # Get position of robot tip
home_pos = panda_1.get_tip().get_position()
home_orient = panda_1.get_tip().get_orientation()
print('Position: ', home_pos, ' Orientation: ', home_orient)

print("Robot joint angles: ", panda_1.get_joint_positions())
print("Robot joint velocities: ", panda_1.get_joint_velocities())

pr.start()

for i in range(1000):
    if (i % 100) == 0:
        print("step", i)
    pr.step()

pr.stop()



pr.shutdown()

print("Phew! We got home nice and safe...")
