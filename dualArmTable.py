from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.objects import Shape
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.backend.utils import script_call
from pyrep.const import PrimitiveShape
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

panda_1.set_position([-0.5, -0.6, table_top_z], relative_to=table)

print("joint positions: ", panda_1.get_joint_positions())

panda_2.set_position([-0.5, 0.6, table_top_z], relative_to=table)


c1_color = [150.0, 0.0, 0.0]
c2_color = [0.0, 150.0, 0.0]
c3_color = [0.0, 0.0, 150.0]
cube_dims = [0.05, 0.05, 0.05]

c1 = Shape.create(type=PrimitiveShape.CUBOID, 
                  color=c1_color,
                  size=cube_dims)
c1.set_position([0, 0, table_top_z + cube_dims[-1] * 0.5], relative_to=table)

c2 = Shape.create(type=PrimitiveShape.CUBOID, 
                  color=c2_color,
                  size=cube_dims)
c2.set_position([0.0, 0.2, table_top_z + cube_dims[-1] * 0.5], relative_to=table)

c3 = Shape.create(type=PrimitiveShape.CUBOID, 
                  color=c3_color,
                  size=cube_dims)
c3.set_position([0.0, 0.4, table_top_z + cube_dims[-1] * 0.5], relative_to=table)


print("Robot joint angles: ", panda_1.get_joint_positions())
print("Robot joint velocities: ", panda_1.get_joint_velocities())

pr.start()

panda_1.set_joint_positions([-0.8, 0.7, 3.305009599330333e-08, -0.8726646304130554, -1.1409618139168742e-07, 1.2217304706573486, 0.7853981256484985])

for i in range(1000):
    if (i % 100) == 0:
        print("step", i)
    pr.step()

pr.stop()



pr.shutdown()

print("Phew! We got home nice and safe...")
