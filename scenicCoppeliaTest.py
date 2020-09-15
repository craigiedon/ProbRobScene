import scenic3d
from scenic3d.core.object_types import Point3D
from scenic3d.core.vectors import Vector3D
from scenic3d.syntax.veneer import With
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
from setupFuncs import setAonB, create_table, top_of
import robotControl as rc

from scenicCoppeliaSimWrapper import cop_from_scenic

scenario = scenic3d.scenarioFromFile("scenarios/simpleCube.scenic")
ex_world, used_its = scenario.generate()

pr = PyRep()
pr.launch("scenes/emptyVortex.ttt", headless=False, responsive_ui=True)

scene_view = Camera('DefaultCamera')
scene_view.set_position([3.45, 0.18, 2.0])
scene_view.set_orientation(np.array([180, -70, 90]) * np.pi / 180.0)

depth_cam = VisionSensor.create([256, 256], position=[0, 0, 2.0], orientation=np.array([0.0, 180.0, -90.0]) * np.pi / 180.0 )

# Import Robots
c_objs = cop_from_scenic(pr, ex_world)


#############################################
pr.start()
pr.step()

### Robot Movement Code Goes Here ####

for i in range(2000):
    pr.step()
    if i % 100 == 0:
        print(i)

#########################################

pr.stop()


pr.shutdown()

print("Phew! We got home nice and safe...")
