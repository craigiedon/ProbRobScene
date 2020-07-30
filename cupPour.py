from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects import Object
from pyrep.backend.utils import script_call
from pyrep.const import PrimitiveShape
from pyrep.objects import Camera, Shape, VisionSensor
from pyrep.backend import sim
from multiprocessing import Process
import numpy as np
import matplotlib.pyplot as plt
from setupFuncs import setAonB, create_table, top_of
import robotControl as rc

pr = PyRep()
pr.launch("", headless=False, responsive_ui=True)

scene_view = Camera('DefaultCamera')
scene_view.set_position([3.45, 0.18, 2.0])
scene_view.set_orientation(np.array([180, -70, 90]) * np.pi / 180.0)

pr.step_ui()

depth_cam = VisionSensor.create([256, 256], position=[0, 0, 2.0], orientation=np.array([0.0, 180.0, -90.0]) * np.pi / 180.0 )

# Import Robots
pr.import_model("models/Panda.ttm")
panda_1 = Panda(0)
gripper_1 = PandaGripper(0)

# Prop Creation
cups_table = create_table(pr, 0.75, 0.75, 0.8)
# cups_table.set_position([2.0, 2.0, 0.0])
c1 = pr.import_model("models/Cup.ttm")
c2 = pr.import_model("models/Cup.ttm")
c3 = pr.import_model("models/Cup.ttm")

setAonB(panda_1, cups_table, -0.5, -0.5)
setAonB(c1, cups_table, 0.0, 0.0)
setAonB(c2, cups_table, 0.0, 0.2)
setAonB(c3, cups_table, 0.0, 0.4)

bowl_table = create_table(pr, 0.6, 0.6, 0.6)
bowl_table.set_position([-1.0, -1.0, 0.0])
bowl = pr.import_model("models/Bowl.ttm")
setAonB(bowl, bowl_table)


pr.start()

pr.stop()
for i in range(2000):
    pr.step()
    if i % 100 == 0:
        print(i)

pr.stop()
pr.shutdown()
