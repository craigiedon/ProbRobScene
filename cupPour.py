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
from setupFuncs import setAonB, setAonPos, create_table, top_of
import robotControl as rc

pr = PyRep()
pr.launch("scenes/emptyBullet28.ttt", headless=False, responsive_ui=True)

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
# setAonPos(cups_table, [0,])
c1 = pr.import_model("models/FilledCup.ttm")
c2 = pr.import_model("models/FilledCup.ttm")
c3 = pr.import_model("models/FilledCup.ttm")

setAonB(panda_1, cups_table, -0.3)
setAonB(c1, cups_table, 0.1, 0.0)
setAonB(c2, cups_table, 0.1, 0.2)
setAonB(c3, cups_table, 0.1, -0.2)

bowl_table = create_table(pr, 0.6, 0.6, 0.5)
setAonPos(bowl_table, [-1.0, 0.0, 0.0])
bowl = pr.import_model("models/Bowl.ttm")
setAonB(bowl, bowl_table)


pr.start()

## TODO: Actually solve manipulation task
rc.move_above_object(pr, panda_1, c1, z_offset=0.1)

pr.stop()
pr.shutdown()
