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
table = create_table(pr, 0.75, 0.75, 0.8)
gear = pr.import_model("models/HexagonalGear.ttm")
# gear.set_bullet_friction(0.3)
# gear.set_bullet_friction(0.99)
g_base = pr.import_model("models/HexagonalPegBase.ttm")

setAonB(panda_1, table, -0.3)

setAonB(gear, table, 0.0, 0.2)
setAonB(g_base, table, 0.0, -0.2)

pr.start()

rc.move_above_object(pr, panda_1, gear, z_offset=-0.02, ig_cols=True)
rc.move_to_pos(pr, panda_1, top_of(table) + np.array([-0.4, 0.2, 0.0]), ig_cols=True)

for i in range(1000):
    pr.step()
    if i % 100 == 0:
        print(i)

pr.stop()
pr.shutdown()