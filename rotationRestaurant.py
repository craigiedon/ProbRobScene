from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.const import PrimitiveShape
from pyrep.objects import Camera, Shape, VisionSensor
import numpy as np
import setupFuncs as sf

### SETUP CODE ###

pr = PyRep()
pr.launch("scenes/emptyVortex.ttt", headless=False, responsive_ui=True)

scene_view = Camera('DefaultCamera')
scene_view.set_position([3.45, 0.18, 2.0])
scene_view.set_orientation(np.array([180, -70, 90]) * np.pi / 180.0)

pr.step_ui()

depth_cam = VisionSensor.create([256, 256], position=[0, 0, 2.0], orientation=np.array([0.0, 180.0, -90.0]) * np.pi / 180.0 )

pr.import_model("models/Panda.ttm")
panda_1 = Panda(0)
gripper_1 = PandaGripper(0)

table = sf.create_table(pr, 0.85, 0.85, 0.75)
chair = pr.import_model("models/DiningChair.ttm")
chair.set_orientation([0.0, 0.0, -np.pi / 2.0]) # Could be "set facing?"
conveyor_belt = pr.import_model("models/CircularConveyorBelt.ttm")
plate = pr.import_model("models/Plate.ttm")


c1 = Shape.create(PrimitiveShape.CYLINDER, [0.05, 0.05, 0.05], 0.01, color=[0.3, 0.45, 0.0])
c2 = Shape.create(PrimitiveShape.CYLINDER, [0.05, 0.05, 0.05], 0.01, color=[0.3, 0.45, 0.0])
c3 = Shape.create(PrimitiveShape.CYLINDER, [0.05, 0.05, 0.05], 0.01, color=[0.3, 0.45, 0.0])

sf.setAonPos(table, [0.0, -0.5, 0.0])
sf.setAonPos(chair, [0.55, -0.5, 0.0])
sf.setAonPos(panda_1, [0.0, 0.25, 0.0])
sf.setAonPos(conveyor_belt, [0.0, 1.5, 0.0])
sf.setAonB(plate, conveyor_belt, 0.0, 0.75)
sf.setAonB(c1, plate)
sf.setAonB(c2, plate, 0.05)
sf.setAonB(c3, plate, -0.05)

###################

pr.start()

### Robot Movement Code Goes Here ###


for i in range(4000):
    pr.step()
    if i % 100 == 0:
        print(i)

######################################

pr.stop()
pr.shutdown()