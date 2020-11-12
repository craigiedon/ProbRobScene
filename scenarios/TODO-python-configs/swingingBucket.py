from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.const import PrimitiveShape
from pyrep.objects import Camera, Shape, VisionSensor
import numpy as np
from wrappers import simExt, setupFuncs as sf

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

table = sf.create_table(pr, 1.0, 1.5, 0.5)

tray = pr.import_model("models/Tray.ttm")
sf.setAonB(tray, table, 0.0, -0.3)

c1 = Shape.create(PrimitiveShape.CUBOID, [0.05, 0.05, 0.05], mass=0.01, color=[0.7, 0.0, 0.0])
c2 = Shape.create(PrimitiveShape.CUBOID, [0.05, 0.05, 0.05], mass=0.01, color=[0.0, 0.7, 0.0])
c3 = Shape.create(PrimitiveShape.CUBOID, [0.05, 0.05, 0.05], mass=0.01, color=[0.0, 0.0, 0.7])

sf.setAonB(c1, tray)
sf.setAonB(c2, c1)
sf.setAonB(c3, c2)

sf.setAonPos(table, [0.0, 0.0, 0.0])
sf.setAonB(panda_1, table, -0.4, 0.0)

rope = sf.create_rope(pr, 24)
rope.set_position([0.1, 0.5, 1.5])
bucket = pr.import_model("models/Bucket.ttm")
sf.attach_to_rope(pr, rope, bucket)




######################


pr.start()


### Robot Code goes here ###

for i in range(10000):
    # Just apply random forces to the bucket at each step for some wobble (theres probably a more disciplined way to simulate wind etc)
    simExt.add_force(bucket.get_handle(), [0.0, 0.0, 0.0], [np.random.uniform(-2.0, 2.0), np.random.uniform(-2.0, 2.0), 0.0])
    pr.step()
    if i % 100 == 0:
        print(i)


############################

pr.stop()
pr.shutdown()
