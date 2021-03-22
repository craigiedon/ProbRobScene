import probRobScene
from pyrep import PyRep
from pyrep.robots.arms.panda import Panda
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects import Camera
import numpy as np
from wrappers import robotControl as rc
import sys
from wrappers.prbCoppeliaWrapper import cop_from_prs

pr = PyRep()

if len(sys.argv) != 2:
    print("python3 copperliaView.py <path-to-scenario-file>")
    sys.exit(0)

scenario_file = sys.argv[1]
scenario = probRobScene.scenario_from_file(scenario_file)

max_sims = 1
sim_result = []
inputs = []
for i in range(max_sims):
    pr.launch("scenes/emptyVortex.ttt", headless=False, responsive_ui=True)

    scene_view = Camera('DefaultCamera')
    scene_view.set_position([3.45, 0.18, 2.0])
    scene_view.set_orientation(np.array([180, -70, 90]) * np.pi / 180.0)

    ex_world, used_its = scenario.generate()
    inputs.append(ex_world)

    c_objs = cop_from_prs(pr, ex_world)

    pr.start()
    pr.step()

    try:
        for steps in range(10000):
            pr.step()
            if steps%100 == 0:
                print(f"Step: {steps}")
    except:
        print('Yikes!')
        pr.shutdown()

print("Phew! We got home nice and safe...")

print(len(sim_result))
print(sim_result)