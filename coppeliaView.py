import probRobScene
from pyrep import PyRep
from pyrep.objects import Camera
import numpy as np
import sys
from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs


if len(sys.argv) != 2:
    print("python3 coppeliaView.py <path-to-scenario-file>")
    sys.exit(0)

scenario_file = sys.argv[1]
scenario = probRobScene.scenario_from_file(scenario_file)

pr = PyRep()
pr.launch("scenes/emptyVortex.ttt", headless=False, responsive_ui=True)

ex_world, used_its = scenario.generate()

try:
    c_objs = cop_from_prs(pr, ex_world)

    pr.start()
    pr.step()

    pr.stop()
    input("Simulation Finished. To Quit, Press Enter")
    pr.shutdown()

except:
    pr.stop()
    pr.shutdown()