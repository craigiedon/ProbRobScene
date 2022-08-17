import sys
import probRobScene
from probRobScene.wrappers.coppelia.takeScreenshot import take_screenshot

if len(sys.argv) != 2:
    print("python3 coppeliaScreenshot.py <path-to-scenario-file>")
    sys.exit(0)

scenario_file = sys.argv[1]
scenario = probRobScene.scenario_from_file(scenario_file)
scene, _ = scenario.generate()

take_screenshot(scene, cameraPos = [-2, 2, 4])
