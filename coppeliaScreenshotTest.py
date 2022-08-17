from pyrep import PyRep
import probRobScene

from probRobScene.wrappers.coppelia.takeScreenshot import take_screenshot

if __name__=='__main__':

    scenario = probRobScene.scenario_from_file("scenarios/gearInsert.prs")
    scene, _ = scenario.generate()

    take_screenshot(scene, cameraPos = [-2, 2, 4])
