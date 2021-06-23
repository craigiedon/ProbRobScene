# python packages
import sys
import argparse
import numpy as np
import importlib
import os

# PyRep packages
from pyrep import PyRep
from pyrep.robots.end_effectors.panda_gripper import PandaGripper
from pyrep.objects import Camera, VisionSensor

# ProbRobScene files and methods
import probRobScene
from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs
from probRobScene.wrappers.coppelia import robotControl as rc
from probRobScene.wrappers.coppelia.prbCoppeliaWrapper import cop_from_prs


class Environment(object):
    """
    Environment for the robot
    """
    def __init__(self, prs_file_path, scene_file_path):
        self.prs_file = prs_file_path
        self.scene_file = scene_file_path

        self.pr = PyRep()

        self.scenario = probRobScene.scenario_from_file(self.prs_file)
    
    def scene_setup(self):
        """
        Setup the environment for the 
        """
        self.pr.launch(self.scene_file, headless=False, responsive_ui=True)

        # setup camera in the scene
        scene_view = Camera('DefaultCamera')
        scene_view.set_position([3.45, 0.18, 2.0])
        scene_view.set_orientation(np.array([180, -70, 90]) * np.pi / 180.0)

        ex_world, used_its = self.scenario.generate()

        # get all objects from the scenario
        objs = cop_from_prs(self.pr, ex_world)

        return objs