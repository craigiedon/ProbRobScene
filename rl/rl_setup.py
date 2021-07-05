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

#python packages
import env_setup
import agent_setup

def main():
    if len(sys.argv) != 6:
        print("Incorret argument")
        print("python env_setup.py <path-to-scenario-file> <path-to-scene-file> <path-to-python-file> <episode-length> <number-of-episodes>")

    parser = argparse.ArgumentParser()

    parser.add_argument('probRobScene_file', type=str, help = 'Path to probRobScene scenario file (.prs)')
    parser.add_argument('scene_file', type=str, help='Path to scene file (.ttt)')
    parser.add_argument('agent_python_file', type=str, help='Agent python file name (.py)')
    parser.add_argument('episode_legnth', type=int, help='Length of a single episode')
    parser.add_argument('episodes', type=int, help='Number of episodes')

    args = parser.parse_args()

    PATH_TO_PRS_FILE = args.probRobScene_file
    PATH_TO_SCENE_FILE = args.scene_file
    PATH_TO_PYTHON_FILE = args.agent_python_file
    EPISODE_LENGTH = args.episode_legnth
    EPISODES = args.episodes

    pr = PyRep()

    # Environment setup for the agent
    env = env_setup.Environment(PATH_TO_PRS_FILE, PATH_TO_SCENE_FILE)

    # accesesing methods associated with the agent
    spec = importlib.util.spec_from_file_location('agent', PATH_TO_PYTHON_FILE)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)

    inputs = []
    reward = []

    #Agent setup
    rl_agent = agent_setup.Agent(PATH_TO_SCENE_FILE, render_mode='rgb_array')

    # Camera setup for the environment
    scene_view = Camera('DefaultCamera')
    scene_view.set_position([3.45, 0.18, 2.0])
    scene_view.set_orientation(np.array([180, -70, 90]) * np.pi / 180.0)

    for episode in range(EPISODES):
        observation = rl_agent.reset()
        for i in range(EPISODE_LENGTH):
            action = rl_agent.act(observation)
            im = rl_agent.render('rgb_array')
            observation, reward, done, info = rl_agent.step(action)

            if done:
                print('EPISODE COMPLETE AFTER {} timestamps in EPISODE {}'.format(i+1, episode))
                print(info)
                break
    rl_agent.close()

if __name__ == "__main__":
    main()