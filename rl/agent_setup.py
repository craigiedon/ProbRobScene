import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects import Shape
from pyrep.robots.arms.panda import Panda
from pyrep.robots.arms.dobot import Dobot
from pyrep.objects import VisionSensor
from typing import Union


class Agent(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, scene_file_path, render_mode: Union[None, str] = None, action_noise_mean=0.0, action_noise_var=0.0, headless=False, control_loop_enabled=False):
        self.action_noise_mean = action_noise_mean
        self.action_noise_variance = action_noise_var
        self.POS_MIN = np.array([0.8, -0.2, 1.0])
        self.POS_MAX = np.array([1.0, 0.2, 1.4])

        self.pr = PyRep()

        self.pr.launch(scene_file_path, headless=headless)

        if render_mode is not None:
            self.camera = VisionSensor.create([512, 512],
                                              position=[2.475, -0.05, 1.9],
                                              orientation=np.array([-180.0, -65.0, 90.0]) * np.pi / 180.0)
            print(self.camera.get_render_mode())
        self.pr.start()
        self.panda = Panda()
        self.panda.set_control_loop_enabled(control_loop_enabled)
        self.panda.set_motor_locked_at_zero_velocity(True)
        self.target = Shape.create(type=PrimitiveShape.SPHERE,
                                   size=[0.05, 0.05, 0.05],
                                   color=[1.0, 0.1, 0.1],
                                   static=True, respondable=False)
        self.panda_ee_tip = self.panda.get_tip()
        self.initial_joint_positions = self.panda.get_joint_positions()

        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(7,))

        self.observation_space = spaces.Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, *self.POS_MIN]),
                                            high=np.array([2.8973, 1.7628, 2.8973, 0.0698, 2.8973, 3.7525, 2.8973, *self.POS_MAX]))

        # self._config_env()

    def _config_env(self):
        pass
    
    def act(self, obs) -> np.array:
        return np.random.uniform(-1.0, 1.0, size=(4,))

    def reward(self):
        ax, ay, az = self.panda_ee_tip.get_position()
        tx, ty, tz = self.target.get_position()
        # Reward is negative distance to target
        return -np.sqrt((ax - tx) ** 2 + (ay - ty) ** 2 + (az - tz) ** 2)

    def is_over(self):
        p_tip = np.array(self.panda_ee_tip.get_position())
        target_pos = np.array(self.target.get_position())

        dist_to_target = np.linalg.norm(p_tip - target_pos)
        return dist_to_target <= 0.05

    def _get_state(self):
        return np.concatenate([self.panda.get_joint_positions(), self.target.get_position()])

    def step(self, action):
        noise = np.random.normal(self.action_noise_mean, self.action_noise_variance, len(action))
        self.panda.set_joint_target_velocities(action + noise)  # Execute action on arm, with actuator noise
        self.pr.step()  # Step the physics simulation
        return self._get_state(), self.reward(), self.is_over(), {}

    def reset(self):
        pos = list(np.random.uniform(self.POS_MIN, self.POS_MAX))
        self.target.set_position(pos)
        self.panda.set_joint_positions(self.initial_joint_positions, True)
        return self._get_state()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.camera.capture_rgb()

    def close(self):
        self.pr.stop()
        self.pr.shutdown()