from pyrep.backend._sim_cffi import ffi, lib
from typing import List, Union
import numpy as np


def add_force(shape_handle: int, position: Union[List[float], np.ndarray], force: Union[List[float], np.ndarray]) -> None:
    lib.simAddForce(shape_handle, position, force)


def add_force_and_torque(shape_handle: int, force: Union[List[float], np.ndarray], torque: Union[List[float], np.ndarray]) -> None:
    lib.simAddForceAndTorque(shape_handle, force, torque)
