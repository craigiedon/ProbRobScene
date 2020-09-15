from typing import Union

import numpy as np
from numpy.core.multiarray import ndarray
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects import Shape
from scenic3d.core.scenarios import Scene


class CoppeliaScene:
    def __init__(self, objs):
        self.objects = objs


def cop_from_scenic(pr: PyRep, scene: Scene) -> CoppeliaScene:
    cop_obs = []
    primitives = (p for p in scene.objects if hasattr(p, 'shape_type'))
    models = (m for m in scene.objects if hasattr(m, 'model_name'))
    for p in primitives:
        c_obj = Shape.create(type=PrimitiveShape[p.shape_type],
                             position=p.position,
                             color=[0.5, 0.0, 0.0],
                             size=[p.width, p.length, p.height],
                             visible_edges=True,
                             mass=0.01)
        cop_obs.append(c_obj)

    for m in models:
        print(m.model_name)
        c_obj = pr.import_model(f'models/{m.model_name}.ttm')
        c_obj.set_position(scenic_to_coppelia_pos(m.position, c_obj))

    return CoppeliaScene(cop_obs)


def scenic_to_coppelia_pos(scenic_pos, coppelia_model) -> Union[list, ndarray]:
    """ Scenic always refers to the position as the *object-centre*
        Coppelia-Sim positions objects based on their *origin_point*"""

    # Get dimensions of coppelia object (plus and minus offsets) Look up this function in the pyrep wrapper
    bounds = coppelia_model.get_model_bounding_box()
    bounds_min = np.array(bounds[0::2])
    bounds_max = np.array(bounds[1::2])
    print("Min bounds: ", bounds_min)
    print("Max bounds: ", bounds_max)

    instance_bounds = coppelia_model.get_bounding_box()
    inst_b_min = np.array(instance_bounds[0::2])
    inst_b_max = np.array(instance_bounds[1::2])

    print("Inst Min bounds: ", bounds_min)
    print("Inst Max bounds: ", bounds_max)

    new_pos = np.array(scenic_pos) - (bounds_max + bounds_min) / 2.0

    return new_pos
