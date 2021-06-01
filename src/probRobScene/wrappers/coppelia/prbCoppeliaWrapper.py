from collections import defaultdict
from typing import Union, Dict

import numpy as np
from numpy.core.multiarray import ndarray
from pyrep import PyRep
from pyrep.const import PrimitiveShape
from pyrep.objects import Shape, VisionSensor
from probRobScene.core.scenarios import Scene
from probRobScene.wrappers.coppelia.setupFuncs import create_table, create_rope, attach_to_rope


def cop_from_prs(pr: PyRep, scene: Scene) -> Dict[str, list]:
    cop_obs = defaultdict(list)
    primitives = (p for p in scene.objects if hasattr(p, 'shape_type'))
    models = (m for m in scene.objects if hasattr(m, 'model_name'))
    for p in primitives:
        c_obj = Shape.create(type=PrimitiveShape[p.shape_type],
                             position=p.position,
                             color=[0.5, 0.0, 0.0],
                             size=[p.width, p.length, p.height],
                             visible_edges=True,
                             mass=0.01)
        cop_obs[p.shape_type].append(c_obj)

    for m in models:
        # print(m.model_name)
        if m.model_name == "Camera":
            c_obj = VisionSensor.create([256, 256], position=m.position, orientation=[np.pi, 0.0, 0.0])
        elif m.model_name == "Table":
            c_obj = create_table(pr, m.width, m.length, m.height)
            c_obj.set_position(prs_to_coppelia_pos(m.position, c_obj))
            c_obj.set_orientation(reversed(m.orientation))
        elif m.model_name == "Panda":
            c_obj = pr.import_model(f'models/{m.model_name}.ttm')
            bounds = c_obj.get_model_bounding_box()
            dims = np.array(bounds[1::2]) - np.array(bounds[0::2])
            adjusted_position = prs_to_coppelia_pos(m.position, c_obj) + np.array([0, dims[1] / 2.0 - m.length / 2.0, 0.0])
            c_obj.set_position(adjusted_position)
            # c_obj.set_position(scenic_to_coppelia_pos(m.position, c_obj))
            c_obj.set_orientation(reversed(m.orientation))
        elif m.model_name == "RopeBucket":
            c_obj = create_rope(pr, m.num_rope_links)
            c_obj.set_position(prs_to_coppelia_pos(m.position, c_obj))
            c_obj.set_orientation(reversed(m.orientation))
            c_obj_two = pr.import_model('models/Bucket.ttm')
            attach_to_rope(pr, c_obj, c_obj_two)
        else:
            c_obj = pr.import_model(f'models/{m.model_name}.ttm')
            c_obj.set_position(prs_to_coppelia_pos(m.position, c_obj))
            c_obj.set_orientation(reversed(m.orientation))
        cop_obs[m.model_name].append(c_obj)

    return cop_obs


def prs_to_coppelia_pos(prs_pos, coppelia_model) -> Union[list, ndarray]:
    """ Scenic always refers to the position as the *object-centre*
        Coppelia-Sim positions objects based on their *origin_point*"""

    # Get dimensions of coppelia object (plus and minus offsets) Look up this function in the pyrep wrapper
    bounds = coppelia_model.get_model_bounding_box()
    bounds_min, bounds_max = np.array(bounds[0::2]), np.array(bounds[1::2])
    # print(f"Min b: {bounds_min}: \n Max b: {bounds_max}")

    new_pos = np.array(prs_pos) - (bounds_max + bounds_min) / 2.0

    return new_pos
