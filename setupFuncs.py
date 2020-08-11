from pyrep.backend import sim
from pyrep.backend.utils import script_call


def setAonB(a_obj, b_obj, x_offset=0.0, y_offset=0.0):
    if a_obj.is_model():
        a_bottom = a_obj.get_model_bounding_box()[-2]
    else:
        a_bottom = a_obj.get_bounding_box()[-2]

    if b_obj.is_model():
        b_top = b_obj.get_model_bounding_box()[-1]
    else:
        b_top = b_obj.get_bounding_box()[-1]

    bx, by, bz = b_obj.get_position()

    a_obj.set_position([bx + x_offset, by + y_offset, bz + b_top - a_bottom])


def setABelowB(a_obj, b_obj, offset=None):
    if offset is None:
        offset = [0.0, 0.0, 0.0]

    a_top = a_obj.get_model_bounding_box()[-1] if a_obj.is_model() else a_obj.get_bounding_box()[-1]
    b_bottom = b_obj.get_model_bounding_box()[-2] if b_obj.is_model() else b_obj.get_bounding_box()[-2]

    bx, by, bz = b_obj.get_position()

    a_obj.set_position([bx + offset[0], by + offset[1], bz + b_bottom - a_top + offset[2]])



def setAonPos(a_obj, pos):
    if a_obj.is_model():
        a_bottom = a_obj.get_model_bounding_box()[-2]
    else:
        a_bottom = a_obj.get_bounding_box()[-2]

    a_obj.set_position([pos[0], pos[1], pos[2] - a_bottom])


def create_table(pr, length: float, width: float, height: float):
    table = pr.import_model("models/Table.ttm")
    print(table.get_name())

    # Its not @Table, its at the name of this specific table!
    script_call("table_length_callback@{}".format(table.get_name()), sim.sim_scripttype_customizationscript, floats=[length])
    script_call("table_width_callback@{}".format(table.get_name()), sim.sim_scripttype_customizationscript, floats=[width])
    script_call("table_height_callback@{}".format(table.get_name()), sim.sim_scripttype_customizationscript, floats=[height])

    return table


def create_rope(pr, num_links: int):
    # Load rope model
    rope_root = pr.import_model("models/RopeLink.ttm")
    rope_parent = rope_root
    print("Dynamic?: ", rope_root.is_model_dynamic())

    for i in range(num_links - 1):
        # Make a copy
        rope_child = pr.import_model("models/RopeLink.ttm")
        rope_child.set_model(False)

        parent_joint = rope_parent.get_objects_in_tree(exclude_base=True)[0]
        rope_child.set_parent(parent_joint)

        setABelowB(rope_child, rope_parent, [0.0, 0.0, -0.001])

        rope_parent = rope_child

    return rope_root


def attach_to_rope(pr, rope, obj):
    final_joint = rope.get_objects_in_tree()[-1]
    obj.set_parent(final_joint)
    setABelowB(obj, final_joint)


def top_of(obj):
    x, y, z = obj.get_position()
    if obj.is_model():
        b_box_top = obj.get_model_bounding_box()[-1]
    else:
        b_box_top = obj.get_bounding_box()[-1]

    return [x, y, z + b_box_top]
