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


def top_of(obj):
    x, y, z = obj.get_position()
    if obj.is_model():
        b_box_top = obj.get_model_bounding_box()[-1]
    else:
        b_box_top = obj.get_bounding_box()[-1]

    return z + b_box_top
