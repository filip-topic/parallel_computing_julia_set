import math

CURVE_START = 48/64*math.pi
CURVE_END = 60/64*math.pi
CURVE_SPAN = CURVE_END-CURVE_START
CURVE_SCALE = 0.755

def c_from_group(group_size: int, group_number: int):
    if group_size is None or group_number is None:
        raise Exception("Please provide your group size and number "+
                        "to the GROUP_SIZE and GROUP_NUMBER variables.")

    # argument checking
    if group_size < 2 or group_size > 3:
        raise Exception("Group size must be either 2 or 3")

    if group_number < 1 or group_number > 30:
        raise Exception("Group number must be between 1 and 30")

    if group_size == 2:
        num_groups = 30
        phi = 2*math.pi-CURVE_END + group_number/(num_groups-1)*CURVE_SPAN
    elif group_size == 3:
        num_groups = 20
        phi = CURVE_END - group_number/(num_groups-1)*CURVE_SPAN
    
    return CURVE_SCALE*math.e**(phi*1j)

