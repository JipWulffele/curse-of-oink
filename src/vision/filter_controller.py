from src.filters.pig_tail import pig_tail_filter
from src.filters.pig_face import pig_nose_filter, pig_ear_left_filter, pig_ear_right_filter
from src.filters.base import add_filters

def apply_pig_filters(image, results, level):
    filters = []

    if level >= 1:
        filters.append(pig_tail_filter)
    if level >= 2:
        filters.extend([pig_nose_filter, pig_ear_left_filter, pig_ear_right_filter])
    # later add: level 3 color, level 4 full pig, level 5 bacon

    return add_filters(image, results, filters)