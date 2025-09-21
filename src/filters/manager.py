from src.filters.pig_tail import pig_tail_filter
from src.filters.pig_face import pig_nose_filter, pig_ear_left_filter, pig_ear_right_filter
from src.filters.pig_vision import pig_vision_filter

def apply_filters(image, results, pig_level):
    if pig_level == 0:
        return image
    elif pig_level == 1:
        return pig_tail_filter(image, results)
    elif pig_level == 2:
        img = pig_tail_filter(image, results)
        img = pig_nose_filter(img, results)
        img = pig_ear_left_filter(img, results)
        img = pig_ear_right_filter(img, results)
        return img
    elif pig_level == 3:
        img = pig_tail_filter(image, results)
        img = pig_nose_filter(img, results)
        img = pig_ear_left_filter(img, results)
        img = pig_ear_right_filter(img, results)
        img = pig_vision_filter(img, intensity=0.8, blur_ksize=3)
        return img
    # add later levels ( 4 full pig, 5 bacon)
    return image