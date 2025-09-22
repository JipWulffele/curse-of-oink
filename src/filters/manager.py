import cv2

from src.filters.pig_tail import pig_tail_filter
from src.filters.pig_face import pig_nose_filter, pig_ear_left_filter, pig_ear_right_filter
from src.filters.pig_vision import pig_vision_filter
from src.filters.bacon_head import bacon_head_filter, pork_chop_hand_filter
from src.filters.pig_full import pig_full_filter
from src.filters.mask_wrapper_vsn0 import load_mask_points, warp_mask_onto_face

# Load pig mask (RGBA)
pig_mask = cv2.imread("assets/stickers/pig_full.png", cv2.IMREAD_UNCHANGED)
# Load mapping points
face_indices, mask_points = load_mask_points("assets/stickers/pig_full_points.csv", pig_mask.shape[:2])


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
    elif pig_level == 4:
        return pig_full_filter(image, results)
        #return warp_mask_onto_face(image, results, pig_mask, face_indices, mask_points)
    elif pig_level == 5:
        img = bacon_head_filter(image, results)
        img = pork_chop_hand_filter(image, results, side='left')
        img = pork_chop_hand_filter(image, results, side='right')
        return img
    return image