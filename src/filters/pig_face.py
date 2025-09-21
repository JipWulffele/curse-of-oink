import cv2
import numpy as np
from src.filters.base import overlay_sticker_from_landmarks

# Load stickers
pig_nose_img = cv2.imread("assets/stickers/pig_nose.png", cv2.IMREAD_UNCHANGED)
pig_ear_left_img = cv2.imread("assets/stickers/pig_ear_left.png", cv2.IMREAD_UNCHANGED)
pig_ear_right_img = cv2.imread("assets/stickers/pig_ear_right.png", cv2.IMREAD_UNCHANGED)

# Sticker configs
pig_nose_src_pts = np.array([[70,0],[0, 60],[140,60]])  
pig_nose_landmarks = [195, 48, 278]  

pig_ear_left_src_pts = np.array([[125,160],[200,90],[25,20]])
pig_ear_left_landmarks = [127, 54]

pig_ear_right_src_pts = np.array([[0,90],[80,165],[190,20]])
pig_ear_right_landmarks = [284, 356]

def pig_nose_filter(image, results):
    if not results.face_landmarks:
        return image
    return overlay_sticker_from_landmarks(
        image, pig_nose_img, pig_nose_src_pts, pig_nose_landmarks, results, "face"
    )

def pig_ear_left_filter(image, results):
    if not results.face_landmarks:
        return image
    return overlay_sticker_from_landmarks(
        image, pig_ear_left_img, pig_ear_left_src_pts, pig_ear_left_landmarks,
        results, "face", tip_offset=(-1.5, -0.5)
    )

def pig_ear_right_filter(image, results):
    if not results.face_landmarks:
        return image
    return overlay_sticker_from_landmarks(
        image, pig_ear_right_img, pig_ear_right_src_pts, pig_ear_right_landmarks,
        results, "face", tip_offset=(1.5, -0.5)
    )
