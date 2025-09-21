import cv2
import numpy as np
from src.filters.base import overlay_sticker_from_landmarks

# Load pig tail sticker
pig_tail_img = cv2.imread("assets/stickers/pig_tail.png", cv2.IMREAD_UNCHANGED)

# Sticker source points (base-left, base-right, tip of curl)
pig_tail_src_pts = np.array([
    [10, 0],   # base left
    [0, 220],   # base right
    [40, 220]      # tip
])

# Use left & right hip landmarks (see https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker?hl=fr)
pig_tail_landmarks = [23, 24]

def pig_tail_filter(image, results):
    if not results.pose_landmarks:
        return image
    return overlay_sticker_from_landmarks(
        image,
        pig_tail_img,
        pig_tail_src_pts,
        pig_tail_landmarks,
        results,
        landmark_type="pose",
        show_landmarks=False,
        tip_offset=(0, 1.5)  # place...
    )