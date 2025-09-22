import cv2
from src.filters.mask_warp import load_mask_points, warp_mask_onto_face

PIG_MASK = cv2.imread("assets/stickers/pig_full.png", cv2.IMREAD_UNCHANGED)
FACE_INDICES, MASK_POINTS = load_mask_points("assets/stickers/pig_full_points.csv")

def pig_full_filter(image, results):
    if not results.face_landmarks:
        return image
    return warp_mask_onto_face(image, results, PIG_MASK, FACE_INDICES, MASK_POINTS)