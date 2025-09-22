import cv2
import numpy as np
import csv
from scipy.spatial import Delaunay


def load_mask_points(csv_path):
    """
    Load annotated mask points from MakeSense export.
    CSV format: face_index, x, y
    Returns:
        face_indices: list of FaceMesh landmark indices (int or -1 if extrapolated)
        mask_points: np.ndarray of (x,y) in mask coords
    """
    face_indices = []
    mask_points = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            face_idx = int(row[0])
            x = float(row[1])
            y = float(row[2])
            face_indices.append(face_idx)
            mask_points.append((x, y))
    return face_indices, np.array(mask_points, dtype=np.float32)

def compute_extra_landmarks(results, image_shape):
    """
    Compute synthetic points (ears, neck) since FaceMesh has none.
    Returns dict of (x,y) in image coordinates.
    """
    if not results.face_landmarks:
        return {}

    h, w, _ = image_shape
    lm = results.face_landmarks.landmark

    # Jawline & temples for extrapolation
    jaw_left  = np.array([lm[234].x * w, lm[234].y * h])
    jaw_right = np.array([lm[454].x * w, lm[454].y * h])
    chin      = np.array([lm[152].x * w, lm[152].y * h])
    temple_left  = np.array([lm[127].x * w, lm[127].y * h])
    temple_right = np.array([lm[356].x * w, lm[356].y * h])

    # Extrapolate ears outward
    ear_left  = temple_left  + 1.2 * (temple_left  - jaw_left)
    ear_right = temple_right + 1.2 * (temple_right - jaw_right)

    # Neck points below chin
    neck_left   = jaw_left  + (chin - jaw_left) * 0.6
    neck_right  = jaw_right + (chin - jaw_right) * 0.6
    neck_center = (jaw_left + jaw_right) / 2 + (chin - (jaw_left + jaw_right) / 2) * 0.6

    return {
        "ear_left": tuple(ear_left),
        "ear_right": tuple(ear_right),
        "neck_left": tuple(neck_left),
        "neck_center": tuple(neck_center),
        "neck_right": tuple(neck_right),
    }


def warp_mask_onto_face(frame_bgr, results, pig_mask_rgba, face_indices, mask_points):
    """
    Warp pig mask (RGBA) onto face using triangulated landmarks.
    """
    if not results.face_landmarks:
        return frame_bgr

    h, w, _ = frame_bgr.shape
    lm = results.face_landmarks.landmark

    # Base points (from mediapipe indices)
    dst_points = []
    for idx in face_indices:
        dst_points.append((lm[idx].x * w, lm[idx].y * h))
    dst_points = list(dst_points)
    mask_points = list(mask_points)

    # Add extrapolated ears/neck
    extras = compute_extra_landmarks(results, frame_bgr.shape)
    if extras:
        extra_pig_points = {
            "ear_left":   (10, 0),
            "ear_right":  (520, 40),
            "neck_left":  (50, 600),
            "neck_center": (250, 710),
            "neck_right": (500, 600),
        }

        for key in ["ear_left", "ear_right", "neck_left", "neck_center", "neck_right"]:
            dst_points.append(extras[key])
            mask_points.append(extra_pig_points[key])

    dst_points = np.array(dst_points, dtype=np.float32)
    mask_points = np.array(mask_points, dtype=np.float32)

    # Triangulate on mask coordinates
    tri = Delaunay(mask_points)
    frame_out = frame_bgr.copy()

    for simplex in tri.simplices:
        src_tri = mask_points[simplex]
        dst_tri = dst_points[simplex]
        frame_out = warp_triangle(pig_mask_rgba, frame_out, src_tri, dst_tri)

    return frame_out


def warp_triangle(src_rgba, dst_bgr, src_tri, dst_tri):
    """
    Warp triangular region of RGBA image onto BGR frame.
    """
    r1 = cv2.boundingRect(np.float32([src_tri]))
    r2 = cv2.boundingRect(np.float32([dst_tri]))

    src_rect = src_rgba[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    src_tri_rect = src_tri - np.array([r1[0], r1[1]])
    dst_tri_rect = dst_tri - np.array([r2[0], r2[1]])

    M = cv2.getAffineTransform(np.float32(src_tri_rect), np.float32(dst_tri_rect))
    warped = cv2.warpAffine(src_rect, M, (r2[2], r2[3]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)

    # mask for blending
    mask = np.zeros((r2[3], r2[2], 4), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_tri_rect), (255, 255, 255, 255))

    warped_masked = cv2.bitwise_and(warped, mask)

    # extract ROI from destination
    roi = dst_bgr[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    # alpha blend
    alpha = warped_masked[..., 3:] / 255.0

    # ensure same size
    h = min(roi.shape[0], warped_masked.shape[0])
    w = min(roi.shape[1], warped_masked.shape[1])

    roi_c = roi[:h, :w]
    warped_rgb_c = warped_masked[:h, :w, :3]
    alpha_c = alpha[:h, :w]

    roi[:h, :w] = (roi_c * (1 - alpha_c) + warped_rgb_c * alpha_c).astype(np.uint8)

    dst_bgr[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = roi
    return dst_bgr
