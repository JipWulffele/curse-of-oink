import cv2
import numpy as np
import csv
from scipy.spatial import Delaunay

def load_mask_points(csv_path, target_img_shape):
    """
    Load annotated mask points from MakeSense export.
    Args:
        csv_path: path to CSV
        target_img_shape: (h, w) of the pig mask PNG
    Returns:
        face_indices: list of FaceMesh landmark indices
        mask_points: np.ndarray of (x,y) in pig mask coords
    """
    face_indices = []
    mask_points = []

    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            face_idx = int(row[0])
            x = float(row[1])
            y = float(row[2])
            # normalize to pig image size
            mask_points.append((x, y))
            face_indices.append(face_idx)

    return face_indices, np.array(mask_points, dtype=np.float32)


def warp_mask_onto_face(frame_bgr, results, pig_mask_rgba, face_indices, mask_points):
    """
    Warp pig mask (RGBA) onto face using triangulated landmarks.
    """
    if not results.face_landmarks:
        return frame_bgr

    h, w, _ = frame_bgr.shape
    lm = results.face_landmarks.landmark

    # Collect corresponding destination points from face
    dst_points = np.array(
        [[lm[idx].x * w, lm[idx].y * h] for idx in face_indices],
        dtype=np.float32
    )

    # Triangulate on mask coordinates
    tri = Delaunay(mask_points)
    frame_out = frame_bgr.copy()

    for simplex in tri.simplices:
        src_tri = mask_points[simplex]
        dst_tri = dst_points[simplex]

        # Warp each triangle
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
    roi = (roi * (1 - alpha) + warped_masked[..., :3] * alpha).astype(np.uint8)

    dst_bgr[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = roi
    return dst_bgr
