import cv2
import numpy as np

def overlay_sticker_from_landmarks(
    image, sticker_img,
    src_pts, landmark_indices,
    results,
    landmark_type="face",
    show_landmarks=False, landmarks_color=(0,255,0),
    tip_offset=None
):
    """
    Overlay a sticker onto the image using landmark indices.

    Args:
        image (np.ndarray): BGR frame.
        sticker_img (np.ndarray): Sticker image (RGBA or BGR).
        src_pts (np.ndarray): 3x2 points on the sticker (base-left, base-right, tip).
        landmark_indices (list[int]): Landmark indices for the sticker anchors (2 or 3 points).
        results: MediaPipe results object.
        landmark_type (str): "face", "left_hand", "right_hand", "pose".
        show_landmarks (bool): Draw the destination points for debugging.
        landmarks_color (tuple): Color of landmark markers.
        tip_offset (tuple or None): (dx, dy) offset in pixels or normalized coordinates to compute tip outside head.

    Returns:
        np.ndarray: Image with sticker overlaid.
    """
    # 1. Select landmarks
    if landmark_type == "face":
        if not results.face_landmarks:
            return image
        lm_list = results.face_landmarks.landmark
    elif landmark_type == "left_hand":
        if not results.left_hand_landmarks:
            return image
        lm_list = results.left_hand_landmarks.landmark
    elif landmark_type == "right_hand":
        if not results.right_hand_landmarks:
            return image
        lm_list = results.right_hand_landmarks.landmark
    elif landmark_type == "pose":
        if not results.pose_landmarks:
            return image
        lm_list = results.pose_landmarks.landmark
    else:
        raise ValueError("Invalid landmark_type")

    h, w, _ = image.shape

    # 2. Compute destination points
    dst_pts = np.array([[lm_list[idx].x * w, lm_list[idx].y * h] for idx in landmark_indices], dtype=np.float32)

    # 3. If 2 landmarks are provided, compute tip dynamically
    if len(dst_pts) == 2:
        p1, p2 = dst_pts
        midpoint = (p1 + p2) / 2
        if tip_offset is not None:
            dx_factor, dy_factor = tip_offset
            # Compute vector between points
            vec = p2 - p1
            distance = np.linalg.norm(vec)
            # Move tip relative to distance
            tip = midpoint + np.array([dx_factor * distance, dy_factor * distance])
        else:
            # Default: point outward along y-axis by 1.5x distance
            scale = np.linalg.norm(p2 - p1) * 1.5
            tip = midpoint + np.array([0, -scale])

        dst_pts = np.vstack([dst_pts, tip])
        dst_pts = np.array(dst_pts, dtype=np.float32).reshape(3, 2)

    # 5. Sticker to BGRA
    if sticker_img.shape[2] == 4:
        sticker_bgra = sticker_img
    else:
        sticker_bgra = cv2.cvtColor(sticker_img, cv2.COLOR_BGR2BGRA)

    # 6. Affine transform & warp
    try:
        M = cv2.getAffineTransform(np.float32(src_pts), dst_pts)
        sticker_warped = cv2.warpAffine(sticker_bgra, M, (w, h),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=[0,0,0])
    except cv2.error:
        return image # when missing coords because no face

    # 7. Overlay using alpha
    alpha_mask = sticker_warped[:, :, 3] / 255.0
    for c in range(3):
        image[:, :, c] = (1 - alpha_mask) * image[:, :, c] + alpha_mask * sticker_warped[:, :, c]

    # 4. Optional: draw landmarks
    if show_landmarks:
        for (x, y) in dst_pts.astype(int):
            cv2.circle(image, (x, y), 3, landmarks_color, -1)

    return image


def add_filters(image, results, filters):
    for f in filters:
        image = f(image, results)
    return image
