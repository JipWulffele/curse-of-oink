import cv2
import numpy as np

# Load images PNG (RGBA)
bacon_head_img = cv2.imread("assets/stickers/bacon_head.png", cv2.IMREAD_UNCHANGED)
chop_left_img = cv2.imread("assets/stickers/pork_chop_left.png", cv2.IMREAD_UNCHANGED)
chop_right_img = cv2.imread("assets/stickers/pork_chop_right.png", cv2.IMREAD_UNCHANGED)

# idx pose_landmarks
head_landmarks = [8, 7]
left_hand_landmarks = [20, 18]
right_hand_landmarks = [19, 17]


def bacon_head_filter(image, results):
    if not results.pose_landmarks:
        return image

    h, w, _ = image.shape
    lm_list = results.pose_landmarks.landmark

    # Compute head center in pixels
    x1, y1 = int(lm_list[head_landmarks[0]].x * w), int(lm_list[head_landmarks[0]].y * h)
    x2, y2 = int(lm_list[head_landmarks[1]].x * w), int(lm_list[head_landmarks[1]].y * h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Scale bacon width to head width
    head_dist = max(1, int(np.hypot(x2 - x1, y2 - y1)))  # avoid divide by zero
    scale = head_dist / bacon_head_img.shape[1] * 2 # width scaling

    tail_h = int(bacon_head_img.shape[0] * scale)
    tail_w = int(bacon_head_img.shape[1] * scale)
    bacon_resized = cv2.resize(bacon_head_img, (tail_w, tail_h), interpolation=cv2.INTER_AREA)

    # Compute top-left corner to center on face
    x_offset = cx - tail_w // 2
    y_offset = cy - tail_h // 2

    # Overlay PNG with alpha
    for c in range(3):  # RGB channels
        alpha = bacon_resized[:, :, 3] / 255.0
        y1_clip = max(0, y_offset)
        y2_clip = min(h, y_offset + tail_h)
        x1_clip = max(0, x_offset)
        x2_clip = min(w, x_offset + tail_w)

        # Check if there is anything to overlay
        if y2_clip <= y1_clip or x2_clip <= x1_clip:
            return image  # nothing to draw

        bacon_crop = bacon_resized[
            y1_clip - y_offset : y2_clip - y_offset,
            x1_clip - x_offset : x2_clip - x_offset
        ]

        alpha_crop = bacon_crop[:, :, 3] / 255.0
        image[y1_clip:y2_clip, x1_clip:x2_clip, c] = (
            (1 - alpha_crop) * image[y1_clip:y2_clip, x1_clip:x2_clip, c] +
            alpha_crop * bacon_crop[:, :, c]
        )

    return image

def pork_chop_hand_filter(image, results, side='left'):
    if not results.pose_landmarks:
        return image

    h, w, _ = image.shape
    lm_list = results.pose_landmarks.landmark

    if side == 'left':
        pork_chop_img = chop_left_img
        hand_landmarks = left_hand_landmarks
    else: 
        pork_chop_img = chop_right_img
        hand_landmarks = right_hand_landmarks

    vis_threshold = 0.3
    if any(lm_list[idx].visibility < vis_threshold for idx in hand_landmarks):
        return image

    # Compute center in pixels
    x1, y1 = int(lm_list[hand_landmarks[0]].x * w), int(lm_list[hand_landmarks[0]].y * h)
    x2, y2 = int(lm_list[hand_landmarks[1]].x * w), int(lm_list[hand_landmarks[1]].y * h)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Scale width
    hand_dist = max(1, int(np.hypot(x2 - x1, y2 - y1)))  # avoid divide by zero
    scale = hand_dist / pork_chop_img.shape[1] * 6 # width scaling

    tail_h = int(pork_chop_img.shape[0] * scale)
    tail_w = int(pork_chop_img.shape[1] * scale)
    pork_chop_resized = cv2.resize(pork_chop_img, (tail_w, tail_h), interpolation=cv2.INTER_AREA)

    # Compute top-left corner to center on face
    x_offset = cx - tail_w // 2
    y_offset = cy - tail_h // 2

    # Overlay PNG with alpha
    for c in range(3):  # RGB channels
        alpha = pork_chop_resized[:, :, 3] / 255.0
        y1_clip = max(0, y_offset)
        y2_clip = min(h, y_offset + tail_h)
        x1_clip = max(0, x_offset)
        x2_clip = min(w, x_offset + tail_w)

        # Check if there is anything to overlay
        if y2_clip <= y1_clip or x2_clip <= x1_clip:
            return image  # nothing to draw

        pork_chop_crop = pork_chop_resized[
            y1_clip - y_offset : y2_clip - y_offset,
            x1_clip - x_offset : x2_clip - x_offset
        ]

        alpha_crop = pork_chop_crop[:, :, 3] / 255.0
        image[y1_clip:y2_clip, x1_clip:x2_clip, c] = (
            (1 - alpha_crop) * image[y1_clip:y2_clip, x1_clip:x2_clip, c] +
            alpha_crop * pork_chop_crop[:, :, c]
        )

    return image
