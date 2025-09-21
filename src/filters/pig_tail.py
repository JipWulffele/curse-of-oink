import cv2
import numpy as np

# Load pig tail PNG (RGBA)
pig_tail_img = cv2.imread("assets/stickers/pig_tail.png", cv2.IMREAD_UNCHANGED)

# Use left & right hip landmarks
hip_landmarks = [23, 24]


def is_back_view(lm_list):

    backview = False # so front view

    left_shoulder = lm_list[11]  # x,y normalized
    right_shoulder = lm_list[12]
    left_hip = lm_list[23]
    right_hip = lm_list[24]

    if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
        if left_shoulder.x < right_shoulder.x:
            backview = True
    elif left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
        if left_hip.x < right_hip.x:
            backview = True
    return backview


def pig_tail_filter(image, results):
    if not results.pose_landmarks:
        return image

    h, w, _ = image.shape
    lm_list = results.pose_landmarks.landmark

    if is_back_view(lm_list):

        # Compute hip center in pixels
        x1, y1 = int(lm_list[hip_landmarks[0]].x * w), int(lm_list[hip_landmarks[0]].y * h)
        x2, y2 = int(lm_list[hip_landmarks[1]].x * w), int(lm_list[hip_landmarks[1]].y * h)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Scale tail width to hip distance
        hip_dist = max(1, int(np.hypot(x2 - x1, y2 - y1)))  # avoid divide by zero
        scale = hip_dist / pig_tail_img.shape[1]  # width scaling

        tail_h = int(pig_tail_img.shape[0] * scale)
        tail_w = int(pig_tail_img.shape[1] * scale)
        tail_resized = cv2.resize(pig_tail_img, (tail_w, tail_h), interpolation=cv2.INTER_AREA)

        # Compute top-left corner to center on hip
        x_offset = cx - tail_w // 2
        y_offset = cy - tail_h // 2

        # Overlay PNG with alpha
        for c in range(3):  # RGB channels
            alpha = tail_resized[:, :, 3] / 255.0
            y1_clip = max(0, y_offset)
            y2_clip = min(h, y_offset + tail_h)
            x1_clip = max(0, x_offset)
            x2_clip = min(w, x_offset + tail_w)

            # Check if there is anything to overlay
            if y2_clip <= y1_clip or x2_clip <= x1_clip:
                return image  # nothing to draw

            tail_crop = tail_resized[
                y1_clip - y_offset : y2_clip - y_offset,
                x1_clip - x_offset : x2_clip - x_offset
            ]

            alpha_crop = tail_crop[:, :, 3] / 255.0
            image[y1_clip:y2_clip, x1_clip:x2_clip, c] = (
                (1 - alpha_crop) * image[y1_clip:y2_clip, x1_clip:x2_clip, c] +
                alpha_crop * tail_crop[:, :, c]
            )

    return image
