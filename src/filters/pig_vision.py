import cv2
import numpy as np

def pig_vision_filter(image, intensity=0.2, blur_ksize=3):
    """
    Mimic pig vision: enhance red/pink hues, reduce other colors.
    intensity: 0..1, how strong the effect is
    """
    img = image.astype(np.float32) / 255.0

    # Split channels
    B, G, R = cv2.split(img)

    # Reduce red perception and slightly emphasize green/blue
    R *= intensity
    B *= 1.0
    G *= 1.0

    # Clip
    B = np.clip(B, 0, 1)
    G = np.clip(G, 0, 1)
    R = np.clip(R, 0, 1)

    pig_img = cv2.merge([B, G, R])

    # Slight blur to simulate near-sightedness
    if blur_ksize > 1:
        pig_img = cv2.GaussianBlur(pig_img, (blur_ksize, blur_ksize), 0)

    pig_img = np.clip(pig_img * 255, 0, 255).astype(np.uint8)
    return pig_img
