import cv2
import numpy as np

def blur_regions(frame: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    """
    Applies a strong Gaussian blur to each bounding box region in the frame.

    Args:
        frame (np.ndarray): The original BGR frame.
        boxes (list[list[int]]): List of bounding boxes, each defined as [x, y, width, height].

    Returns:
        np.ndarray: The frame with specified regions redacted.
    """
    result = frame.copy()
    h_frame, w_frame = result.shape[:2]

    for (x, y, w, h) in boxes:
        # Ensure bounding box is within frame boundaries
        x, y = max(0, x), max(0, y)
        w, h = min(w_frame - x, w), min(h_frame - y, h)

        if w <= 0 or h <= 0:
            continue

        roi = result[y:y+h, x:x+w]
        
        # Apply a strong Gaussian blur
        # Kernel size must be odd and large for heavy blur
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
        
        result[y:y+h, x:x+w] = blurred_roi

    return result
