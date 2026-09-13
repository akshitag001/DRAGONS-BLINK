import cv2
from typing import Generator

def capture_frames(camera_index: int = 0, width: int = 640, height: int = 480) -> Generator[cv2.typing.MatLike, None, None]:
    """
    Grabs frames from the webcam via OpenCV.

    Args:
        camera_index (int): Device index for the webcam (default is 0).
        width (int): Requested width of the frame.
        height (int): Requested height of the frame.

    Yields:
        np.ndarray: The captured frame as a BGR numpy array.
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()
