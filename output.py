import cv2
import numpy as np
import pyvirtualcam

class VirtualCamOutput:
    """
    Wrapper for pyvirtualcam to output frames to a virtual webcam.
    Can be used as a context manager.
    """
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """
        Initializes the virtual camera settings.

        Args:
            width (int): Frame width.
            height (int): Frame height.
            fps (int): Frames per second.
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None

    def __enter__(self):
        # pyvirtualcam expects BGR if we specify it, or RGB by default. 
        # OpenCV works in BGR. We can use PixelFormat.BGR to avoid manual conversion.
        self.cam = pyvirtualcam.Camera(
            width=self.width, 
            height=self.height, 
            fps=self.fps, 
            fmt=pyvirtualcam.PixelFormat.BGR
        )
        print(f"Using virtual camera: {self.cam.device}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cam is not None:
            self.cam.close()
            self.cam = None

    def send(self, frame: np.ndarray):
        """
        Sends a frame to the virtual camera.

        Args:
            frame (np.ndarray): The BGR frame to output.
        """
        if self.cam is not None:
            self.cam.send(frame)
            self.cam.sleep_until_next_frame()
