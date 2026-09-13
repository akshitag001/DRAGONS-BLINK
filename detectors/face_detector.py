import cv2
import numpy as np
import onnxruntime as ort
import torch
from qai_hub_models.models.face_det_lite.utils import detect

from config import EXECUTION_PROVIDERS

class FaceDetector:
    """
    QAI Hub Face Detector using ONNX Runtime.
    """
    def __init__(self, model_path: str = "models/face_det_lite.onnx"):
        """
        Initializes the face detector with the ONNX model.
        """
        self.session = ort.InferenceSession(model_path, providers=EXECUTION_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        
        # Model expects 480x640
        self.model_h = 480
        self.model_w = 640

    def detect(self, frame: np.ndarray) -> list[list[int]]:
        """
        Detects faces in the given frame.

        Args:
            frame (np.ndarray): The BGR frame from OpenCV.

        Returns:
            list[list[int]]: List of bounding boxes as [x, y, w, h].
        """
        # Save original dimensions for scaling bounding boxes back
        orig_h, orig_w = frame.shape[:2]
        
        # Pre-process: BGR -> Grayscale -> Resize -> Normalize -> NCHW
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.model_w, self.model_h))
        
        # Normalize: Image is scaled to 0-1, then (img - 0.442) / 0.280
        img_float = resized.astype(np.float32) / 255.0
        normalized = (img_float - 0.442) / 0.280
        
        # Add N and C dimensions: (1, 1, 480, 640)
        input_tensor = np.expand_dims(np.expand_dims(normalized, axis=0), axis=0)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # outputs are [heatmap, bbox, landmark]
        hm = torch.from_numpy(outputs[0])
        box = torch.from_numpy(outputs[1])
        landmark = torch.from_numpy(outputs[2])
        
        # Post-process using qai_hub_models utility
        dets = detect(hm, box, landmark, threshold=0.55, nms_iou=-1, stride=8)
        
        scale_x = orig_w / self.model_w
        scale_y = orig_h / self.model_h
        
        face_boxes = []
        for det in dets:
            xmin, ymin, w, h = det.xywh
            # Scale coordinates back to original frame
            x_orig = int(xmin * scale_x)
            y_orig = int(ymin * scale_y)
            w_orig = int(w * scale_x)
            h_orig = int(h * scale_y)
            
            # Optionally enlarge box slightly to ensure whole face is covered
            w_new = int(w_orig * 1.2)
            h_new = int(h_orig * 1.2)
            x_new = int(x_orig - w_orig * 0.1)
            y_new = int(y_orig - h_orig * 0.1)
            
            face_boxes.append([x_new, y_new, w_new, h_new])
            
        return face_boxes
