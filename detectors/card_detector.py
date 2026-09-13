import cv2
import numpy as np
import onnxruntime as ort
import os

from config import EXECUTION_PROVIDERS

class CardDetector:
    """
    Card Detector using YOLOv8 ONNX model.
    """
    def __init__(self, model_path: str = "models/card_detector.onnx", conf_thres: float = 0.5, iou_thres: float = 0.4):
        """
        Initializes the YOLOv8 card detector.

        Args:
            model_path (str): Path to the ONNX model.
            conf_thres (float): Confidence threshold for detections.
            iou_thres (float): IoU threshold for NMS.
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        if os.path.exists(model_path):
            self.session = ort.InferenceSession(model_path, providers=EXECUTION_PROVIDERS)
            self.input_name = self.session.get_inputs()[0].name
            # YOLOv8 default is 640x640
            self.model_h = 640
            self.model_w = 640
            self.is_loaded = True
        else:
            print(f"Warning: {model_path} not found. Card detector disabled.")
            self.is_loaded = False

    def detect(self, frame: np.ndarray) -> list[list[int]]:
        """
        Detects cards in the given frame.

        Args:
            frame (np.ndarray): The BGR frame from OpenCV.

        Returns:
            list[list[int]]: List of bounding boxes as [x, y, w, h].
        """
        if not self.is_loaded:
            return []

        orig_h, orig_w = frame.shape[:2]
        
        # Pre-process: BGR -> RGB, Resize, Normalize 0-1, NCHW
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, (self.model_w, self.model_h))
        img_float = resized.astype(np.float32) / 255.0
        # Transpose from HWC to CHW
        img_chw = np.transpose(img_float, (2, 0, 1))
        # Add N dim
        input_tensor = np.expand_dims(img_chw, axis=0)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Output shape: (1, 4 + num_classes, 8400)
        preds = outputs[0][0]  # shape (4+num_classes, 8400)
        
        boxes = []
        scores = []
        
        # We can vectorize this parsing
        preds = preds.T # shape (8400, 4+num_classes)
        
        # Bounding boxes are first 4 columns [xc, yc, w, h]
        # Class scores are from 4th column onwards
        box_preds = preds[:, :4]
        class_preds = preds[:, 4:]
        
        if class_preds.shape[1] > 0:
            max_scores = np.max(class_preds, axis=1)
            # Filter by confidence
            mask = max_scores > self.conf_thres
            
            box_preds = box_preds[mask]
            max_scores = max_scores[mask]
            
            for box, score in zip(box_preds, max_scores):
                xc, yc, w, h = box
                
                # Convert to x, y, w, h (top left)
                x = xc - (w / 2)
                y = yc - (h / 2)
                
                boxes.append([int(x), int(y), int(w), int(h)])
                scores.append(float(score))
                
        if len(boxes) == 0:
            return []
            
        # Apply NMS
        # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h]
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        
        scale_x = orig_w / self.model_w
        scale_y = orig_h / self.model_h
        
        final_boxes = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                
                # Scale back
                x_orig = int(x * scale_x)
                y_orig = int(y * scale_y)
                w_orig = int(w * scale_x)
                h_orig = int(h * scale_y)
                
                # Enlarge slightly for redaction
                w_new = int(w_orig * 1.1)
                h_new = int(h_orig * 1.1)
                x_new = int(x_orig - w_orig * 0.05)
                y_new = int(y_orig - h_orig * 0.05)
                
                final_boxes.append([x_new, y_new, w_new, h_new])
                
        return final_boxes
