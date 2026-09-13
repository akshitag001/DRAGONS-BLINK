import os
import torch
from qai_hub_models.models.face_det_lite.model import FaceDetLite

def export_face_det():
    print("Loading pre-trained FaceDetLite model...")
    # This automatically downloads the weights if they are not cached locally
    model = FaceDetLite.from_pretrained()
    model.eval()
    
    # 1-channel grayscale image, HxW = 480x640, values between 0-1
    dummy_input = torch.rand(1, 1, 480, 640)
    
    output_path = os.path.join("models", "face_det_lite.onnx")
    os.makedirs("models", exist_ok=True)
    
    print(f"Exporting model to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        opset_version=14, 
        input_names=["input"], 
        output_names=["heatmap", "bbox", "landmark"]
    )
    print("Export complete.")

if __name__ == "__main__":
    export_face_det()
