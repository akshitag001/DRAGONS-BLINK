import os
import shutil
from ultralytics import YOLO

def train_and_export():
    """
    Fine-tunes a YOLOv8 nano model on the custom card dataset
    and exports it to ONNX.
    """
    data_yaml = os.path.abspath(os.path.join("data", "cards", "data.yaml"))
    
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset config not found at {data_yaml}")
        print("Please extract your dataset into the data/cards/ directory first.")
        return
        
    print("Initializing YOLOv8n...")
    model = YOLO('yolov8n.pt')
    
    print("Starting training...")
    # Train for 75 epochs (good for 50-100 images dataset to avoid overfitting)
    results = model.train(
        data=data_yaml,
        epochs=75,
        imgsz=640,
        project="runs",
        name="card_detector",
        exist_ok=True
    )
    
    print("\nTraining complete. Exporting best model to ONNX...")
    # Export the model
    export_path = model.export(format='onnx', imgsz=640)
    
    # Copy exported model to our models directory
    target_path = os.path.join("models", "card_detector.onnx")
    os.makedirs("models", exist_ok=True)
    
    if export_path and os.path.exists(export_path):
        shutil.copy(export_path, target_path)
        print(f"\nModel exported successfully to: {target_path}")
        print(f"To see metrics, check the runs/card_detector/ directory.")
    else:
        print("\nFailed to export model or find exported ONNX file.")

if __name__ == "__main__":
    # Ensure script is run from the project root
    if not os.path.exists("models"):
        print("Please run this script from the project root (e.g., `python scripts/train_card_detector.py`)")
    else:
        train_and_export()
