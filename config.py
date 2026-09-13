"""
Configuration for PrivacyGuard
"""

# ---------------------------------------------------------
# Execution Settings
# ---------------------------------------------------------
# To leverage the Snapdragon NPU, QNNExecutionProvider is required.
# Note: The QNNExecutionProvider requires the `onnxruntime-qnn` package 
# and a Windows ARM64 device (e.g. Snapdragon X Elite).
# On standard x86 devices, it will gracefully fallback to CPUExecutionProvider.
EXECUTION_PROVIDERS = [
    "QNNExecutionProvider", 
    "CPUExecutionProvider"
]

# ---------------------------------------------------------
# Redaction Settings
# ---------------------------------------------------------
# Toggle which categories of information to redact.
BLUR_FACES = True
BLUR_CARDS = True

# ---------------------------------------------------------
# Camera Settings
# ---------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
USE_VIRTUAL_CAM = True
