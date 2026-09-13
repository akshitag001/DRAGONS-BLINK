import cv2
import time
import os

from config import (
    FRAME_WIDTH, 
    FRAME_HEIGHT, 
    CAMERA_INDEX
)
from state import state

from capture import capture_frames
from detectors.face_detector import FaceDetector
from detectors.card_detector import CardDetector
from redact import blur_regions
from output import VirtualCamOutput

def run_pipeline():
    """
    Main entry point for the PrivacyGuard pipeline.
    Capture -> Detect -> Redact -> Display -> Loop
    """
    print("Initializing PrivacyGuard Pipeline...")
    
    # Initialize detectors (conditionally based on whether model files exist)
    try:
        face_detector = FaceDetector()
    except Exception as e:
        print(f"Face detector failed to load (model missing?): {e}")
        face_detector = None
        
    try:
        card_detector = CardDetector()
    except Exception as e:
        print(f"Card detector failed to load (model missing?): {e}")
        card_detector = None
    
    print("Starting capture...")
    
    # Set up Virtual Camera context if enabled
    if state.use_virtual_cam:
        try:
            import pyvirtualcam
            vcam_context = VirtualCamOutput(width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30)
        except Exception as e:
            print(f"Virtual camera failed to initialize: {e}. Falling back to preview only.")
            vcam_context = None
    else:
        vcam_context = None
        
    try:
        # 1. Open virtual camera context (or a dummy if unavailable)
        with (vcam_context if vcam_context else open(os.devnull, 'w')) as vcam:
            prev_time = time.time()
            
            # 2. Capture frame-by-frame
            for frame in capture_frames(camera_index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT):
                if not state.is_running:
                    break
                    
                # Measure FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
                prev_time = curr_time
                
                # Update state FPS for dashboard
                state.fps = fps
                
                all_boxes = []
                
                # 3. Detect configured categories dynamically reading from state
                if state.blur_faces and face_detector:
                    face_boxes = face_detector.detect(frame)
                    all_boxes.extend(face_boxes)
                    state.total_faces += len(face_boxes)
                    
                if state.blur_cards and card_detector:
                    card_boxes = card_detector.detect(frame)
                    all_boxes.extend(card_boxes)
                    state.total_cards += len(card_boxes)
                
                # 4. Redact the detected regions
                redacted_frame = blur_regions(frame, all_boxes)
                
                # 5. Overlay Polish / UI for Demo
                cv2.putText(redacted_frame, f"FPS: {fps:.1f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Active categories overlay
                active_str = "Active: "
                if state.blur_faces: active_str += "[Faces] "
                if state.blur_cards: active_str += "[Cards] "
                cv2.putText(redacted_frame, active_str, (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                # Detection counters
                cv2.putText(redacted_frame, f"Faces caught: {state.total_faces}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                cv2.putText(redacted_frame, f"Cards caught: {state.total_cards}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                # 6. Output to Virtual Cam and Preview Window
                if vcam_context and hasattr(vcam_context, 'cam') and vcam_context.cam is not None:
                    vcam_context.send(redacted_frame)
                
                cv2.imshow('PrivacyGuard Preview', redacted_frame)
                
                # Break loop on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    state.is_running = False
                    break
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError during capture loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        state.is_running = False
        print("PrivacyGuard pipeline stopped.")

if __name__ == "__main__":
    run_pipeline()
