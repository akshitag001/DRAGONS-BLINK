import threading

class AppState:
    """
    Thread-safe singleton to hold application state,
    shared between the web dashboard and the video pipeline.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AppState, cls).__new__(cls)
                cls._instance._init_state()
            return cls._instance

    def _init_state(self):
        # Toggles
        self.blur_faces = True
        self.blur_cards = True
        self.use_virtual_cam = True
        
        # Metrics
        self.fps = 0.0
        self.total_faces = 0
        self.total_cards = 0
        
        self.is_running = True

# Global instance
state = AppState()
