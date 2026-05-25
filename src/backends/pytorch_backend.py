from .base import InferenceBackend
import numpy as np
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    YOLO, RTDETR = (None, None)
try:
    import supervision as sv
except ImportError:
    sv = None

class PyTorchBackend(InferenceBackend):

    def __init__(self):
        self.model = None
        self.device = 'cpu'

    def load(self, model_path: str, device: str) -> None:
        if YOLO is None:
            raise ImportError('ultralytics missing')
        self.device = device
        self.model = RTDETR(model_path) if 'rtdetr' in model_path.lower() else YOLO(model_path)

    def infer(self, frame: np.ndarray):
        if self.model is None or sv is None:
            return sv.Detections.empty() if sv else None
        results = self.model(frame, device=self.device, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

    def warmup(self, n_runs: int=10) -> None:
        if self.model is None:
            return
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n_runs):
            self.infer(dummy)

    @property
    def backend_name(self) -> str:
        return 'PyTorch'