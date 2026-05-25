import numpy as np
from .base import InferenceBackend
try:
    import supervision as sv
except ImportError:
    sv = None
import ultralytics.utils.checks
ultralytics.utils.checks.check_requirements = lambda *args, **kwargs: None
import onnxruntime as ort
original_session = ort.InferenceSession

def patched_session(*args, **kwargs):
    available = ort.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
    if 'DmlExecutionProvider' in available:
        providers.append('DmlExecutionProvider')
    providers.append('CPUExecutionProvider')
    kwargs['providers'] = providers
    return original_session(*args, **kwargs)
ort.InferenceSession = patched_session
from ultralytics import YOLO

class ONNXBackend(InferenceBackend):

    def __init__(self, model_path='models/signs/best.onnx'):
        super().__init__()
        self.model = YOLO(model_path, task='detect')

    def infer(self, frame: np.ndarray):
        if self.model is None or sv is None:
            return sv.Detections.empty() if sv else None
        results = self.model(frame, conf=0.4, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)

    def load(self, model_path: str, device: str) -> None:
        pass

    def warmup(self, n_runs: int=10) -> None:
        pass

    @property
    def backend_name(self) -> str:
        return 'ONNX'