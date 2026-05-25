from .base import InferenceBackend
import numpy as np
try:
    import tensorrt as trt
except ImportError:
    trt = None
try:
    import supervision as sv
except ImportError:
    sv = None

class TensorRTBackend(InferenceBackend):

    def __init__(self):
        self.engine = None

    def load(self, model_path: str, device: str) -> None:
        if trt is None:
            raise ImportError('tensorrt missing. TRT is optional, skipping.')
        self.engine = 'loaded'

    def infer(self, frame: np.ndarray):
        return sv.Detections.empty() if sv else None

    def warmup(self, n_runs: int=10) -> None:
        pass

    @property
    def backend_name(self) -> str:
        return 'TensorRT'