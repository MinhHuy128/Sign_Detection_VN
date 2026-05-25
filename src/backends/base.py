from abc import ABC, abstractmethod
import numpy as np
try:
    import supervision as sv
except ImportError:

    class MockSV:

        class Detections:
            pass
    sv = MockSV()

class InferenceBackend(ABC):

    @abstractmethod
    def load(self, model_path: str, device: str) -> None:
        pass

    @abstractmethod
    def infer(self, frame: np.ndarray) -> sv.Detections:
        pass

    @abstractmethod
    def warmup(self, n_runs: int=10) -> None:
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass