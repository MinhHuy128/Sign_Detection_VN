from collections import deque
from typing import Dict, Any, List
try:
    import supervision as sv
except ImportError:
    sv = None

class AdaptiveInferenceSlicer:

    def __init__(self, callback, slice_size: int=640, overlap_ratio: float=0.1):
        self.callback = callback
        if sv is None:
            self.slicer = None
        else:
            self.slicer = sv.InferenceSlicer(callback=callback, slice_wh=(slice_size, slice_size))
        self.history = deque(maxlen=10)
        self.current_interval = 5
        self.frame_count = 0

    def __call__(self, frame) -> Any:
        self.frame_count += 1
        if self.slicer is None:
            res = self.callback(frame)
        elif self.frame_count % self.current_interval == 0:
            res = self.slicer(frame)
        else:
            res = self.callback(frame)
        num_objects = len(res) if res is not None and hasattr(res, '__len__') else 0
        self.history.append(num_objects)
        avg = sum(self.history) / max(1, len(self.history))
        if avg > 5:
            self.current_interval = max(2, self.current_interval - 1)
        elif avg < 2:
            self.current_interval = min(10, self.current_interval + 1)
        return res

    @property
    def stats(self) -> Dict[str, Any]:
        return {'current_interval': self.current_interval, 'avg_objects_last_10': sum(self.history) / max(1, len(self.history)) if self.history else 0}