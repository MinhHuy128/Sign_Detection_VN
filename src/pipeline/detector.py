import numpy as np
from typing import Dict, Any
from .stabilizer import KalmanVotingStabilizer
from .slicer import AdaptiveInferenceSlicer
from .tracker import SignTracker
from ..backends import load_best_backend

class DetectionPipeline:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = load_best_backend(config)
        self.tracker = SignTracker(track_activation_threshold=config.get('model', {}).get('conf_track', 0.55))
        self.stabilizer = KalmanVotingStabilizer()
        if config.get('inference', {}).get('use_slicing', True):
            self.slicer = AdaptiveInferenceSlicer(callback=self.backend.infer, slice_size=config.get('inference', {}).get('slice_size', 640), overlap_ratio=config.get('inference', {}).get('overlap_ratio', 0.1))
        else:
            self.slicer = self.backend.infer

    def run(self, frame: np.ndarray):
        detections = self.slicer(frame)
        if detections is None:
            return (None, {})
        tracked = self.tracker.update(detections)
        if tracked is not None and hasattr(tracked, 'tracker_id') and (tracked.tracker_id is not None):
            for i, trk_id in enumerate(tracked.tracker_id):
                label = str(tracked.class_id[i]) if hasattr(tracked, 'class_id') else 'unknown'
                conf = float(tracked.confidence[i]) if hasattr(tracked, 'confidence') else 1.0
                best_label, smooth_conf = self.stabilizer.vote(trk_id, label, conf)
        stats = {}
        if hasattr(self.slicer, 'stats'):
            stats = self.slicer.stats
        return (tracked, stats)