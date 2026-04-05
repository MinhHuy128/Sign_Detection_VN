try:
    import supervision as sv
except ImportError:
    sv = None

class SignTracker:

    def __init__(self, track_activation_threshold=0.55):
        if sv is None:
            self.tracker = None
        else:
            self.tracker = sv.ByteTrack(track_activation_threshold=track_activation_threshold)

    def update(self, detections):
        if self.tracker is None or detections is None:
            return detections
        return self.tracker.update_with_detections(detections=detections)