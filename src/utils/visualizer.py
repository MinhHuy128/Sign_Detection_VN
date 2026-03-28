import cv2
import numpy as np
from .class_map import CLASS_MAP, name_en
try:
    import supervision as sv
except ImportError:
    sv = None

class HUDRenderer:

    def __init__(self, language='vi'):
        self.language = language
        if sv is not None:
            self.box_annotator = sv.BoxAnnotator()
            self.label_annotator = sv.LabelAnnotator()
        else:
            self.box_annotator = None
            self.label_annotator = None

    def _get_label_name(self, class_id: int) -> str:
        if self.language == 'vi':
            class_name = CLASS_MAP.get(class_id, {}).get('name_vi', f'Sign {class_id}')
        else:
            class_name = name_en.get(class_id, f'Class {class_id}')
        return class_name

    def draw_boxes(self, frame: np.ndarray, detections) -> np.ndarray:
        if self.box_annotator is None or detections is None:
            return frame
        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        labels = []
        if hasattr(detections, 'class_id'):
            for i, class_id in enumerate(detections.class_id):
                name = self._get_label_name(class_id)
                conf = detections.confidence[i] if hasattr(detections, 'confidence') else 0.0
                if hasattr(detections, 'tracker_id') and detections.tracker_id is not None:
                    trk_id = detections.tracker_id[i]
                    labels.append(f'#{trk_id} {name} {conf:.2f}')
                else:
                    labels.append(f'{name} {conf:.2f}')
        if self.label_annotator is not None and labels:
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame

    def draw_hud(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        y0, dy = (30, 30)
        for i, (k, v) in enumerate(stats.items()):
            y = y0 + i * dy
            text = f'{k}: {v}'
            if isinstance(v, float):
                text = f'{k}: {v:.2f}'
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame