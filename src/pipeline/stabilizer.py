from collections import deque
from typing import Dict, Tuple

class SimpleKalman1D:

    def __init__(self, initial_state: float=0.0, process_noise: float=0.1, measurement_noise: float=0.1):
        self.state = initial_state
        self.P = 1.0
        self.Q = process_noise
        self.R = measurement_noise

    def update(self, measurement: float) -> float:
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.state = self.state + K * (measurement - self.state)
        self.P = (1 - K) * self.P
        return self.state

class TrackState:

    def __init__(self):
        self.label_votes: Dict[str, float] = {}
        self.confidence_history = deque(maxlen=10)
        self.kalman = SimpleKalman1D()

    def add_vote(self, label: str, conf: float):
        for k in self.label_votes.keys():
            self.label_votes[k] *= 0.85
        self.label_votes[label] = self.label_votes.get(label, 0.0) + conf

    def get_best_label(self) -> str:
        if not self.label_votes:
            return ''
        return max(self.label_votes.items(), key=lambda x: x[1])[0]

class KalmanVotingStabilizer:
    """"""

    def __init__(self):
        self.tracks: Dict[int, TrackState] = {}

    def vote(self, tracker_id: int, label: str, conf: float) -> Tuple[str, float]:
        if tracker_id not in self.tracks:
            self.tracks[tracker_id] = TrackState()
        state = self.tracks[tracker_id]
        state.add_vote(label, conf)
        smoothed_conf = state.kalman.update(conf)
        state.confidence_history.append(smoothed_conf)
        return (state.get_best_label(), smoothed_conf)