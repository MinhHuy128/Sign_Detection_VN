import pandas as pd
from typing import Dict
from .evaluator import EvalResult

class ModelComparator:

    def __init__(self):
        self.results: Dict[str, EvalResult] = {}

    def add_result(self, name: str, result: EvalResult):
        self.results[name] = result

    def compare(self) -> pd.DataFrame:
        data = []
        for name, res in self.results.items():
            data.append({'Model': name, 'mAP50': res.map50, 'mAP50-95': res.map50_95, 'Latency (ms)': res.inference_latency_ms, 'F1': res.f1})
        df = pd.DataFrame(data)
        if len(df) > 0:
            df = df.sort_values(by='mAP50', ascending=False)
        return df

    def best_model(self, metric: str='map50') -> str:
        if not self.results:
            return ''
        best_name = max(self.results.items(), key=lambda x: getattr(x[1], metric, 0))[0]
        return best_name