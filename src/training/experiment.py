import mlflow
import os
os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
from typing import Any, Dict

class ExperimentManager:

    def __init__(self, experiment_name: str, tracking_uri: str='mlruns/'):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(self.tracking_uri)
        self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if self.experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id
        self.active_run = None

    def __enter__(self):
        self.active_run = mlflow.start_run(experiment_id=self.experiment_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int=None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifacts(self, local_dir: str):
        if os.path.exists(local_dir):
            mlflow.log_artifacts(local_dir)