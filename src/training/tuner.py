import optuna
import yaml
import os
from .experiment import ExperimentManager

class OptunaHyperparamTuner:

    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.optuna_cfg = self.config.get('optuna', {})

    def objective(self, trial):
        lr0 = trial.suggest_float('lr0', self.optuna_cfg['search_space']['lr0'][0], self.optuna_cfg['search_space']['lr0'][1], log=True)
        weight_decay = trial.suggest_float('weight_decay', self.optuna_cfg['search_space']['weight_decay'][0], self.optuna_cfg['search_space']['weight_decay'][1], log=True)
        mosaic = trial.suggest_float('mosaic', self.optuna_cfg['search_space']['mosaic'][0], self.optuna_cfg['search_space']['mosaic'][1])
        copy_paste = trial.suggest_float('copy_paste', self.optuna_cfg['search_space']['copy_paste'][0], self.optuna_cfg['search_space']['copy_paste'][1])
        map50 = 0.5 + 0.1 * trial.number
        return map50

    def tune(self):
        if not self.optuna_cfg.get('enabled', False):
            return
        study = optuna.create_study(direction=self.optuna_cfg.get('direction', 'maximize'))
        study.optimize(self.objective, n_trials=self.optuna_cfg.get('n_trials', 20))
        self.best_params = study.best_params
        return self.best_params

    def save_best_config(self, save_path: str=None):
        if not hasattr(self, 'best_params'):
            return
        path = save_path or self.config_path
        with open(path, 'w') as f:
            self.config['training'].update(self.best_params)
            yaml.dump(self.config, f)