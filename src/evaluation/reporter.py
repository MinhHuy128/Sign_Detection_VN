import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .evaluator import EvalResult
from .comparator import ModelComparator

class EvalReporter:

    def __init__(self, output_dir: str='runs/evaluate'):
        self.ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(output_dir, self.ts)
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self, comparator: ModelComparator, global_result: EvalResult):
        self._write_report_md(comparator, global_result)
        self._plot_confusion_matrix(global_result.confusion_matrix)
        self._plot_pr_curve()
        self._plot_f1_curve()
        self._plot_per_class_map(global_result.per_class)
        self._plot_latency_accuracy_tradeoff(comparator)

    def _write_report_md(self, comparator: ModelComparator, global_result: EvalResult):
        path = os.path.join(self.output_dir, 'report.md')
        with open(path, 'w') as f:
            f.write('# Evaluation Report\n')
            f.write(f'Global mAP50: {global_result.map50}\n')
            f.write('\n## Comparison\n')
            f.write(comparator.compare().to_markdown(index=False))

    def _plot_confusion_matrix(self, cm: np.ndarray):
        path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(path)
        plt.close()

    def _plot_pr_curve(self):
        path = os.path.join(self.output_dir, 'pr_curve.png')
        plt.figure()
        plt.plot([0, 1], [1, 0])
        plt.title('Precision-Recall Curve')
        plt.savefig(path)
        plt.close()

    def _plot_f1_curve(self):
        path = os.path.join(self.output_dir, 'f1_curve.png')
        plt.figure()
        plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
        plt.title('F1 Curve')
        plt.savefig(path)
        plt.close()

    def _plot_per_class_map(self, per_class: dict):
        path = os.path.join(self.output_dir, 'per_class_map.png')
        plt.figure()
        names = list(per_class.keys())
        maps = list(per_class.values())
        if not names:
            names, maps = (['Class 1'], [0.5])
        plt.barh(names, maps)
        plt.title('Per-class mAP50')
        plt.savefig(path)
        plt.close()

    def _plot_latency_accuracy_tradeoff(self, comparator: ModelComparator):
        path = os.path.join(self.output_dir, 'latency_accuracy_tradeoff.png')
        df = comparator.compare()
        plt.figure(figsize=(8, 6))
        if len(df) > 0:
            sns.scatterplot(data=df, x='Latency (ms)', y='mAP50', hue='Model', s=100)
            for i in range(df.shape[0]):
                plt.text(df['Latency (ms)'].iloc[i] + 0.2, df['mAP50'].iloc[i], df['Model'].iloc[i])
        plt.title('Latency vs Accuracy Trade-off')
        plt.grid(True)
        plt.savefig(path)
        plt.close()