import os
import time
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from ..io.video_reader import ThreadedVideoReader

class BenchmarkSuite:

    def __init__(self, video_path: str, n_frames: int=300):
        self.video_path = video_path
        self.n_frames = n_frames
        self.backends = {}
        self.results = []
        self.output_dir = 'runs/benchmark'
        os.makedirs(self.output_dir, exist_ok=True)

    def add_backend(self, name: str, backend, model_path: str):
        self.backends[name] = {'backend': backend, 'model_path': model_path}

    def _get_vram(self) -> float:
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], encoding='utf-8')
            return float(out.strip().split('\n')[0])
        except Exception:
            return None

    def run(self) -> pd.DataFrame:
        for name, info in self.backends.items():
            print(f'Benchmarking {name}...')
            backend = info['backend']
            try:
                backend.warmup(10)
                reader = ThreadedVideoReader(self.video_path).start()
                latencies = []
                frames_processed = 0
                start_time = time.time()
                while reader.more() and frames_processed < self.n_frames:
                    frame = reader.read()
                    if frame is None:
                        continue
                    t0 = time.time()
                    backend.infer(frame)
                    t1 = time.time()
                    latencies.append((t1 - t0) * 1000)
                    frames_processed += 1
                reader.stop()
                total_time = time.time() - start_time
                fps = frames_processed / total_time if total_time > 0 else 0
                vram = self._get_vram()
                self.results.append({'Backend': name, 'Avg Latency (ms)': np.mean(latencies) if latencies else 0, 'Std Latency (ms)': np.std(latencies) if latencies else 0, 'Min Latency (ms)': np.min(latencies) if latencies else 0, 'Max Latency (ms)': np.max(latencies) if latencies else 0, 'Avg FPS': fps, 'VRAM (MB)': vram if vram else 0})
            except Exception as e:
                print(f'[WARN] Benchmark failed for {name}: {e}')
        df = pd.DataFrame(self.results)
        self._generate_report(df)
        return df

    def _generate_report(self, df: pd.DataFrame):
        if len(df) == 0:
            return
        md_path = os.path.join(self.output_dir, 'report.md')
        with open(md_path, 'w') as f:
            f.write('# Benchmark Report\n\n')
            f.write(df.to_markdown(index=False))
        plt.figure()
        plt.bar(df['Backend'], df['Avg FPS'])
        plt.title('Avg FPS by Backend')
        plt.savefig(os.path.join(self.output_dir, 'fps_bar.png'))
        plt.close()
        plt.figure()
        plt.bar(df['Backend'], df['Avg Latency (ms)'], yerr=df['Std Latency (ms)'], capsize=5)
        plt.title('Latency Boxplot (Mocked with Error Bars)')
        plt.savefig(os.path.join(self.output_dir, 'latency_boxplot.png'))
        plt.close()