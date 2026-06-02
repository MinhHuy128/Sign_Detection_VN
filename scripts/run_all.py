""""""
import argparse
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from src.utils.logger import get_logger
logger = get_logger(__name__)

def run_stage(name, fn, *args, **kwargs):
    print(f'\n[STAGE {name}]')
    start = time.time()
    try:
        res = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[✓] {name.split(' ', 1)[1]} — {elapsed:.2f}s")
        return res
    except Exception as e:
        elapsed = time.time() - start
        print(f"[✗] {name.split(' ', 1)[1]} — {str(e)}")
        import json
        os.makedirs('runs', exist_ok=True)
        log_path = 'runs/pipeline_log.json'
        log_data = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            except Exception:
                pass
        log_data.append({'stage': name, 'error': str(e), 'time': elapsed})
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)
        return None

def verify_dependencies():
    import ultralytics
    import supervision
    import mlflow
    import torch
    print('Dependencies verified.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Test video for smoke test')
    parser.add_argument('--skip-train', action='store_true', help='Skip full training phase')
    args = parser.parse_args()
    stages_passed = 0
    res = run_stage('1/8 Verify Dependencies', verify_dependencies)
    if res is not None:
        stages_passed += 1
    from scripts.setup_data import setup_data
    res = run_stage('2/8 Setup Dataset', setup_data)
    if res is not None:
        stages_passed += 1
    from src.training.trainer import ModelTrainer
    trainer = ModelTrainer('configs/train_config.yaml')
    best_arch = run_stage('3/8 Architecture Selection', trainer.select_architecture)
    if best_arch is not None:
        stages_passed += 1
    else:
        best_arch = 'yolo11s.pt'
    if not args.skip_train:
        best_pt = run_stage('4/8 Full Training', trainer.train, best_arch)
        if best_pt is not None:
            stages_passed += 1
    else:
        best_pt = 'models/signs/best.pt'
        print('\n[STAGE 4/8 Full Training]')
        print('[✓] Skipped by user')
        stages_passed += 1
    from src.evaluation.evaluator import ModelEvaluator
    from src.evaluation.comparator import ModelComparator
    from src.evaluation.reporter import EvalReporter

    def run_eval():
        evaluator = ModelEvaluator()
        from src.backends.pytorch_backend import PyTorchBackend
        backend = PyTorchBackend()
        res1 = evaluator.evaluate(best_pt, backend, 'data.yaml')
        comp = ModelComparator()
        comp.add_result('PyTorch', res1)
        rep = EvalReporter()
        rep.generate_report(comp, res1)
        return res1.map50
    map50 = run_stage('5/8 Full Evaluation', run_eval)
    if map50 is not None:
        stages_passed += 1
    else:
        map50 = 0.0
    paths = run_stage('6/8 Export PT -> ONNX -> TRT', trainer.export, best_pt)
    if paths is not None:
        stages_passed += 1
    from src.benchmark.suite import BenchmarkSuite

    def run_bench():
        suite = BenchmarkSuite(args.input, n_frames=10)
        from src.backends.pytorch_backend import PyTorchBackend
        suite.add_backend('PyTorch', PyTorchBackend(), best_pt)
        df = suite.run()
        best_fps = df['Avg FPS'].max() if len(df) > 0 else 0
        return best_fps
    best_fps = run_stage('7/8 Inference Benchmark', run_bench)
    if best_fps is not None:
        stages_passed += 1
    else:
        best_fps = 0.0

    def run_smoke_test():
        with open('configs/default.yaml', 'r') as f:
            config = yaml.safe_load(f)
        from src.pipeline.detector import DetectionPipeline
        from src.io.video_reader import ThreadedVideoReader
        from src.io.video_writer import ThreadedVideoWriter
        from src.utils.visualizer import HUDRenderer
        pipeline = DetectionPipeline(config)
        reader = ThreadedVideoReader(args.input).start()
        frame = reader.read()
        if frame is None:
            return False
        h, w = frame.shape[:2]
        os.makedirs('runs/inference', exist_ok=True)
        writer = ThreadedVideoWriter('runs/inference/output.mp4', w, h, fps=30).start()
        renderer = HUDRenderer()
        frames = 0
        total_detections = 0
        t0 = time.time()
        det, stats = pipeline.run(frame)
        if det is not None and hasattr(det, '__len__'):
            total_detections += len(det)
        out_frame = frame.copy()
        if det is not None:
            out_frame = renderer.draw_boxes(out_frame, det)
        out_frame = renderer.draw_hud(out_frame, stats)
        writer.write(out_frame)
        frames += 1
        while reader.more() and frames < 50:
            frame = reader.read()
            if frame is None:
                continue
            det, stats = pipeline.run(frame)
            if det is not None and hasattr(det, '__len__'):
                total_detections += len(det)
            out_frame = frame.copy()
            if det is not None:
                out_frame = renderer.draw_boxes(out_frame, det)
            out_frame = renderer.draw_hud(out_frame, stats)
            writer.write(out_frame)
            frames += 1
        reader.stop()
        writer.stop()
        fps = frames / (time.time() - t0)
        print('\n[SMOKE TEST RESULT]')
        print(f'Input  : {args.input}')
        print(f'Output : runs/inference/output.mp4')
        print(f'Frames : {frames}')
        print(f'Avg FPS: {fps:.1f}')
        print(f'Total detections: {total_detections}')
        print('Status : PASS ✓')
        return True
    res = run_stage('8/8 Smoke Test', run_smoke_test)
    if res is not None:
        stages_passed += 1
    print('\n╔══════════════════════════════════════════╗')
    print('║   ROADSENSE AI v2 — PIPELINE COMPLETE    ║')
    print('╠══════════════════════════════════════════╣')
    print(f'║  Stages passed : {stages_passed}/8                     ║')
    print(f'║  Best arch     : {best_arch:<23} ║')
    print(f'║  mAP@0.5       : {map50:<23.3f} ║')
    print(f'║  Best FPS      : {best_fps:<23.1f} ║')
    print('╠══════════════════════════════════════════╣')
    print('║  OUTPUT FILES                            ║')
    print('║  models/signs/best.pt                    ║')
    print('║  models/signs/best.onnx                  ║')
    print('║  models/signs/best.engine (if TRT)       ║')
    print('║  runs/evaluate/<ts>/report.md            ║')
    print('║  runs/benchmark/report.md                ║')
    print('║  runs/inference/output.mp4               ║')
    print('╠══════════════════════════════════════════╣')
    print('║  RUN INFERENCE ONLY:                     ║')
    print('║  python main.py --input <video.mp4>      ║')
    print('╚══════════════════════════════════════════╝')
if __name__ == '__main__':
    main()