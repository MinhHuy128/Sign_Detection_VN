import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.comparator import ModelComparator
from src.evaluation.reporter import EvalReporter
from src.backends import load_best_backend

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    evaluator = ModelEvaluator()

    class DummyBackend:
        backend_name = 'PyTorch (Stubbed)'
    res = evaluator.evaluate(args.weights, DummyBackend(), args.config)
    print(f'Evaluation of {args.weights}: mAP50 = {res.map50}, F1 = {res.f1}')
if __name__ == '__main__':
    main()