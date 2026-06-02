import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--weights', required=True)
    args = parser.parse_args()
    trainer = ModelTrainer(args.config)
    paths = trainer.export(args.weights)
    print(f'Exported to: {paths}')
if __name__ == '__main__':
    main()