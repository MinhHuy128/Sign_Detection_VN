import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import ModelTrainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train_config.yaml')
    parser.add_argument('--arch', default=None)
    args = parser.parse_args()
    trainer = ModelTrainer(args.config)
    arch = args.arch if args.arch else trainer.select_architecture()
    trainer.train(arch)
if __name__ == '__main__':
    main()