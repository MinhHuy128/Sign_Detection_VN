import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.benchmark.suite import BenchmarkSuite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    args = parser.parse_args()
    suite = BenchmarkSuite(args.input)
    print('Benchmark triggered.')
if __name__ == '__main__':
    main()