import json
from argparse import ArgumentParser
from pathlib import Path

from mmscan import GPTEvaluator

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--tmp_path', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--eval_size', type=int, default=-1)
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    leo_file_path = args.file

    evaluator = GPTEvaluator(eval_size =args.eval_size,\
        API_key=args.api_key)

    with open(leo_file_path, 'r') as f:
        results = json.load(f)
    print(evaluator.load_and_eval(results,num_threads=args.nproc,\
        tmp_path =args.tmp_path))
