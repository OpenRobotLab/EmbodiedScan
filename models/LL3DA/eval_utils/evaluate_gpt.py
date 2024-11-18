import json
from argparse import ArgumentParser

from mmscan import GPT_Evaluator


def parse_form(results):
    """Parse the format of output to comform with mmscan format."""
    item_list = []
    for id_with_Q in results:
        item_ = {}
        item_['ID'] = id_with_Q.split('@')[0]
        item_['question'] = results[id_with_Q]['instruction']
        item_['pred'] = [results[id_with_Q]['pred']]
        item_['gt'] = results[id_with_Q]['gt']
        item_list.append(item_)
    return item_list


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--tmp_path', type=str, required=True)
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--eval_size', type=int, default=-1)
    parser.add_argument('--nproc', type=int, default=8)
    args = parser.parse_args()

    ll3da_file_path = args.file

    evaluator = GPT_Evaluator(eval_size =args.eval_size,\
        API_key=args.api_key)

    with open(ll3da_file_path, 'r') as f:
        results = json.load(f)
    print(evaluator.load_and_eval(parse_form(results),num_threads=args.nproc,\
        tmp_path =args.tmp_path))
