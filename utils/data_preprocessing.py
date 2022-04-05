import os.path
from data_util import *
from tqdm import tqdm
from transformers import BertTokenizer
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../data/DuSinc', type=str)
    parser.add_argument('--file', default='train', type=str)
    parser.add_argument('--save', default='../data/DuSinc', type=str)
    parser.add_argument('--save_file', default='train.json', type=str)
    parser.add_argument('--model', default='bert-base-chinese', type=str)

    return parser.parse_args()


def load(path, tokenizer):
    lines = load_json(path)
    res = []
    if path.count('test'):
        for line in tqdm(lines):
            history = ''
            for cnt in range(0, len(line) - 1):
                history = history + line[cnt] + ' [SEP] '
            res.append({
                "history": tokenizer(history.rstrip('[ SEP ]'))['input_ids'],
                'response': tokenizer(line[-1])['input_ids']
            })
    else:
        for line in tqdm(lines):
            history = ''

            for cnt in range(0, len(line) - 1):
                history = history + line[cnt] + ' [SEP] '
                response = line[cnt + 1]
                if sum([len(u) for u in history]) > 12 and len(list(response.split())) > 12:
                    res.append({
                        "history": tokenizer(history.rstrip('[ SEP ]'))['input_ids'],
                        "response": tokenizer(response)['input_ids']
                    })
    return res

if __name__ == '__main__':
    args = init_args()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    input_path = os.path.join(args.path, args.file)
    output_path = os.path.join(args.save, args.save_file)
    res = load(input_path, tokenizer)
    save_json(res, output_path)