import os.path
from data_util import *
from tqdm import tqdm
from transformers import BertTokenizer
import argparse

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='../data/DuSinc', type=str)
    parser.add_argument('--file', default='test_dial_1.txt', type=str)
    parser.add_argument('--save', default='../data', type=str)
    parser.add_argument('--save_file', default='test.json', type=str)
    parser.add_argument('--model', default='fnlp/bart-base-chinese', type=str)

    return parser.parse_args()

def load_test(path, tokenizer):
    input_data = load_txt(path)
    output_data = []
    for data in tqdm(input_data):
        data = json.loads(data)
        history, past_knowledge = [], []
        for item in data['conversation']:
            if item['utterance'] != "":
                history.append(item['utterance'])
            else:
                output_data.append({
                    'history': '[SEP]'.join(history),
                    'response': item['utterance'],
                    'knowledge': item['use_knowledge'] if item['use_kg_label'] == 'true' else None,
                    'other_knowledge': [i['search_knowledge'] for i in item['other_search']] if item['use_kg_label'] == 'true' else None
                })
    for data in tqdm(output_data):
        data['history'] = tokenizer(data['history'])['input_ids']
        data['response'] = tokenizer(data['response'])['input_ids']
        data['knowledge'] = tokenizer(data['knowledge'] if data['knowledge'] is not None else '')['input_ids']
        if data['other_knowledge'] is not None:
            data['other_knowledge'] = [tokenizer(i)['input_ids'] for i in data['other_knowledge']]
        else:
            data['other_knowledge'] = [tokenizer('')['input_ids']]
    return output_data


def load(path, tokenizer):
    input_data = load_txt(path)
    output_data = []
    for data in tqdm(input_data):
        data = json.loads(data)
        history, past_knowledge = [], []
        for item in data['conversation']:
            if item['role'] == 'user':
                history.append(item['utterance'])
            else:
                output_data.append({
                    'history': '[SEP]'.join(history),
                    'response': item['utterance'],
                    'knowledge': item['use_knowledge'] if item['use_kg_label'] == 'true' else None,
                    'other_knowledge': [i['search_knowledge'] for i in item['other_search']] if item['use_kg_label'] == 'true' else None
                })
                history.append(item['utterance'])
    for data in tqdm(output_data):
        data['history'] = tokenizer(data['history'])['input_ids']
        data['response'] = tokenizer(data['response'])['input_ids']
        data['knowledge'] = tokenizer(data['knowledge'] if data['knowledge'] is not None else '')['input_ids']
        if data['other_knowledge'] is not None:
            data['other_knowledge'] = [tokenizer(i)['input_ids'] for i in data['other_knowledge']]
        else:
            data['other_knowledge'] = [tokenizer('')['input_ids']]
    return output_data

if __name__ == '__main__':
    args = init_args()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    input_path = os.path.join(args.path, args.file)
    output_path = os.path.join(args.save, args.save_file)

    output_data = load_test(input_path, tokenizer)
    save_json(output_data, output_path)
