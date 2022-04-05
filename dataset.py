import argparse
import os.path

import torch
from torch.utils import data
from utils.data_util import *
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

class ChitChatDataset(data.Dataset):
    def __init__(self, tokenizer: BertTokenizer, args, type):
        self.tokenizer = tokenizer
        self.data = load_json(os.path.join(args.data_path, "{}.json".format(type)))
        self.cls_token_id, self.sep_token_id, self.pad_token_id = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]'])

        self.history = []
        self.response = []
        for i in tqdm(self.data):
            if len(i['history']) > 512 or len(i['response']) > 512:
                continue
            self.history.append(i['history'])
            self.response.append(i['response'])

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        return {
            "history": self.history[idx],
            "response": self.response[idx]
        }

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def collate_fn(batch):
            history = [torch.tensor(i['history']) for i in batch]
            response = [torch.tensor(i['response']) for i in batch]

            history = pad_sequence(history, batch_first=True, padding_value=self.pad_token_id)
            response = pad_sequence(response, batch_first=True, padding_value=self.pad_token_id)

            history_mask = history != self.pad_token_id
            response_mask = response != self.pad_token_id

            return {
                "input_ids": history,
                "attention_mask": history_mask,
                "decoder_input_ids": response,
                "decoder_attention_mask": response_mask
            }

        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="../pretrained_models/cpt-base")
    parser.add_argument("--data_path", default="../dataset/cooked/LCCC")
    parser.add_argument("--task_type", default="test")
    return parser.parse_args()


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained("../pretrained_models/cpt-base")
    args = init_args()
    train_dataset = ChitChatDataset(tokenizer, args, 'test')
    d = train_dataset.get_dataloader(batch_size=4, shuffle=False, num_workers=8)
    cnt = 0

    for dd in tqdm(d):
        cnt += 1
        pass