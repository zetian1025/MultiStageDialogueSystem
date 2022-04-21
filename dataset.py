import argparse
import os.path

import torch
from torch.utils import data
from utils.data_util import *
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

class ChitChatDataset(data.Dataset):
    def __init__(self, tokenizer, args, type):
        self.tokenizer = tokenizer
        self.data = load_json(os.path.join(args.data_path, "{}.json".format(type)))
        self.cls_token_id, self.sep_token_id, self.pad_token_id = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]'])

        self.history = []
        self.response = []
        self.knowledge = []
        for i in tqdm(self.data):
            # if len(i['history']) > 512 or len(i['response']) > 512:
            #     continue
            self.history.append(i['history'][-128:])
            self.response.append(i['response'])
            self.knowledge.append(i['knowledge'][-384:])

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        return {
            "history": self.history[idx],
            "response": self.response[idx],
            "knowledge": self.knowledge[idx]
        }

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def collate_fn(batch):
            history = [torch.tensor(i['history']) for i in batch]
            response = [torch.tensor(i['response']) for i in batch]
            knowledge = [torch.tensor(i['knowledge']) for i in batch]

            encoder_input = [torch.cat((history[i], knowledge[i]), dim=-1)[..., -512:] for i in range(batch_size)]

            # history = pad_sequence(history, batch_first=True, padding_value=self.pad_token_id)
            response = pad_sequence(response, batch_first=True, padding_value=self.pad_token_id)
            # knowledge = pad_sequence(knowledge, batch_first=True, padding_value=self.pad_token_id)

            encoder_input = pad_sequence(encoder_input, batch_first=True, padding_value=self.pad_token_id)

            # history_mask = history != self.pad_token_id
            response_mask = response != self.pad_token_id
            # knowledge_mask = knowledge != self.pad_token_id

            encoder_mask = encoder_input != self.pad_token_id

            return {
                'input_ids': encoder_input,
                'attention_mask': encoder_mask,
                'decoder_input_ids': response,
                'decoder_attention_mask': response_mask
            }

        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True, collate_fn=collate_fn)


class ChitChatDataset_test(data.Dataset):
    def __init__(self, tokenizer, args, type):
        self.tokenizer = tokenizer
        self.data = load_json(os.path.join(args.data_path, "{}.json".format(type)))
        self.cls_token_id, self.sep_token_id, self.pad_token_id = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]', '[PAD]'])

        self.history = []
        self.knowledge = []
        for i in tqdm(self.data):
            self.history.append(i['history'][-256:])
            self.knowledge.append(i['knowledge'][-256:])

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        return {
            "history": self.history[idx],
            "knowledge": self.knowledge[idx]
        }

    def get_dataloader(self, batch_size, shuffle, num_workers):

        def collate_fn(batch):
            history = [torch.tensor(i['history']) for i in batch]
            knowledge = [torch.tensor(i['knowledge']) for i in batch]

            encoder_input = [torch.cat((history[i], knowledge[i]), dim=-1)[..., -512:] for i in range(len(history))]

            # history = pad_sequence(history, batch_first=True, padding_value=self.pad_token_id)
            # knowledge = pad_sequence(knowledge, batch_first=True, padding_value=self.pad_token_id)

            encoder_input = pad_sequence(encoder_input, batch_first=True, padding_value=self.pad_token_id)

            # history_mask = history != self.pad_token_id
            # knowledge_mask = knowledge != self.pad_token_id

            encoder_mask = encoder_input != self.pad_token_id

            return {
                'input_ids': encoder_input,
                'attention_mask': encoder_mask
            }

        return data.DataLoader(dataset=self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)


class FiDDataset(data.Dataset):
    def __init__(self,
                 data,
                 n_context=None,
                 question_prefix='question:',
                 title_prefix='title:',
                 passage_prefix='context:'):
        self.data = data
        self.n_context = n_context
        self.question_prefix = question_prefix
        self.title_prefix = title_prefix
        self.passage_prefix = passage_prefix
        self.sort_data()

    def __len__(self):
        return len(self.data)

    def get_target(self, example):
        if 'target' in example:
            target = example['target']
            return target
        else:
            return None

    def __getitem__(self, index):
        example = self.data[index]
        question = self.question_prefix + " " + example['question']
        target = self.get_target(example)

        if 'ctxs' in example and self.n_context is not None:
            f = self.title_prefix + " {} " + self.passage_prefix + " {}"
            contexts = example['ctxs'][:self.n_context]
            passages = [f.format(c['title'], c['text']) for c in contexts]
            scores = [1.0/len(contexts)] * len(contexts)
            scores = torch.tensor(scores)
            # TODO(egrave): do we want to keep this?
            if len(contexts) == 0:
                contexts = [question]
        else:
            passages, scores = None, None


        return {
            'index' : index,
            'question' : question,
            'target' : target,
            'passages' : passages,
            'scores' : scores
        }

    def sort_data(self):
        if self.n_context is None or not 'score' in self.data[0]['ctxs'][0]:
            return
        for ex in self.data:
            ex['ctxs'].sort(key=lambda x: float(x['score']), reverse=True)

    def get_example(self, index):
        return self.data[index]

def encode_passages(batch_text_passages, tokenizer, max_length):
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(
            text_passages,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True
        )
        passage_ids.append(p['input_ids'][None])
        passage_masks.append(p['attention_mask'][None])

    passage_ids = torch.cat(passage_ids, dim=0)
    passage_masks = torch.cat(passage_masks, dim=0)
    return passage_ids, passage_masks.bool()


class FiDCollator(object):
    def __init__(self, text_maxlength, tokenizer, answer_maxlength=20):
        self.tokenizer = tokenizer
        self.text_maxlength = text_maxlength
        self.answer_maxlength = answer_maxlength

    def __call__(self, batch):
        assert(batch[0]['target'] != None)
        index = torch.tensor([ex['index'] for ex in batch])
        target = [ex['target'] for ex in batch]
        target = self.tokenizer.batch_encode_plus(
            target,
            max_length=self.answer_maxlength if self.answer_maxlength > 0 else None,
            pad_to_max_length=True,
            return_tensors='pt',
            truncation=True if self.answer_maxlength > 0 else False,
        )
        target_ids = target["input_ids"]
        target_mask = target["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        def append_question(example):
            if example['passages'] is None:
                return [example['question']]
            return [example['question'] + " " + t for t in example['passages']]
        text_passages = [append_question(example) for example in batch]
        passage_ids, passage_masks = encode_passages(text_passages,
                                                     self.tokenizer,
                                                     self.text_maxlength)

        return (index, target_ids, target_mask, passage_ids, passage_masks)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fnlp/bart-base-chinese")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--data_type", default="train")
    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    train_dataset = ChitChatDataset(tokenizer, args, 'train')
    d = train_dataset.get_dataloader(batch_size=4, shuffle=False, num_workers=8)

    for dd in tqdm(d):
        pass