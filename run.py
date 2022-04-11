from transformers import BertTokenizer
from transformers.models.bart import BartForConditionalGeneration
import argparse
from train import Trainer, Tester


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="fnlp/bart-large-chinese", type=str)
    parser.add_argument("--state_dict", default=None, type=str)
    parser.add_argument("--data_path", default="data", type=str)
    parser.add_argument("--task_type", default="train", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--num_warmup_steps", default=2000, type=int)
    parser.add_argument("--epoch", default=10, type=int)
    parser.add_argument("--batch_expand_times", default=2, type=int)

    return parser.parse_args()

if __name__ == '__main__':
    args = init_args()
    tokenizer = BertTokenizer.from_pretrained(args.model)
    model = BartForConditionalGeneration.from_pretrained(args.model)
    trainer = Trainer(model=model, tokenizer=tokenizer, args=args)
    # tester = Tester(model=model, tokenizer=tokenizer, args=args)
    trainer.train()
    # tester.test()