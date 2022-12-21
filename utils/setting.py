import torch
import random
import logging
import numpy as np
import transformers

from argparse import ArgumentParser


class Arguments(object):

    def __init__(self):
        self.parser = ArgumentParser()

    def add_type_of_processing(self):
        self.add_argument('--seed', type=int, default=42)
        self.add_argument('--is_train', action='store_true')
        self.add_argument('--use_amp', action='store_true')
        self.add_argument('--device', type=str, default='cpu')
        self.add_argument('--wandb', action='store_true')

    def add_hyper_parameters(self):
        self.add_argument('--method', default='ce', type=str, choices=['ce', 'scl', 'dualcl'])
        self.add_argument('--model_name', default='tbert', type=str, choices=['tbert', 'bert', 'roberta'])
        self.add_argument('--max_len', type=int, default=256)
        self.add_argument('--epochs', type=int, default=3)
        self.add_argument('--lr', type=float, default=5e-5)
        self.add_argument('--warmup_proportion', default=0.06, type=float)
        self.add_argument('--eval_steps', default=1000, type=int)
        self.add_argument('--adam_epsilon', default=1e-8, type=float)
        self.add_argument('--max_grad_norm', default=1, type=int)
        self.add_argument('--weight_decay', default=0.1, type=float)
        self.add_argument('--num_classes', default=2, type=int)
        self.add_argument('--warmup_steps', default=0, type=int)
        self.add_argument('--train_batch_size', default=32, type=int, choices=[1, 4, 8, 16, 32])
        self.add_argument('--valid_batch_size', default=128, type=int, choices=[64, 128, 256])
        self.add_argument('--accumulation_steps', default=1, type=int, choices=[1, 2, 4, 8, 16])
        self.add_argument('--patience', default=7, type=int)
        self.add_argument('--alpha', default=0.5, type=float)
        self.add_argument('--temp', default=0.1, type=float)

    def add_data_parameters(self):
        self.add_argument('--data_path', default='./data', type=str)
        self.add_argument('--output_path', type=str, default='./model/saved_model')
        self.add_argument('--pretrained_model_path', default='./model/tbert_1.9/', type=str)
        self.add_argument('--pretrained_tokenizer_path', default='./tokenizer/version_1.9/', type=str)
        self.add_argument('--saved_model_state_path', default=None, type=str)

    def print_args(self, args):
        for idx, (key, value) in enumerate(args.__dict__.items()):
            if idx == 0:
                print("argparse{\n", "\t", key, ":", value)
            elif idx == len(args.__dict__) - 1:
                print("\t", key, ":", value, "\n}")
            else:
                print("\t", key, ":", value)

    def add_argument(self, *args, **kw_args):
        return self.parser.add_argument(*args, **kw_args)

    def parse(self):
        args = self.parser.parse_args()
        if args.device == '0':
            args.device = torch.device('cuda:0')
        elif args.device == '1':
            args.device = torch.device('cuda:1')
        else:
            args.device = torch.device('cpu')

        self.print_args(args)

        return args


class Setting(object):

    def set_logger(self):
        _logger = logging.getLogger()
        formatter = logging.Formatter(
            '[%(levelname)s] %(asctime)s [ %(message)s ] | file::%(filename)s | line::%(lineno)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        _logger.addHandler(stream_handler)
        _logger.setLevel(logging.INFO)

        transformers.logging.set_verbosity_error()

        return _logger

    def set_seed(self, args):
        seed = args.seed

        random.seed(seed)
        np.random.seed(seed)

        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def run(self):
        parser = Arguments()
        parser.add_type_of_processing()
        parser.add_hyper_parameters()
        parser.add_data_parameters()

        args = parser.parse()
        logger = self.set_logger()
        self.set_seed(args)

        return args, logger
