import os

import numpy as np
import pandas as pd
import torch

from konlpy.tag import Mecab
from torch.utils.data import Dataset, DataLoader


class NSMCDataset(Dataset):

    def __init__(self, args, raw_data, tokenizer, mecab, label_dict):
        self.args = args
        self.data = raw_data
        self.data['mecab_document'] = self.data['document'].apply(lambda x: mecab.morphs(x))
        if args.method == 'dualcl':
            self.collate_fn = CollateDualCL(args, tokenizer, label_dict, mecab)
        else:
            self.collate_fn = CollateCESCL(args, tokenizer, mecab)
        self.loader = DataLoader(dataset=self,
                                 batch_size=args.train_batch_size if args.is_train else args.valid_batch_size,
                                 shuffle=True if args.is_train else False,
                                 sampler=None,
                                 collate_fn=self.collate_fn)

    def __getitem__(self, index):
        # sentence
        sentence = self.data['mecab_document'].iloc[index]

        # label
        if self.args.is_train:
            label = self.data['label'].iloc[index]
            return {'sentence': sentence, 'label': label}
        return {'sentence': sentence}

    def __len__(self):
        return len(self.data)


class CollateCESCL:

    def __init__(self, args, tokenizer, mecab):
        self.args = args
        self.tokenizer = tokenizer
        self.mecab = mecab

    def __call__(self, batches):
        b_input_ids = []
        b_attention_mask = []
        b_token_type_ids = []
        b_label = []

        for b in batches:
            mecab_sentence = ' '.join(b['sentence'])
            tokenized_result = self.tokenizer(mecab_sentence,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.args.max_len)

            input_ids = tokenized_result['input_ids']
            attention_mask = tokenized_result['attention_mask']
            token_type_ids = tokenized_result['token_type_ids']

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            b_token_type_ids.append(torch.tensor(token_type_ids, dtype=torch.long))
            if self.args.is_train:
                b_label.append(b['label'])

        t_input_ids = torch.stack(b_input_ids)  # List[Tensor] -> Tensor List
        t_attention_mask = torch.stack(b_attention_mask)  # List[Tensor] -> Tensor List
        t_token_type_ids = torch.stack(b_token_type_ids)  # List[Tensor] -> Tensor List

        if self.args.is_train:
            t_label = torch.tensor(b_label)  # List -> Tensor
            return {'input_ids': t_input_ids,
                    'attention_mask': t_attention_mask,
                    'token_type_ids': t_token_type_ids,
                    }, t_label
        return {'input_ids': t_input_ids,
                'attention_mask': t_attention_mask,
                'token_type_ids': t_token_type_ids,
                }


class CollateDualCL:

    def __init__(self, args, tokenizer, label_dict, mecab):
        self.args = args
        self.tokenizer = tokenizer
        self.label_dict = label_dict
        self.mecab = mecab

    def __call__(self, batches):
        b_input_ids = []
        b_attention_mask = []
        b_token_type_ids = []
        b_position_ids = []
        b_label = []

        for b in batches:
            label_list = list(self.label_dict.keys())
            tokens = b['sentence']
            sep_token = [self.tokenizer.sep_token]
            tokens = label_list + sep_token + tokens

            tokenized_result = self.tokenizer(tokens,
                                              padding='max_length',
                                              truncation=True,
                                              max_length=self.args.max_len,
                                              is_split_into_words=True,
                                              add_special_tokens=True)

            input_ids = tokenized_result['input_ids']
            attention_mask = tokenized_result['attention_mask']
            token_type_ids = tokenized_result['token_type_ids']

            # positions = np.zeros_like(input_ids)
            # # [CLS] 와 Label token 의 position_id 를 0으로 해줘서 위치에 상관 없는 예측이 가능함
            # positions[self.args.num_classes:] = np.arange(0, len(input_ids) - self.args.num_classes)
            # position_ids = positions

            b_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            b_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))
            b_token_type_ids.append(torch.tensor(token_type_ids, dtype=torch.long))
            # b_position_ids.append(torch.tensor(position_ids, dtype=torch.long))
            if self.args.is_train:
                b_label.append(b['label'])

        t_input_ids = torch.stack(b_input_ids)  # List[Tensor] -> Tensor List
        t_attention_mask = torch.stack(b_attention_mask)  # List[Tensor] -> Tensor List
        t_token_type_ids = torch.stack(b_token_type_ids)  # List[Tensor] -> Tensor List
        # t_position_ids = torch.stack(b_position_ids)  # List[Tensor] -> Tensor List

        if self.args.is_train:
            t_label = torch.tensor(b_label)  # List -> Tensor
            return {'input_ids': t_input_ids,
                    'attention_mask': t_attention_mask,
                    'token_type_ids': t_token_type_ids,
                    # 'position_ids': t_position_ids,
                    }, t_label
        return {'input_ids': t_input_ids,
                'attention_mask': t_attention_mask,
                'token_type_ids': t_token_type_ids,
                # 'position_ids': t_position_ids,
                }


def load_dataloader(args, tokenizer):
    train_data = pd.read_csv(os.path.join(args.data_path, 'ratings_train.txt'), sep='\t')
    train_data = train_data.dropna()
    test_data = pd.read_csv(os.path.join(args.data_path, 'ratings_test.txt'), sep='\t')
    test_data = test_data.dropna()
    label_dict = {'[NEG]': 0, '[POS]': 1}

    mecab = Mecab()

    train_dataset = NSMCDataset(args, train_data, tokenizer, mecab, label_dict)
    test_dataset = NSMCDataset(args, test_data, tokenizer, mecab, label_dict)

    train_dataloader = train_dataset.loader
    test_dataloader = test_dataset.loader

    return {'train': train_dataloader,
            'valid': test_dataloader,
            }
