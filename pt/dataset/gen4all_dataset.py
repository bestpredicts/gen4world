
# -*- coding: utf-8 -*-
# @file gen4all_dataset.py
# @author DengYong
# @date 2023-05-15
# @copyright Copyright (c) KE
# Description:

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # seq2seq package path
sys.path.insert(0, os.getcwd())
import multiprocessing
import json
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import BertTokenizer
from tqdm import tqdm
import random
import os
import itertools
import copy
import argparse
from transformers import AutoTokenizer
import json

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"

class Pretrain_DatasetIter(IterableDataset):
    def __init__(self,
                 json_path,
                 tokenizer: BertTokenizer,
                 bos_token_id = None,
                 eos_token_id = None,
                 max_length = 1024,
                 pretrain = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pretrain=pretrain
        self.data_path = json_path 

        if self.eos_token_id is not None:
            print ('Add eos token to the end of the input')
        if self.bos_token_id is not None:
            print ('Add bos token to the beginning of the input')


    def get_file_line_count(self):
        with open(self.data_path, 'r') as file:
            line_count = sum(1 for _ in file)
        return line_count
    
    def __len__(self):
        return self.data_size

    def line_mapper(self, line):
        data = json.loads(line)
        data = str(data)

        input_ids = self.tokenizer.encode(
                data, add_special_tokens=False)

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        if not any(x > -100 for x in labels):
            # labels can not have all values being -100. 18 and 24 are just random numbers
            labels[18:24] = input_ids[18:24]

        attention_mask = [1] * len(input_ids)
        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return tokenized_full_prompt

    def __iter__(self):   
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        world_size =  torch.distributed.get_world_size() if worker_info is not None else 1
        process_rank =  torch.distributed.get_rank() if worker_info is not None else 0
        worker_num_per_world = worker_info.num_workers if worker_info is not None else 1

        worker_total_num = world_size * worker_num_per_world
        worker_id = process_rank * worker_num_per_world + worker_id

        # Load lines lazily
        with open(self.data_path, 'r') as f:
            file_itr = f

        # Randomize data per epoch
        # Note: This may not be ideal as it involves a complete pass through the dataset before iteration
        file_itr = sorted(file_itr, key=lambda x: random.random())

        # Map each element using the line_mapper
        mapped_itr = map(self.line_mapper, file_itr)

        # Add multiworker functionality
        # itertools.islice(iterable, start, stop[, step])
        mapped_itr = itertools.islice(mapped_itr, worker_id, None, worker_total_num)

        # Filter out None elements (failed encodings)
        mapped_itr = filter(None, mapped_itr)

        return mapped_itr




class GEN4ALLDatasetPT:
    def __init__(self,
                 json_path,
                 tokenizer,
                 bos_token_id=None,
                 eos_token_id=None,
                 max_length=4096,
                 data_tokenized=False,
                 system_prompt=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_tokenized = data_tokenized
        self.data_path = json_path
        self.system_prompt = system_prompt

        self.all_data = self._load_data()

        if self.eos_token_id is not None:
            print(
                f'Add eos token (id: {self.tokenizer.eos_token_id}) to the end of the input')
        if self.bos_token_id is not None:
            print(
                f'Add bos token (id: {self.tokenizer.bos_token_id}) to the beginning of the input')

        print("{}: the number of examples = {}".format(
            json_path, len(self.all_data)))

    def __len__(self):
        return len(self.all_data)

    def _load_data(self):
        all_data = []
        remove_columns = ['id']
        with open(self.data_path, mode="r") as reader:
            for line in tqdm(reader):
                try:
                    data = json.loads(line)
                except:
                    print("ERROR_JSON_LINE: ", line)
                if type(data) != dict:
                    continue
                id = data['id']
                for c in remove_columns:
                    del data[c]
                all_data.append((id, data))

        return all_data

    def __getitem__(self, item):

        id,data_point = self.all_data[item]
        data_point = str(data_point)
        input_ids = self.tokenizer.encode(
                data_point, add_special_tokens=True)
        input_ids = input_ids[:self.max_length]
        labels = copy.deepcopy( input_ids)
        attention_mask = [1] * len(input_ids)

        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return tokenized_full_prompt

