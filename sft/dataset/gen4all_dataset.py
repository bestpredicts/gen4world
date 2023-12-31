
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

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""




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




class GEN4ALLDatasetConv:
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
        with open(self.data_path, mode="r") as reader:
            for line in tqdm(reader):
                try:
                    data = json.loads(line)
                except:
                    print("ERROR_JSON_LINE: ", line)
                if type(data) != dict:
                    continue
                id = data['id']
                all_data.append((id, data))

        return all_data

    def __getitem__(self, item):
        sample_id = self.all_data[item][0]

        sample_id,  data_point = self.all_data[item]
        source = data_point['conversations']
        source = data_point['conversations']

        input_ids = []
        labels = []
        source = data_point["conversations"]

        PROMPT_BEGIN: str = ''
        PROMPT_USER: str = '\n\nHuman: {input} '
        PROMPT_ASSISTANT: str = '\n\nAssistant: '  # should not have a space at the end
        PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

        for idx,sentence in enumerate(source):
            sentence_from = sentence["from"].lower()
            # https://github.com/LianjiaTech/BELLE/issues/337
            if sentence_from == 'system':
                sentence_value = B_SYS+sentence['value']+E_SYS
            else:
                sentence_value = PROMPT_INPUT.format(input=sentence['value']) if sentence_from == 'human' else sentence["value"]

            sentence_ids = self.tokenizer.encode(
                sentence_value, add_special_tokens=False)  # do not add bos_token_id
            label = copy.deepcopy(sentence_ids) if sentence_from not in ['human','system']  else [
                IGNORE_INDEX] * len(sentence_ids)
            input_ids += sentence_ids
            labels += label
            # add eos at every end of assistant sentence
            if sentence_from != 'human':
                # make sure eos_token_id is correct
                input_ids += [self.tokenizer.eos_token_id]
                labels += [self.tokenizer.eos_token_id]

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]
        if not any(x > -100 for x in labels):
            # labels can not have all values being -100. 18 and 24 are just random numbers
            labels[18:24] = input_ids[18:24]

        attention_mask = [1] * len(input_ids)

        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        labels = torch.tensor(labels, dtype=torch.long)
        
        tokenized_full_prompt = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return tokenized_full_prompt
