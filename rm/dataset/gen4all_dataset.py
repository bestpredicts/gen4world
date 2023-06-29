
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





PROMPT_BEGIN: str = ''
PROMPT_USER: str = 'Human: \n{input} '
PROMPT_ASSISTANT: str = '\n\nAssistant: \n'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT



class PreferenceDataset:
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

        sample_id,  raw_sample= self.all_data[item]


        prompt = PROMPT_INPUT.format(input=raw_sample['prompt'])

        better_answer = raw_sample['chosen']
        worse_answer = raw_sample['rejected']

        better_input_ids = self.tokenizer.encode(prompt + better_answer + self.tokenizer.eos_token, add_special_tokens=False)
        worse_input_ids = self.tokenizer.encode(prompt + worse_answer + self.tokenizer.eos_token, add_special_tokens=False)
  

        return {
            'better_input_ids': better_input_ids,  # size = (L,)
            'worse_input_ids': worse_input_ids,  # size = (L,)
        }
    

   