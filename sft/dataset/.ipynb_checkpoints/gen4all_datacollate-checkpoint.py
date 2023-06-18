
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import datasets
import evaluate
import torch
from datasets import load_dataset
import numpy as np
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
import torch

IGNORE_INDEX = -100


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [
                torch.as_tensor(instance[key], dtype=torch.int64)
                for instance in instances
            ]
            for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        torch.set_printoptions(profile="full")
        return ret
    
    
class CLMCollate:
    def __init__(self, tokenizer, isTrain=True, user_max_length=None):
        self.tokenizer = tokenizer
        self.isTrain = isTrain
        self.user_max_length = user_max_length

    def __call__(self, batch):
        output = dict()
        output["input_ids"] = [sample["input_ids"] for sample in batch]
        # output["attention_mask"] = [sample["attention_mask"]
        #                             for sample in batch]
        output["attention_mask"] = [[1] * len(sample["input_ids"])
                                    for sample in batch]
        if self.isTrain:
            output["labels"] = [sample["labels"]
                                for sample in batch]

        # calculate max token length of this batch
        # calculate max token length of this batch
        if self.user_max_length is not None:
            batch_max = self.user_max_length
        else:
            batch_max = max([len(ids) for ids in output["input_ids"]])

        # add padding
        # if self.tokenizer.padding_side == "right":
        output["input_ids"] = [
            s + (batch_max - len(s)) * [self.tokenizer.pad_token_id]
            for s in output["input_ids"]
        ]
        output["attention_mask"] = [
            s + (batch_max - len(s)) * [0] for s in output["attention_mask"]
        ]

        output["labels"] = [

            s + (batch_max - len(s)) * [-100] for s in output["labels"]
        ]

        # convert to tensors
        output["input_ids"] = torch.tensor(
            np.array(output["input_ids"]), dtype=torch.long
        )
        output["attention_mask"] = torch.tensor(
            np.array(output["attention_mask"]), dtype=torch.long
        )
        if self.isTrain:
            output["labels"] = torch.tensor(
                np.array(output["labels"]), dtype=torch.long
            )

        return {
            "input_ids": output["input_ids"],
            "attention_mask": output["attention_mask"],
            "labels": output["labels"]}