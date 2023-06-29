
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # seq2seq package path
sys.path.insert(0, os.getcwd())

import logging
import math

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
import argparse
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.types import Number
import torch
from typing import Any, ClassVar
from typing import List, Union


def right_padding(sequences: List[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return pad_sequence(sequences, batch_first=True, padding_value=padding_value)


def left_padding(sequences: List[torch.Tensor], padding_value: Number) -> torch.Tensor:
    return right_padding(
        [seq.flip(0) for seq in sequences],
        padding_value=padding_value,
    ).flip(1)



IGNORE_INDEX = -100
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"


class PreferenceCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        input_ids = [sample['better_input_ids'] for sample in samples] + [
            sample['worse_input_ids'] for sample in samples
        ]  # size = (2 * B, L)
        attention_mask = [[1] * len(sample)
                                    for sample in input_ids]
        
        # size = (2 * B, L)




        batch_max = max([len(ids) for ids in input_ids])

        # add padding
        # if self.tokenizer.padding_side == "right":
        input_ids = [
            s + (batch_max - len(s)) * [self.tokenizer.pad_token_id]
            for s in input_ids]
        
        attention_mask = [
            s + (batch_max - len(s)) * [0] for s in attention_mask
        ]


        # convert to tensors
        input_ids = torch.tensor(
            np.array(input_ids), dtype=torch.long
        )
        attention_mask= torch.tensor(
            np.array(attention_mask), dtype=torch.long
        )



        (
            better_input_ids,  # size = (B, L)
            worse_input_ids,  # size = (B, L)
        ) = input_ids.chunk(chunks=2, dim=0)
        (
            better_attention_mask,  # size = (B, L)
            worse_attention_mask,  # size = (B, L)
        ) = attention_mask.chunk(chunks=2, dim=0)


        return {
            'better_input_ids': better_input_ids,  # size = (B, L)
            'better_attention_mask': better_attention_mask,  # size = (B, L)
            'worse_input_ids': worse_input_ids,  # size = (B, L)
            'worse_attention_mask': worse_attention_mask,  # size = (B, L)
        }
    
