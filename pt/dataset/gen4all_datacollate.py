
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

IGNORE_INDEX = -100
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "</s>"


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*

            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"]
                  for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * \
                    (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] +
                        remainder if padding_side == "right" else remainder +
                        feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str,
                        default="/data/pretrained_ckpt/bloomz-7b1-mt/")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
