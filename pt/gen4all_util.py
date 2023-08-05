
from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from accelerate.utils import DummyOptim, DummyScheduler
from transformers.utils.versions import require_version
from transformers.utils import (
    check_min_version,
    get_full_repo_name,
    send_example_telemetry,
)
from torch.utils.data import Dataset, DataLoader

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
import pandas as pd
import transformers
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from huggingface_hub import Repository, create_repo
from datasets import load_dataset
from accelerate.utils import set_seed
# from accelerate.logging import get_logger
from accelerate import Accelerator, DistributedType
import torch
import datasets
from pathlib import Path
from itertools import chain
import random
import argparse
import json
import logging
import math
import os
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
import numpy as np

from accelerate.state import AcceleratorState
from accelerate.utils import DistributedType
import sys
import datetime
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
import torch
from typing import Dict, Optional, Sequence

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['WANDB_DISABLE_SERVICE'] = 'true'


MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "[UNK]"

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg




def setup_logger(
    LOGGER, out_file="log.txt", stderr=True, stderr_level=logging.INFO, file_level=logging.INFO
):
    LOGGER.handlers = []
    LOGGER.setLevel(min(stderr_level, file_level))
    FORMATTER = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if stderr:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(FORMATTER)
        handler.setLevel(stderr_level)
        LOGGER.addHandler(handler)

    if out_file is not None:
        handler = logging.FileHandler(out_file)
        handler.setFormatter(FORMATTER)
        handler.setLevel(file_level)
        LOGGER.addHandler(handler)

    LOGGER.info("logger set up")
    return LOGGER


class MultiProcessAdapter(logging.LoggerAdapter):
    """
    An adapter to assist with logging in multiprocess.
    `log` takes in an additional `main_process_only` kwarg, which dictates whether it should be called on all processes
    or only the main executed one. Default is `main_process_only=True`.
    """

    @staticmethod
    def _should_log(main_process_only):
        "Check if log should be performed"
        state = AcceleratorState()
        if state.distributed_type != DistributedType.MEGATRON_LM:
            process_index_flag = state.local_process_index == 0
        else:
            process_index_flag = state.process_index == state.num_processes - 1
        return not main_process_only or (main_process_only and process_index_flag)

    def log(self, level, msg, *args, **kwargs):
        """
        Delegates logger call after checking if we should log.
        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed
        """
        main_process_only = kwargs.pop("main_process_only", True)
        if self.isEnabledFor(level) and self._should_log(main_process_only):
            msg, kwargs = self.process(msg, kwargs)
            self.logger.log(level, msg, *args, **kwargs)


def get_logger(name: str, logger_path: str, log_level: str = None):
    """
    Returns a `logging.Logger` for `name` that can handle multiprocessing.
    If a log should be called on all processes, pass `main_process_only=False`
    Args:
        name (`str`):
            The name for the logger, such as `__file__`
        log_level (`str`, *optional*):
            The log level to use. If not passed, will default to the `LOG_LEVEL` environment variable, or `INFO` if not
    Example:
    ```python
    >>> from accelerate.logging import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("My log", main_process_only=False)
    >>> logger.debug("My log", main_process_only=True)
    >>> logger = get_logger(__name__, accelerate_log_level="DEBUG")
    >>> logger.info("My log")
    >>> logger.debug("My second log")
    ```
    """
    if log_level is None:
        log_level = os.environ.get("ACCELERATE_LOG_LEVEL", None)
    logger = logging.getLogger(name)
    setup_logger(logger, out_file=logger_path)
    if log_level is not None:
        logger.setLevel(log_level.upper())
    return MultiProcessAdapter(logger, {})


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


"""
合并lora权重到base模型

"""


def apply_lora(base_model_path, target_model_path, lora_path):
    print(f"Loading the base model from {base_model_path}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, use_fast=False)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    print(f"Saving the target model to {target_model_path}")
    model.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)


# In order to keep `trainer.py` compact and easy to understand, place any secondary PT Trainer
# helper methods here


def get_learning_rate(lr_scheduler, logger, deepspeed=True):
    if deepspeed:
        # with deepspeed's fp16 and dynamic loss scale enabled the optimizer/scheduler steps may
        # not run for the first few dozen steps while loss scale is too large, and thus during
        # that time `get_last_lr` will fail if called during that warm up stage, so work around it:
        try:
            last_lr = lr_scheduler.get_last_lr()[0]
        except AssertionError as e:
            if "need to call step" in str(e):
                logger.warning(
                    "tried to get lr value before scheduler/optimizer started stepping, returning lr=0")
                last_lr = 0
            else:
                raise
    else:
        last_lr = lr_scheduler.get_last_lr()[0]
        if torch.is_tensor(last_lr):
            last_lr = last_lr.item()
    return last_lr
