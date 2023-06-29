#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
from model.rm_baichuan import BaiChuanModelForScore,ScoreModelOutput
from typing import Any
import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm

import transformers
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
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from typing import Dict, Optional, Sequence
from transformers import LlamaTokenizerFast

from transformers.utils import (
    check_min_version,
    get_full_repo_name,
    send_example_telemetry,
)
from accelerate.utils import ProjectConfiguration, set_seed


from transformers.utils.versions import require_version
from dataset import gen4all_dataset
from dataset import gen4all_datacollate
from util.gen4all_util import AverageMeter, get_learning_rate
import numpy as np

logger = get_logger(__name__)


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


    


def loss_fn(
    args,
    model,
    better_input_ids: torch.LongTensor,  # size = (B, L)
    better_attention_mask: torch.BoolTensor,  # size = (B, L)
    worse_input_ids: torch.LongTensor,  # size = (B, L)
    worse_attention_mask: torch.BoolTensor,  # size = (B, L)
) :
    """Loss function for the reward model.

    Args:
        better_input_ids (torch.LongTensor): The input ids of the better answer.
        better_attention_mask (torch.BoolTensor): The attention mask of the better answer.
        worse_input_ids (torch.LongTensor): The input ids of the worse answer.
        worse_attention_mask (torch.BoolTensor): The attention mask of the worse answer.

    Returns:
        dict[str, torch.Tensor]: loss, higher_end_rewards, lower_end_rewards, accuracy
    """
    assert better_input_ids.size(0) == worse_input_ids.size(0), 'batch size mismatch!'
    batch_size = better_input_ids.size(0)


    output: ScoreModelOutput = model(
        input_ids = torch.cat([better_input_ids, worse_input_ids], dim=0),
        attention_mask=torch.cat([better_attention_mask, worse_attention_mask], dim=0),
    )
    scores = output.scores  # size = (2 * B, L, 1)
    end_scores = output.end_scores  # size = (2 * B, 1)
    # size = (B, L)
    higher_rewards, lower_rewards = scores.squeeze(dim=-1).chunk(chunks=2, dim=0)
    # size = (B,)
    higher_end_rewards, lower_end_rewards = end_scores.squeeze(dim=-1).chunk(chunks=2, dim=0)



    if args.loss_type == 'token-wise':
        losses = []
        for i in range(batch_size):
            assert not torch.all(
                torch.eq(better_input_ids[i], worse_input_ids[i]),
            ).item(), 'The better and worse answers are the same!'
            higher_end_index = better_attention_mask[i].nonzero()[-1]
            lower_end_index = worse_attention_mask[i].nonzero()[-1]
            end_index = max(higher_end_index, lower_end_index)

            divergence_index = (better_input_ids[i] != worse_input_ids[i]).nonzero()[0]
            assert 0 <= divergence_index <= end_index, 'divergence index is out of range!'

            # size = (L,)
            higher_truncated_rewards = higher_rewards[i, divergence_index : end_index + 1]
            lower_truncated_rewards = lower_rewards[i, divergence_index : end_index + 1]

            losses.append(
                -F.logsigmoid(higher_truncated_rewards - lower_truncated_rewards).mean(),
            )

        loss = torch.stack(losses).mean()  # size = ()
    elif args.loss_type == 'sequence-wise':
        loss = -F.logsigmoid(higher_end_rewards - lower_end_rewards).mean()
    else:
        raise ValueError(f'Unknown loss type: {args.loss_type}')

    accuracy = (higher_end_rewards > lower_end_rewards).float().mean()  # size = ()
    return {
        'loss': loss,  # size = ()
        'higher_end_rewards': higher_end_rewards,  # size = (B,)
        'lower_end_rewards': lower_end_rewards,  # size = (B,)
        'accuracy': accuracy,  # size = ()
    }
        

def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a causal language modeling task"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default=None,
        help="A csv or a json file containing the training data.",
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        default=None,
        help="A csv or a json file containing the validation data.",
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="token-wise",
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--no_keep_linebreaks",
        action="store_true",
        help="Do not keep line breaks when using TXT files.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--hub_token", type=str, help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--save_state",
        action="store_true",
        help=(
            "IF need save_state."
        ),
    )

    parser.add_argument(
        "--use_int8_training",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--lora_config",
        type=str,
        default=None,
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    parser.add_argument(
        "--gradient_checkpointing_enable",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=None,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--eval_step",
        type=int,
        default=1,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry(args.output_dir, args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    ds_gradient_accumulation_steps = 1
    # mixed_precision, deepspeed_plugin = get_deepspeed_plugin(
    #     ds_gradient_accumulation_steps)

    # accelerator = Accelerator(
    #     gradient_accumulation_steps=args.gradient_accumulation_steps,
    #     **accelerator_log_kwargs,
    # )

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=accelerator_project_config,
         **accelerator_log_kwargs,
    )




    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token
                )
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(
                args.output_dir, clone_from=repo_name, token=args.hub_token
            )

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name,trust_remote_code=True)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if  "llama" in args.model_name_or_path.lower() :
        logger.info("Setting speical LlamaTokenizer")
        tokenizer = LlamaTokenizerFast.from_pretrained(
            args.model_name_or_path
        ,trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True
        ,trust_remote_code=True)

    if args.model_name_or_path:
        model = BaiChuanModelForScore.from_pretrained(
            args.model_name_or_path,trust_remote_code=True)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # # tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"  # Allow batched inference
    if model.config.model_type.lower() == "llama":
        logger.info("Setting special tokens for LLAMA model")
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    if len(special_tokens_dict) > 0:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

    # peft model
    if args.use_lora:
        logger.info("Loading lora config from {}".format(args.lora_config))
        lora_config = json.load(open(args.lora_config))
        logger.info("Lora config: {}".format(lora_config))
        # if args.use_int8_training:
        #     logger.info(
        #         "training_args.use_int8_training!!! (int8 is not compatible with DeepSpeed)"
        #     )
        #     model = prepare_model_for_int8_training(model)
        config = LoraConfig(
            r=lora_config["lora_r"],
            lora_alpha=lora_config["lora_alpha"],
            target_modules=lora_config["lora_target_modules"],
            lora_dropout=lora_config["lora_dropout"],
            inference_mode=False,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        # model.config.use_cache = False

        # "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # old_state_dict = model.state_dict
        # model.state_dict = (
        #     lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        # ).__get__(model, type(model))

    if args.gradient_checkpointing_enable:
        model.gradient_checkpointing_enable()

    train_dataset = gen4all_dataset.PreferenceDataset(
        json_path=args.train_file,
        tokenizer=tokenizer,
        bos_token_id=None,
        eos_token_id=tokenizer.eos_token_id,
        max_length=args.max_length,
        data_tokenized=False,
    )

    # eval_dataset = None

    from torch.utils.data import random_split

    # 假设 train_dataset 是你的初始训练数据集
    # train_dataset = ...

    # 确定划分的长度
    total_size = len(train_dataset)
    dev_size = 300 # 剩下的作为开发集
    train_size = total_size - dev_size  # 使用80%的数据作为训练集
    eval_dataset= None 


    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    collate_fn = gen4all_datacollate.PreferenceCollator( tokenizer )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size,
    )
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            collate_fn=collate_fn,
            batch_size=args.per_device_eval_batch_size,
        )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, betas=(0.9, 0.999), lr=args.learning_rate
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if args.warmup_ratio is not None:
        args.warmup_ratio = args.warmup_ratio
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
    #     num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    # )
    # New Code #
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (
        accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer,
            total_num_steps=args.max_train_steps,
            warmup_num_steps=args.num_warmup_steps,
        )

    # Prepare everything with our `accelerator`.
    if eval_dataset is not None:
        (
            model,
            optimizer,
            train_dataloader,
            eval_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config[
            "lr_scheduler_type"
        ].value

        project_name = "default"
        if args.wandb_project is not None:
            project_name = args.wandb_project
        else:
            project_name = args.output_dir.split("/")[-1]

        accelerator.init_trackers(project_name, experiment_config)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)
            completed_steps = resume_step

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)



    if args.use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        if accelerator.is_main_process and args.with_tracking:
            loss_average = AverageMeter()
            acc_average = AverageMeter()

        if args.with_tracking:
            total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader

        

        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # outputs = model(**batch)
                # loss = outputs.loss
                better_input_ids = batch['better_input_ids']
                better_attention_mask = batch['better_attention_mask']
                worse_input_ids = batch['worse_input_ids']
                worse_attention_mask=batch['worse_attention_mask']

                result = loss_fn(args,model,better_input_ids,better_attention_mask,worse_input_ids,worse_attention_mask)

                loss = result['loss']
                train_acc = result['accuracy']

                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()

                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and completed_steps>0:
                    output_dir = f"step_{completed_steps }"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.wait_for_everyone()
                    if args.use_lora != True:
                        if args.save_state:
                            accelerator.save_state(output_dir)
                    accelerator.wait_for_everyone()
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    state_dict = accelerator.get_state_dict(model)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped_model.save_pretrained(
                            output_dir, state_dict=state_dict
                        )

                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)
                    accelerator.wait_for_everyone()

            if completed_steps >= args.max_train_steps:
                break

            if accelerator.is_main_process and args.with_tracking:
                loss_average.update(loss.detach().float().item())
                acc_average .update(train_acc.detach().float().item())

                learning_rate = get_learning_rate(lr_scheduler, logger, deepspeed=True)

            if accelerator.is_main_process and completed_steps % args.eval_step == 0:
                if args.with_tracking:
                    accelerator.log(
                        {
                            "train_loss": loss_average.avg,
                            "train_acc": acc_average.avg,
                            "learning_rate": learning_rate,
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )
                loss_average.reset()  # reset for next step

        if eval_dataset is not None:
            model.eval()
            losses = []
            acces = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    better_input_ids = batch['better_input_ids']
                    better_attention_mask = batch['better_attention_mask']
                    worse_input_ids = batch['worse_input_ids']
                    worse_attention_mask=batch['worse_attention_mask']

                    result = loss_fn(args,model,better_input_ids,better_attention_mask,worse_input_ids,worse_attention_mask)

                    loss = result['loss']
                    dev_acc = result['accuracy']

                losses.append(
                    accelerator.gather_for_metrics(
                        loss.repeat(args.per_device_eval_batch_size)
                    )
                )

                acces.append(
                    accelerator.gather_for_metrics(
                        dev_acc.repeat(args.per_device_eval_batch_size)
                    )
                )


            losses = torch.cat(losses)
            eval_loss = torch.mean(losses)
            eval_acc =  torch.mean(acces)

            if accelerator.is_main_process and args.with_tracking:
                accelerator.log(
                    {
                        "dev_acc": eval_acc,
                        "epoch": epoch,
                        "step": completed_steps,
                    },
                    step=completed_steps,
                )

        if args.push_to_hub and epoch < args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}",
                    blocking=False,
                    auto_lfs_prune=True,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)

            accelerator.wait_for_everyone()
            if args.save_state:
                accelerator.save_state(output_dir)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            unwrapped_model.save_pretrained(
                output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model),
            )
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
            accelerator.wait_for_everyone()

            if args.use_lora:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                model.save_pretrained(args.output_dir)
                accelerator.wait_for_everyone()

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)



if __name__ == "__main__":
    main()
