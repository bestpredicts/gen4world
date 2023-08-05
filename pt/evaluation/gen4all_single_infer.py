#!/usr/bin/env python
# coding: utf-8

import sys
import os
# sys.path.insert(0, os.path.dirname(__file__))  # seq2seq package path
# sys.path.insert(0, os.getcwd())

import time
import pandas as pd
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
import torch
import numpy as np
import argparse
import logging
import numpy as np
import os
import argparse
from peft import PeftModel
import json 
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


BEAM_SEARCH = False 


def model_infer_single(model, tokenizer, args, prompt_text):
    prompt_text = 'Human: \n' +  prompt_text + '\n\nAssistant: \n'
    encoded_prompt = tokenizer.encode(
        prompt_text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)
    if BEAM_SEARCH:
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_new_tokens=1024,  num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=2,
            top_p=0.95,
            top_k=30,
            repetition_penalty=1.2,
            do_sample=False,
            temperature=0.01)
    else:
        # https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/inference.py
        output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_new_tokens=1024,  num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            do_sample=True,
            temperature=1.0)
        


    # Remove the batch dimension when returning multiple sequences
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    generated_sequences = []

    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = tokenizer.decode(
            generated_sequence, clean_up_tokenization_spaces=True)

        # Remove all text after the stop token

        text = text[: text.find(args.stop_token) if args.stop_token else None]

        # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
        total_sequence = (
            text[len(tokenizer.decode(encoded_prompt[0],
                     clean_up_tokenization_spaces=True)):]
        )

        generated_sequences.append(total_sequence)

    return generated_sequences



def arg_parse():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--model_path', type=str, default=None,
                        help='base model path')
    parser.add_argument('--model_type', type=str, default="bloom")
    parser.add_argument("--use_lora", action="store_true", help="use lora") 
    parser.add_argument("--lora_model", type=str, default=None )
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str,  default=None)

    parser.add_argument('--device', type=str, default="cuda",
                        help='device')
    parser.add_argument('--stop_token', type=str, default=None,
                        help='stop_token')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='max_new_tokens')
    parser.add_argument('--do_sample', type=bool, default=True,
                        help='do_sample')
    parser.add_argument('--num_return_sequences', type=int, default=1,
                        help='num_return_sequences')
    parser.add_argument('--top_k', type=int, default=50,
                        help='top_k')
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    args = arg_parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path ,device_map='auto',torch_dtype=torch.float16,use_cache=False,trust_remote_code=True)
    
    if args.use_lora:
        model = PeftModel.from_pretrained(
        model,
        args.lora_model,
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True,trust_remote_code=True)
    tokenizer.padding_side = "left"  # Allow batched inference

    if args.model_type.lower()=='llama':
        print("add llama special tokens")
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 2
        tokenizer.bos_token_id = 1
    args.tokenizer = tokenizer

    test_txt = ["你好啊", "写一篇情书", "介绍一下百度", "介绍一下周杰伦", "以难而正确为主题写一首作文",
                "周杰伦的出生日期是啥？", "今天是什么节日", "人类存在的意义是什么?","实现一个二分法查找算法","实现一个冒泡排序算法"]
    for txt in test_txt:
        print("-----------------------------------------------------")
        print("输入：", txt)
        start = time.time()
        ans = model_infer_single(model, tokenizer, args, txt)[0]
        toatal_tokens = len(tokenizer(txt+" "+ans)['input_ids'])
        print("输出：", ans)
        print("耗时：", time.time()-start)
        print("总token数：", toatal_tokens)
        print("-----------------------------------------------------")

    belle_tests = []
    with open("eval_data/belle_test.test.json",'r') as f:
        for l in f.readlines():
            belle_tests.append(json.loads(l))

    
    with open(args.output_file,'w') as f:
        for test in tqdm(belle_tests):
            ans = model_infer_single(model, tokenizer, args, test['instruction'])[0]
            test['test_response'] = ans
            new_test = {}
            new_test['id']=test['id']
            new_test['prompt']=test['instruction']
            new_test['response']=ans.replace("</s>","")
            f.write(json.dumps(new_test,ensure_ascii=False)+'\n')

