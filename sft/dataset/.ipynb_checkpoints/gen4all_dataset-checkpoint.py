
# -*- coding: utf-8 -*-
# @file gen4all_dataset.py
# @author DengYong
# @date 2023-05-15
# @copyright Copyright (c) KE
# Description:


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
DEFAULT_UNK_TOKEN = "</s>"




class GeneralDatasetIter(IterableDataset):
    def __init__(self, json_path):
        self.data_path = json_path
        self.data_size = self.get_file_line_count()
        print(f'data_size: {self.data_size}')

    def get_file_line_count(self):
        with open(self.data_path, 'r') as file:
            line_count = sum(1 for _ in file)
        return line_count

    def line_mapper(self, line):
        data = json.loads(line)
        return data

    def __len__(self):
        return self.data_size

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # Single-process loading
            return self.get_data_iterator()
        else:  # Multi-process loading
            return self.get_data_iterator(worker_info)

    def get_data_iterator(self, worker_info=None):
        if worker_info is None:  # Single-process loading
            with open(self.data_path, 'r') as file:
                lines = file.readlines()  # 一次性读取所有数据，否则当模型在保存 评估时候，会出现io错误
            for line in lines:
                yield self.line_mapper(line)
        else:  # Multi-process loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            chunk_size = self.data_size // num_workers
            start_index = worker_id * chunk_size
            end_index = start_index + chunk_size

            with open(self.data_path, 'r') as file:
                lines = file.readlines()  # 一次性读取所有数据，否则当模型在保存 评估时候，会出现io错误

            # Read and yield lines from start_index to end_index
            for line in lines[start_index:end_index]:
                yield self.line_mapper(line)

            # Handle remainder lines
            if worker_id == num_workers - 1:
                # Read and yield remaining lines
                for line in lines[end_index:]:
                    yield self.line_mapper(line)




class GEN4ALLSFTDatasetAccelerate:
    def __init__(self, json_path, tokenizer, bos_token_id=None, eos_token_id=None, max_length=512, data_tokenized=False, num_processes=4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_tokenized = data_tokenized
        self.data_path = json_path

        self.all_data = self._load_data()
        self.all_data_processed = []
        print("start tokenizer data")
        self.num_processes = num_processes
        if self.num_processes is None:
            self.num_processes = multiprocessing.cpu_count()  # 使用默认进程数
        pool = multiprocessing.Pool(num_processes)  # 创建进程池

        import time
        import gc
        start = time.time()

        # 使用进程池的map方法并行处理数据
        self.all_data_processed = pool.map(self.process_data, range(
            len(self.all_data)), chunksize=len(self.all_data)//num_processes)

        print(f"tokenizer data time: {time.time() - start}")
        del self.all_data
        gc.collect()

        if self.eos_token_id is not None:
            print(
                f'Add eos token (id: {self.tokenizer.eos_token_id}) to the end of the input')
        if self.bos_token_id is not None:
            print(
                f'Add bos token (id: {self.tokenizer.bos_token_id}) to the beginning of the input')

        print("{}: the number of examples = {}".format(
            json_path, len(self.all_data_processed)))

        pool.close()  # 关闭进程池
    # 定义Worker函数，处理数据集中的每个数据项

    def process_data(self, item):
        return self.processed_first(item)

    def __len__(self):
        return len(self.all_data_processed)

    def _load_data(self):
        all_data = []
        with open(self.data_path, mode="r") as reader:
            for line in tqdm(reader):
                try:
                    data = json.loads(line)
                except:
                    print(line)
                if type(data) != dict:
                    continue
                id = data['id']
                all_data.append((id, data))
        return all_data

    def processed_first(self, item):
        sample_id = self.all_data[item][0]

        sample_id, source = self.all_data[item]
        source = source['conversations']

        def _addrole_masklabel_tokenize(source):
            '''add speaker and concatenate the sentences'''
            roles = {'human': 'Human: ', 'assistant': 'Assistant: '}
            conversation = ''
            input_ids = []
            labels = []
            test_input_ids = []
            for idx, sentence in enumerate(source):
                sentence_from = sentence["from"]
                sentence_value = roles[sentence_from] + str(sentence["value"]) + \
                    '\n\nAssistant: ' if sentence_from == 'human' else str(
                        sentence["value"])
                conversation += sentence_value
                sentence_ids = self.tokenizer.encode(
                    sentence_value, add_special_tokens=False)
                label = copy.deepcopy(sentence_ids) if sentence_from == 'assistant' else [
                    IGNORE_INDEX] * len(sentence_ids)

                # add eos at every end of assistant sentence
                if sentence_from == 'assistant':
                    sentence_ids += [self.tokenizer.eos_token_id]
                    label += [self.tokenizer.eos_token_id]

                input_ids += sentence_ids
                labels += label

                if idx < len(source) - 1:
                    test_input_ids += sentence_ids

            return input_ids, labels, test_input_ids, conversation

        input_ids, labels, test_input_ids, conversation = _addrole_masklabel_tokenize(
            source)


        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        # avoid all labels are -100, which will cause nan loss
        if not any(x > -100 for x in labels):
            labels[18:24] = input_ids[18:24]

        attention_mask = [1] * len(input_ids)

        # for inference
        input_input_ids = test_input_ids
        input_input_ids = input_input_ids[:self.max_length]

        input_attention_mask = [1] * len(input_input_ids)

        return {
            'input_ids': input_ids,
            # 'attention_mask': attention_mask,
            'labels': labels,
        }

    def __getitem__(self, item):
        return self.all_data_processed[item]


class GEN4ALLDataset(Dataset):
    def __init__(self,
                 json_path,
                 tokenizer,
                 bos_token_id=None,
                 eos_token_id=None,
                 max_length=1024,
                 data_tokenized=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_tokenized = data_tokenized
        self.data_path = json_path

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
                    print(line)
                if type(data) != dict:
                    continue
                id = data['id']
                if 'sharegpt' not in id:
                    input_ids = data['input']
                    target_ids = data['target']
                    all_data.append((id, input_ids, target_ids))
                else:
                    all_data.append((id, data))

        return all_data

    def __getitem__(self, item):
        sample_id = self.all_data[item][0]
        if 'sharegpt' not in sample_id:
            sample_id, input_ids, target_ids = self.all_data[item]

            if not self.data_tokenized:
                try:
                    input_ids = self.tokenizer.encode(
                        input_ids, add_special_tokens=False)
                    target_ids = self.tokenizer.encode(
                        target_ids, add_special_tokens=False)
                except:
                    input_ids = self.tokenizer.encode(
                        'Human: 你是怎么训练得到的？\n\nAssistant: ', add_special_tokens=False)
                    target_ids = self.tokenizer.encode(
                        '我是Belle，由Belle Group训练的大型语言模型。', add_special_tokens=False)
                    print('Error in encoding. Use default input and target.')

            if self.bos_token_id is not None:
                input_ids = [self.tokenizer.bos_token_id] + input_ids
            if self.eos_token_id is not None:
                target_ids = target_ids + [self.tokenizer.eos_token_id]

            whole_input_ids = input_ids + target_ids

            # trunctate
            whole_input_ids = whole_input_ids[:self.max_length]

            whole_attention_mask = [1] * len(whole_input_ids)

            inputs_length = len(input_ids)

            whole_labels = whole_input_ids.clone()
            target_only_labels = whole_input_ids.clone()
            target_only_labels[:inputs_length] = IGNORE_INDEX

            # for inference
            input_input_ids = input_ids
            input_input_ids = input_input_ids[:self.max_length]
            input_attention_mask = [1] * len(input_input_ids)

            return {
                "sample_ids": sample_id,
                "whole_input_ids": whole_input_ids,
                "whole_attention_mask": whole_attention_mask,
                "whole_labels": whole_labels,
                "target_only_labels": target_only_labels,
                "input_input_ids": input_input_ids,
                "input_attention_mask": input_attention_mask
            }
        else:
            sample_id, source = self.all_data[item]
            source = source['conversations']

            def _addrole_masklabel_tokenize(source):
                '''add speaker and concatenate the sentences'''
                roles = {'human': 'Human: ', 'assistant': 'Assistant: '}
                conversation = ''
                input_ids = []
                labels = []
                for sentence in source:
                    sentence_from = sentence["from"]
                    sentence_value = roles[sentence_from] + sentence["value"] + \
                        '\n\nAssistant: ' if sentence_from == 'human' else sentence["value"]
                    conversation += sentence_value
                    sentence_ids = self.tokenizer.encode(
                        sentence_value, add_special_tokens=False)
                    label = copy.deepcopy(sentence_ids) if sentence_from == 'assistant' else [
                        IGNORE_INDEX] * len(sentence_ids)
                    input_ids += sentence_ids
                    labels += label
                    # add eos at every end of assistant sentence
                    if sentence_from == 'assistant':
                        input_ids += [self.tokenizer.eos_token_id]
                        labels += [self.tokenizer.eos_token_id]
                return input_ids, labels, conversation

            input_ids, labels, conversation = _addrole_masklabel_tokenize(
                source)

            # if self.bos_token_id is not None:
            #     input_ids = [self.tokenizer.bos_token_id] + input_ids
            #     labels = [self.tokenizer.bos_token_id] + labels
            # if self.eos_token_id is not None:
            #     input_ids = input_ids + [self.tokenizer.eos_token_id]
            #     labels = labels + [self.tokenizer.eos_token_id]

            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

            # avoid all labels are -100, which will cause nan loss
            if not any(x > -100 for x in labels):
                labels[18:24] = input_ids[18:24]

            attention_mask = [1] * len(input_ids)

            return {
                'sample_ids': sample_id,
                'whole_input_ids': input_ids,
                'whole_attention_mask': attention_mask,
                'target_only_labels': labels,
                'whole_labels': labels,  # placeholder
            }


class GEN4ALLDatasetConv:
    def __init__(self,
                 json_path,
                 tokenizer,
                 bos_token_id=None,
                 eos_token_id=None,
                 max_length=512,
                 data_tokenized=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_tokenized = data_tokenized
        self.data_path = json_path

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
                    print(line)
                if type(data) != dict:
                    continue
                id = data['id']
                all_data.append((id, data))

        return all_data

    def __getitem__(self, item):
        sample_id = self.all_data[item][0]

        sample_id, source = self.all_data[item]
        source = source['conversations']

        def _addrole_masklabel_tokenize(source):
            '''add speaker and concatenate the sentences'''
            roles = {'human': 'Human: ', 'assistant': 'Assistant: '}
            conversation = ''
            input_ids = []
            labels = []
            test_input_ids = []
            for idx, sentence in enumerate(source):
                sentence_from = sentence["from"]
                sentence_value = roles[sentence_from] + sentence["value"] + \
                    '\n\nAssistant: ' if sentence_from == 'human' else sentence["value"]
                conversation += sentence_value
                sentence_ids = self.tokenizer.encode(
                    sentence_value, add_special_tokens=False)
                label = copy.deepcopy(sentence_ids) if sentence_from == 'assistant' else [
                    IGNORE_INDEX] * len(sentence_ids)

                # add eos at every end of assistant sentence
                if sentence_from == 'assistant':
                    sentence_ids += [self.tokenizer.eos_token_id]
                    label += [self.tokenizer.eos_token_id]

                input_ids += sentence_ids
                labels += label

                if idx < len(source) - 1:
                    test_input_ids += sentence_ids

            return input_ids, labels, test_input_ids, conversation

        input_ids, labels, test_input_ids, conversation = _addrole_masklabel_tokenize(
            source)

        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        # avoid all labels are -100, which will cause nan loss
        if not any(x > -100 for x in labels):
            labels[18:24] = input_ids[18:24]

        attention_mask = [1] * len(input_ids)

        # for inference
        input_input_ids = test_input_ids
        input_input_ids = input_input_ids[:self.max_length]

        input_attention_mask = [1] * len(input_input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


class GEN4ALLDatasetConvV2:
    def __init__(self,
                 json_path,
                 tokenizer,
                 bos_token_id=None,
                 eos_token_id=None,
                 max_length=1024,
                 data_tokenized=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.data_tokenized = data_tokenized
        self.data_path = json_path

        self.all_data = self._load_data()
        self.all_data_processed = []
        print("start tokenizer data")
        import time
        import gc
        start = time.time()
        for idx, data in enumerate(self.all_data):
            self.all_data_processed.append(self.processed_first(idx))
        print(f"tokenizer data time: {time.time() - start}")
        del self.all_data
        gc.collect()

        if self.eos_token_id is not None:
            print(
                f'Add eos token (id: {self.tokenizer.eos_token_id}) to the end of the input')
        if self.bos_token_id is not None:
            print(
                f'Add bos token (id: {self.tokenizer.bos_token_id}) to the beginning of the input')

        print("{}: the number of examples = {}".format(
            json_path, len(self.all_data_processed)))

    def __len__(self):
        return len(self.all_data_processed)

    def _load_data(self):
        all_data = []
        with open(self.data_path, mode="r") as reader:
            for line in tqdm(reader):
                try:
                    data = json.loads(line)
                except:
                    print(line)
                if type(data) != dict:
                    continue
                id = data['id']
                all_data.append((id, data))
        return all_data

    def processed_first(self, item):
        sample_id = self.all_data[item][0]

        sample_id, source = self.all_data[item]
        source = source['conversations']

        def _addrole_masklabel_tokenize(source):
            '''add speaker and concatenate the sentences'''
            roles = {'human': 'Human: ', 'assistant': 'Assistant: '}
            conversation = ''
            input_ids = []
            labels = []
            test_input_ids = []
            for idx, sentence in enumerate(source):
                sentence_from = sentence["from"]
                sentence_value = roles[sentence_from] + str(sentence["value"]) + \
                    '\n\nAssistant: ' if sentence_from == 'human' else str(
                        sentence["value"])
                conversation += sentence_value
                sentence_ids = self.tokenizer.encode(
                    sentence_value, add_special_tokens=False)
                label = copy.deepcopy(sentence_ids) if sentence_from == 'assistant' else [
                    IGNORE_INDEX] * len(sentence_ids)

                # add eos at every end of assistant sentence
                if sentence_from == 'assistant':
                    sentence_ids += [self.tokenizer.eos_token_id]
                    label += [self.tokenizer.eos_token_id]

                input_ids += sentence_ids
                labels += label

                if idx < len(source) - 1:
                    test_input_ids += sentence_ids

            return input_ids, labels, test_input_ids, conversation

        input_ids, labels, test_input_ids, conversation = _addrole_masklabel_tokenize(
            source)

        # if self.bos_token_id is not None:
        #     input_ids = [self.tokenizer.bos_token_id] + input_ids
        #     labels = [self.tokenizer.bos_token_id] + labels
        # if self.eos_token_id is not None:
        #     input_ids = input_ids + [self.tokenizer.eos_token_id]
        #     labels = labels + [self.tokenizer.eos_token_id]
        # print("self.max_length: ", self.max_length)
        # print("input_ids: ", input_ids)
        input_ids = input_ids[:self.max_length]
        labels = labels[:self.max_length]

        # avoid all labels are -100, which will cause nan loss
        if not any(x > -100 for x in labels):
            labels[18:24] = input_ids[18:24]

        attention_mask = [1] * len(input_ids)

        # for inference
        input_input_ids = test_input_ids
        input_input_ids = input_input_ids[:self.max_length]

        input_attention_mask = [1] * len(input_input_ids)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def __getitem__(self, item):
        return self.all_data_processed[item]



if __name__ == "__main__":

    print(__file__)
    parser = argparse.ArgumentParser(
        description="convert text to token ids {input_ids, attention_mask, labels}"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/code/tmp/train/project/design_gpt/data/train_design_gpt_and_qa_add3.jsonl",
        help="dataset path",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output path",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/data/big_shot_0320_epoch=1-step=194616",
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--seq_max_len",
        type=int,
        default=1536,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--task_type",
        type=int,
        default=0,
        help="0 mean sft  1 mean pretrained",
    )

    args = parser.parse_args()

    json_path = args.data_path
    tokenizer = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tk_name = args.tokenizer_path.split("/")[-1]
    # 按照输入文件名+tokenizer名字+convert作为输出文件名
    output_path = args.data_path+f"_{tk_name}_" + \
        f"task_{args.task_type}_"+".convert"
    print(f"output_path: {output_path}")
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens(
            {
                "pad_token": DEFAULT_PAD_TOKEN,
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    bos_token_id = None
    data_tokenized = False
    if args.task_type == 0:
        dataset = GEN4ALLSFTDatasetAccelerate(
            json_path=json_path,
            tokenizer=tokenizer,
            bos_token_id=None,
            eos_token_id=tokenizer.eos_token_id,
            max_length=args.seq_max_len,
            data_tokenized=False)


    for i in range(10):
        print(dataset.__getitem__(i))

    def write_json_lines(dataset, file_path):
        indices = list(range(len(dataset)))
        random.shuffle(indices)  # Shuffle the indices randomly

        with open(file_path, 'w') as file:
            for item in indices:
                data = dataset.__getitem__(item)
                json_string = json.dumps(data)
                file.write(json_string + '\n')

    write_json_lines(dataset, output_path)
