from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.utils import logger
from huggingface_hub import snapshot_download
import mdtex2html
import gradio as gr
import argparse
import warnings
import torch
import os


logger.setLevel("ERROR")
warnings.filterwarnings("ignore")




import sys
import os
from flask import Flask, request, jsonify

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

sys.path.insert(0, os.path.dirname(__file__))  # seq2seq package path
sys.path.insert(0, os.getcwd())
import torch
import json
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    GenerationConfig,
)
from peft import PeftModel
import argparse
from tqdm import tqdm
import gradio as gr
import json, os
from flask import Flask, request
import logging


import logging
import logging.handlers
import os
import socket
from logging.handlers import TimedRotatingFileHandler

from pythonjsonlogger import jsonlogger

# 日志记录提效，不记录的字段无需计算
logging._srcfile = None
logging.logThreads = 0
logging.logMultiprocessing = 0
os.environ['LOG_DIR'] = "./"


def log_init(log_name):
    """
    项目日志记录器初始化
    """
    # 获取启动IP
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)

    logger = logging.getLogger(log_name)
    log_path = os.path.join(os.getenv('LOG_DIR', "./log"), log_name)
    log_file = os.path.join(log_path, log_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.setLevel(logging.DEBUG)
    # 日志输出格式和路径定义
    log_keys = ['process', 'asctime', 'message', 'levelname']
    custom_format = ' '.join(['%({0:s})'.format(i) for i in log_keys])
    formatter = jsonlogger.JsonFormatter(custom_format, json_ensure_ascii=False, prefix=str(ip) + '#')

    file_handler = TimedRotatingFileHandler(log_file, when='midnight', interval=1, backupCount=60)
    file_handler.setFormatter(formatter)
    file_handler.suffix = "%Y-%m-%d.log"
    logger.addHandler(file_handler)

    return logger


logger = log_init("model")


app = Flask(__name__)

# 配置日志记录
logging.basicConfig(
    filename="belle.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="/code/project/nlp2agi/output_sft_accelerate_baichuan-7B",
)

args = parser.parse_args()




def evaluate(
    input,
    temperature=0.5,
    top_p=0.75,
    top_k=0,
    num_beams=1,
    max_new_tokens=2048,
    min_new_tokens=1,
    **kwargs,
):
    prompt = input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,  # max_length=max_new_tokens+input_sequence
        min_new_tokens=min_new_tokens,  # min_length=min_new_tokens+input_sequence
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=True,
            repetition_penalty=1.0
        )
        output = generation_output.sequences[0]
        output = (
            tokenizer.decode(output, skip_special_tokens=True)
            .split("Assistant:")[-1]
            .strip()
        )
        return output


load_type = torch.float16  # Sometimes may need torch.float32
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device("cpu")

    
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    

tokenizer.padding_side = "left"
model_config = AutoConfig.from_pretrained(args.model_name_or_path,trust_remote_code=True)


model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=load_type, config=model_config,trust_remote_code=True,device_map='auto')
    

# model.to(device)
model.eval()
print("Load model successfully")



"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    query = parse_text(input)
    chatbot.append((query, ""))
    prompt =  ""
    for i, (old_query, response) in enumerate(history):
        prompt += 'Human: \n' + old_query + '\n\nAssistant:\n'+response


    prompt += 'Human: \n' + query + '\n\nAssistant:\n'
    # print(f"prompt is {prompt} ")

    response = evaluate(
            prompt,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            do_sample=True,
            temperature=1.0)
    

    # print(f"response is {response} ")

    logger.info(f"input: {prompt} \t output: {response} ")

            
    chatbot[-1] = (query, parse_text(response.replace("Human: \n", "")))
    history = history + [(query, response)]
    # print(f"chatbot is {chatbot}")
    # print(f"history is {history}")

    return chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">欢迎使用 BELLE-BaiChuan-SFT ！</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 2048, value=2048, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0, 1, value=0.7, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, inbrowser=True, server_name="0.0.0.0", server_port=8006)
