import sys
import os

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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name_or_path",
    type=str,
    default="/code/project/nlp2agi/output_sft_trainer_bloomz-7b/epoch_13",
)
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="/code/project/nlp2agi/output_sft_trainer_bloomz-7b/epoch_13",
)
parser.add_argument("--use_lora", action="store_true")
parser.add_argument("--llama", action="store_true")
args = parser.parse_args()


def generate_prompt(input_text):
    return "Human: \n" + input_text + "\n\nAssistant:\n"


def evaluate(
    input,
    temperature=0.01,
    top_p=0.75,
    top_k=30,
    num_beams=2,
    max_new_tokens=128,
    min_new_tokens=1,
    repetition_penalty=1.2,
    **kwargs,
):
    prompt = generate_prompt(input)
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
            do_sample=False,
            repetition_penalty=1.3,
        )
        output = generation_output.sequences[0]
        output = (
            tokenizer.decode(output, skip_special_tokens=True)
            .split("Assistant:")[1]
            .strip()
        )
        print(output)
        yield output


if __name__ == "__main__":
    load_type = torch.float16  # Sometimes may need torch.float32
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    if args.llama:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    tokenizer.padding_side = "left"
    model_config = AutoConfig.from_pretrained(args.model_name_or_path)

    print("Loading model...")
    if args.use_lora:
        base_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, torch_dtype=load_type
        )
        model = PeftModel.from_pretrained(
            base_model, args.ckpt_path, torch_dtype=load_type
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt_path, torch_dtype=load_type, config=model_config
        )

    if device == torch.device("cpu"):
        model.float()

    model.to(device)
    model.eval()
    print("Load model successfully")
    # https://gradio.app/docs/
    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Input", placeholder="Welcome to the BELLE model"
            ),
            gr.components.Slider(minimum=0, maximum=1, value=0.3, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=1, step=3, value=2, label="Beams Number"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=10, value=1024, label="Max New Tokens"
            ),
            gr.components.Slider(
                minimum=1, maximum=300, step=10, value=1, label="Min New Tokens"
            ),
            gr.components.Slider(
                minimum=1.0,
                maximum=2.0,
                step=0.1,
                value=1.2,
                label="Repetition Penalty",
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=25,
                label="Output",
            )
        ],
        title="BELLE: Be Everyone's Large Language model Engine",
    ).queue().launch(share=False, server_name="0.0.0.0", server_port=8006)
