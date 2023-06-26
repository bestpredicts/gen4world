#!/bin/bash
export CUDA_VISIBLE_DEVICES=3 
LOG_DIR=./ 
MODEL_PATH=/code/project/nlp2agi/output_sft_trainer_bloomz-7b/epoch_13 
nohup python -u llm_app.py --model_name_or_path=$MODEL_PATH > $LOG_DIR/log.txt 2>&1 &